import gymnasium as gym
import numpy as np
import torch, os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import argparse
import ray, time
from datetime import datetime
import random
from tqdm import tqdm

from trajectory_optimizer.MPPI import mppi_runner
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def get_env_properties(env_name):
    if "Hopper" in env_name:
        env_targets = [3600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif "HalfCheetah" in env_name:
        env_targets = [15000]
        scale = 1000.
    elif "Walker" in env_name:
        env_targets = [5000]
        scale = 1000.
    else:
        raise NotImplementedError
    
    return env_targets, scale

def process_new_trajectories(mode, new_trajectories, state_dim, act_dim):
    # save all path information into separate lists
    returns = defaultdict(list)

    for idx in range(len(new_trajectories)):
        path, env_id = new_trajectories[idx]
        # padding the state and action
        traj_len = path['observations'].shape[0]
        real_state_dim = path['observations'].shape[1]
        real_act_dim = path['actions'].shape[1]
        state_pad = np.zeros((traj_len, state_dim), dtype=np.float32)
        act_pad = np.zeros((traj_len, act_dim), dtype=np.float32)
        state_pad[:, :real_state_dim] = path['observations'].copy()
        act_pad[:, :real_act_dim] = path['actions'].copy()
        path['observations'] = state_pad.copy()
        path['actions'] = act_pad.copy()

        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        returns[env_id].append(path['rewards'].sum())
        # important
        new_trajectories[idx] = (path, env_id)

    info = {}
    for k in returns:
        info[f"local_trajs/{k}/mean"] = np.mean(returns[k])
        info[f"local_trajs/{k}/std"] = np.std(returns[k])
        info[f"local_trajs/{k}/max"] = np.max(returns[k])
        info[f"local_trajs/{k}/min"] = np.min(returns[k])

    print('=' * 50)
    for k, v in info.items():
        print(f'{k}: {v:.2f}')
    print('=' * 50)

    return info


def process_trajectories(trajectories):
    # save all path information into separate lists
    traj_lens = []
    states_dict = defaultdict(list)
    for path, env_id in trajectories:
        states_dict[env_id].append(path['observations'])
        traj_lens.append(len(path['observations']))
    traj_lens = np.array(traj_lens)

    # used for input normalization
    state_mean_dict, state_std_dict = {}, {}
    for k in states_dict:
        states = np.concatenate(states_dict[k], axis=0) # (10000, 17)
        state_mean_dict[k] = np.mean(states, axis=0)
        state_std_dict[k] = np.std(states, axis=0) + 1e-6

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens / sum(traj_lens)

    return p_sample, state_mean_dict, state_std_dict
    

def experiment(
        exp_prefix,
        env_name_list,
        variant,
):
    # ray.init()
    ray.init(log_to_driver=False)
    device = variant.get('device', 'cuda')

    # region set exp name
    env_num = len(env_name_list)
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_num}'
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    exp_prefix = f'{group_name}-{timestamp}'
    # endregion

    # region get env setup
    env_list = [gym.make(env_name) for env_name in env_name_list]
    max_ep_len_list = [env.spec.max_episode_steps for env in env_list]
    max_max_ep_len = max(max_ep_len_list)

    target_num = variant['target_num']
    env_targets_dict, scale_dict = {}, {}
    for env_name in env_name_list:
        _target, _scale = get_env_properties(env_name)
        assert len(_target) == target_num
        env_targets_dict[env_name] = _target
        scale_dict[env_name] = _scale

    state_dim = max([env.observation_space.shape[0] for env in env_list])
    act_dim_list = [env.action_space.shape[0] for env in env_list] # can be restrictive
    act_dim = max(act_dim_list)

    act_dim_dict, max_ep_len_dict = {}, {}
    for env_idx in range(env_num):
        act_dim_dict[env_name_list[env_idx]] = act_dim_list[env_idx]
        max_ep_len_dict[env_name_list[env_idx]] = max_ep_len_list[env_idx]
    # endregion

    # region register the central model and its optimizer
    K = variant['K']
    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    # endregion

    # region get the initial dataset for training the DT
    s_time = time.time()
    worker_num = variant.get('worker_num')
    local_workers = [mppi_runner.remote() for _ in range(env_num * worker_num)]
    new_trajectories = ray.get([local_workers[i].run.remote(env_name_list[i//worker_num]) for i in range(len(local_workers))])
    print("Time spent to collect trajectories at the initial stage: {} mins".format((time.time()-s_time)/60.0))
    # endregion

    ############################################################################################################
    # main loop
    log_dir = f'./logs/{exp_prefix}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    traj_set_size = variant['traj_set_size']
    mode = variant.get('mode', 'normal')
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    max_iters = variant['max_iters']
    trajectories = []

    for iter in range(max_iters):
        # process new trajectories
        local_trajs_info = process_new_trajectories(mode, new_trajectories, state_dim, act_dim) 

        trajectories.extend(new_trajectories)
        while len(trajectories) > traj_set_size:
            trajectories.pop(0) # TODO: pop elements based on their returns
        num_trajectories = len(trajectories)
        print("The current size of the trajectory set is {}.".format(num_trajectories))
        
        p_sample, state_mean_dict, state_std_dict = process_trajectories(trajectories)

        # region an ugly coding style inheritaged from DT
        def get_batch(batch_size=256, max_len=K):
            batch_inds = np.random.choice(
                np.arange(num_trajectories),
                size=batch_size,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )

            s, a, r, d, rtg, timesteps, mask, action_mask = [], [], [], [], [], [], [], []
            for i in range(batch_size):
                traj, env_id = trajectories[batch_inds[i]]
                si = random.randint(0, traj['rewards'].shape[0] - 1)
                state_mean, state_std, max_ep_len, scale, real_act_dim\
                      = state_mean_dict[env_id], state_std_dict[env_id], max_ep_len_dict[env_id], scale_dict[env_id], act_dim_dict[env_id]

                # get sequences from dataset
                s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
                r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
                rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # print(s[-1].shape, a[-1].shape, r[-1].shape, d[-1].shape, rtg[-1].shape, timesteps[-1].shape)
                # (1, 20, 17) (1, 20, 6) (1, 20, 1) (1, 20) (1, 21, 1) (1, 20)

                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
                r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
                a_mask = np.zeros_like(a[-1])
                a_mask[:, :, :real_act_dim] = 1.0
                action_mask.append(a_mask.copy())

            s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
            a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
            r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
            d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
            mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # torch.Size([64, 20])
            action_mask = torch.from_numpy(np.concatenate(action_mask, axis=0)).to(device=device)

            return s, a, r, d, rtg, timesteps, mask, action_mask
        
        def eval_episodes(target_idx):
            def fn(model):
                init_states, init_pos_vels, u_inits = [], [], [] # required by local trajectory optimization
                return_info = {}

                for env_idx in range(env_num):
                    env = env_list[env_idx]
                    env_id = env_name_list[env_idx]
                    print("Running evaluations for the environment: {}......".format(env_id))
                    returns, lengths = [], []

                    state_mean, state_std, max_ep_len, scale, target_rew_list\
                          = state_mean_dict[env_id], state_std_dict[env_id], max_ep_len_dict[env_id], scale_dict[env_id], env_targets_dict[env_id]
                    target_rew = target_rew_list[target_idx]

                    for _ in tqdm(range(num_eval_episodes)):
                        with torch.no_grad():
                            if model_type == 'dt':
                                ret, length, init_state, init_pos_vel, u_init = evaluate_episode_rtg(
                                    env,
                                    state_dim,
                                    act_dim,
                                    model,
                                    max_ep_len=max_ep_len,
                                    scale=scale,
                                    target_return=target_rew/scale,
                                    mode=mode,
                                    state_mean=state_mean,
                                    state_std=state_std,
                                    device=device,
                                )
                            else:
                                ret, length, init_state, init_pos_vel, u_init = evaluate_episode(
                                    env,
                                    state_dim,
                                    act_dim,
                                    model,
                                    max_ep_len=max_ep_len,
                                    target_return=target_rew/scale,
                                    mode=mode,
                                    state_mean=state_mean,
                                    state_std=state_std,
                                    device=device,
                                )
                        returns.append(ret)
                        lengths.append(length)
                        init_states.append(init_state)
                        init_pos_vels.append(init_pos_vel)
                        u_inits.append(u_init)
                    
                    return_info[f'env_{env_id}_target_{target_rew}_return_mean'] = np.mean(returns)
                    return_info[f'env_{env_id}_target_{target_rew}_return_std'] = np.std(returns)
                    return_info[f'env_{env_id}_target_{target_rew}_length_mean'] = np.mean(lengths)
                    return_info[f'env_{env_id}_target_{target_rew}_length_std'] = np.std(lengths)

                return return_info, init_states, init_pos_vels, u_inits
            return fn
        
        if model_type == 'dt':
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(target_idx) for target_idx in range(target_num)],
            )
        elif model_type == 'bc':
            trainer = ActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(target_idx) for target_idx in range(target_num)],
            )
        # endregion

        # main function
        num_steps = variant['num_steps_per_iter'] * min((iter + 1), 10)
        outputs, init_states, init_pos_vels, u_inits = trainer.train_iteration(num_steps=num_steps, iter_num=iter+1, print_logs=True)

        # log the infos
        for k, v in outputs.items():
            writer.add_scalar(k, v, global_step=iter)
        for k, v in local_trajs_info.items():
            writer.add_scalar(k, v, global_step=iter)
        
        # update the trajectory set
        if iter < max_iters - 1:
            s_time = time.time()
            new_trajectories = ray.get([local_workers[i].run.remote(env_name_list[i//worker_num], init_states[i], init_pos_vels[i], u_inits[i]) for i in range(env_num * worker_num)])
            print("Time spent to collect trajectories at the training iteration {} is {} mins".format(iter, (time.time()-s_time)/60.0))
    
    ray.shutdown()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--target_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64) # number of sampled trajectory segments (of length K)
    parser.add_argument('--model_type', type=str, default='bc')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    # important
    parser.add_argument('--worker_num', type=int, default=20) # number of workers for each env
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000
    parser.add_argument('--num_eval_episodes', type=int, default=20) # number of eval_episodes for each env
    parser.add_argument('--max_iters', type=int, default=100) # 10
    parser.add_argument('--num_steps_per_iter', type=int, default=10000) # 10000
    parser.add_argument('--traj_set_size', type=int, default=5000)

    parser.add_argument('--device', type=str, default='cuda:3')
    
    args = parser.parse_args()

    env_name_list = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
    assert args.num_eval_episodes >= args.worker_num

    import warnings
    warnings.filterwarnings("ignore")

    experiment('gym-experiment', env_name_list, variant=vars(args))
