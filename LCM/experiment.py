import gymnasium as gym
import numpy as np
import torch, os, pickle
from torch.utils.tensorboard import SummaryWriter

import argparse
import ray, time
from datetime import datetime
import random
from tqdm import tqdm

from trajectory_optimizer.MPPI import mppi_runner
from trajectory_optimizer.PI2 import pi2_runner, update_policy_after_exploration, get_new_trajectories
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.value_func import ValueFunction
from decision_transformer.training.pi_func import PolicyFunction
from utils import discount_cumsum, get_env_properties, process_new_trajectories, process_trajectories, LinearScheduler

   
def experiment(
        exp_prefix,
        variant,
):
    # ray.init()
    ray.init(log_to_driver=False)
    device = variant.get('device', 'cuda')

    # region set exp name
    env_name = variant['env']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}'
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    exp_prefix = f'{group_name}-{timestamp}'
    traj_optimizer = variant['traj_optimizer']
    # endregion

    # region get env setup
    env = gym.make(env_name)
    max_ep_len = env.spec.max_episode_steps
    env_targets, scale = get_env_properties(env_name)

    # if model_type == 'bc':
    #     env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # endregion

    # region register the central model and its optimizer
    K = variant['K']
    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
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

    # init a value/policy function
    if traj_optimizer == "MPPI":
        value_function = ValueFunction(state_dim, device)
    if traj_optimizer == "PI2":
        policy_function = PolicyFunction(state_dim, act_dim, device)
    # endregion

    # region get the initial dataset for training the DT
    s_time = time.time()
    worker_num = variant.get('worker_num')

    if traj_optimizer == "MPPI":
        local_workers = [mppi_runner.remote() for _ in range(worker_num)]
        new_trajectories = ray.get([local_worker.run.remote(env_name) for local_worker in local_workers])
    elif traj_optimizer == "PI2":
        lam_scheduler = LinearScheduler(100.0, 10.0, 20)
        local_workers = [pi2_runner.remote() for _ in range(worker_num)]
        exp_num_samples = 10000 // worker_num
        run_num_samples = 1000 // worker_num
        state_mean=np.zeros(state_dim, dtype=np.float32)
        state_std=np.ones(state_dim, dtype=np.float32)

        policy_function.policy_network.to("cpu")
        ## phase 1
        print("Exploration is going on ......")
        exp_noise_scheme = None
        for _ in tqdm(range(10)):
            ray.get([local_worker.set_policy_function.remote(env_name, policy_function.policy_network) for local_worker in local_workers])
            exploration_results = ray.get([local_worker.exploration.remote(env_name, state_mean, state_std, exp_num_samples, exp_noise_scheme) for local_worker in local_workers])
            # exp_noise_scheme = update_policy_after_exploration(exploration_results, policy_function, lam=2.0) # TODO: do not update the policy function here, hyper
            update_policy_after_exploration(exploration_results, policy_function, lam=100.0)

        ## phase 2
        ray.get([local_worker.set_policy_function.remote(env_name, policy_function.policy_network) for local_worker in local_workers])
        print("Collecting trajectories for distillation ......")
        new_trajectories_list = ray.get([local_worker.run.remote(env_name, state_mean, state_std, run_num_samples) for local_worker in local_workers])
        new_trajectories = get_new_trajectories(new_trajectories_list)
        policy_function.policy_network.to(device)
    else:
        raise NotImplementedError
    
    print("Time spent to collect trajectories at the initial stage: {} mins".format((time.time()-s_time)/60.0))

    # with open('initial_trajs.pkl', 'wb') as file:
    #     pickle.dump(new_trajectories, file, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('initial_trajs.pkl', 'rb') as file:
    #     new_trajectories = pickle.load(file)

    # dataset format
    # print(len(trajectories), trajectories[0].keys()) # 10, ['observations', 'next_observations', 'actions', 'rewards', 'dones']
    # print(trajectories[0]['observations'].shape, trajectories[0]['actions'].shape, trajectories[0]['rewards'].shape) # (1000, 17) (1000, 6) (1000,)
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
    pct_traj = variant.get('pct_traj', 1.)
    max_iters = variant['max_iters']
    trajectories = []

    for iter in range(max_iters):
        # process new trajectories
        local_trajs_info = process_new_trajectories(mode, new_trajectories) 

        trajectories.extend(new_trajectories)
        if len(trajectories) > traj_set_size:
            trajectories = trajectories[-traj_set_size:] # TODO: pop elements based on their returns
        
        num_trajectories, p_sample, sorted_inds, state_mean, state_std = process_trajectories(pct_traj, trajectories)

        # region an ugly coding style inheritaged from DT
        def get_batch(batch_size=256, max_len=K):
            batch_inds = np.random.choice(
                np.arange(num_trajectories),
                size=batch_size,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for i in range(batch_size):
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
                si = random.randint(0, traj['rewards'].shape[0] - 1)

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
                s[-1] = (s[-1] - state_mean) / state_std
                s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
                a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
                r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

            s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
            a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
            r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
            d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
            mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # torch.Size([64, 20])

            return s, a, r, d, rtg, timesteps, mask
        
        def eval_episodes(target_rew):
            def fn(model, eval_mode):
                returns, lengths = [], []
                init_states, init_pos_vels, u_inits, traj_states, traj_actions, traj_rwds, traj_dones = [], [], [], [], [], [], [] # required by local trajectory optimization
                for _ in tqdm(range(num_eval_episodes)):
                    with torch.no_grad():
                        if model_type == 'dt':
                            ret, length, init_state, init_pos_vel, u_init, traj_state, traj_action, traj_rwd, traj_done = evaluate_episode_rtg(
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
                                eval_mode=eval_mode
                            )
                        else:
                            ret, length, init_state, init_pos_vel, u_init, traj_state, traj_action, traj_rwd, traj_done = evaluate_episode(
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
                                eval_mode=eval_mode
                            )
                    returns.append(ret)
                    lengths.append(length)
                    init_states.append(init_state)
                    init_pos_vels.append(init_pos_vel)
                    u_inits.append(u_init)
                    traj_states.append(traj_state)
                    traj_actions.append(traj_action)
                    traj_rwds.append(traj_rwd)
                    traj_dones.append(traj_done)
                return {
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                }, init_states, init_pos_vels, u_inits, traj_states, traj_actions, traj_rwds, traj_dones
            return fn
        
        if model_type == 'dt':
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )
        elif model_type == 'bc':
            trainer = ActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )
        # endregion

        # main function
        # num_steps = variant['num_steps_per_iter'] * min((iter + 1), 10)
        num_steps = variant['num_steps_per_iter']

        if traj_optimizer == "MPPI":
            outputs, init_states, init_pos_vels, u_inits = trainer.train_iteration(value_function=value_function, state_mean=state_mean, state_std=state_std, \
                                                                                   num_steps=num_steps, iter_num=iter+1, print_logs=True)
        elif traj_optimizer == "PI2":
            outputs, _, _, _ = trainer.train_iteration(policy_function=policy_function, state_mean=state_mean, state_std=state_std, \
                                                                                   num_steps=num_steps, iter_num=iter+1, print_logs=True)
        else:
            raise NotImplementedError
        
        # log the infos
        for k, v in outputs.items():
            writer.add_scalar(k, v, global_step=iter)
        for k, v in local_trajs_info.items():
            writer.add_scalar(k, v, global_step=iter)
        
        # update the trajectory set
        if iter < max_iters - 1:
            s_time = time.time()

            if traj_optimizer == "MPPI":
                value_function.value_network.to("cpu")
                ray.get([local_worker.set_value_function.remote(env_name, value_function.value_network) for local_worker in local_workers])
                new_trajectories = ray.get([local_workers[i].run.remote(env_name, init_states[i], init_pos_vels[i], u_inits[i], state_mean, state_std) for i in range(worker_num)])
                value_function.value_network.to(device)
            elif traj_optimizer == "PI2":
                policy_function.policy_network.to("cpu")
                ## phase 1
                print("Exploration is going on ......")
                exp_noise_scheme = None
                cur_lam = lam_scheduler.step()
                for _ in tqdm(range(10)):
                    ray.get([local_worker.set_policy_function.remote(env_name, policy_function.policy_network) for local_worker in local_workers])
                    exploration_results = ray.get([local_worker.exploration.remote(env_name, state_mean, state_std, exp_num_samples, exp_noise_scheme) for local_worker in local_workers])
                    # exp_noise_scheme = update_policy_after_exploration(exploration_results, policy_function, lam=2.0) # TODO: do not update the policy function here, hyper
                    update_policy_after_exploration(exploration_results, policy_function, lam=cur_lam)
                ## phase 2
                ray.get([local_worker.set_policy_function.remote(env_name, policy_function.policy_network) for local_worker in local_workers])
                new_trajectories_list = ray.get([local_worker.run.remote(env_name, state_mean, state_std, run_num_samples) for local_worker in local_workers])
                new_trajectories = get_new_trajectories(new_trajectories_list)
                policy_function.policy_network.to(device)
            else:
                raise NotImplementedError

            print("Time spent to collect trajectories at the training iteration {} is {} mins".format(iter, (time.time()-s_time)/60.0))
    
    ray.shutdown()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v4')
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64) # number of sampled trajectory segments (of length K)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    # important
    parser.add_argument('--worker_num', type=int, default=50) # 50
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000
    parser.add_argument('--num_eval_episodes', type=int, default=50) # 50
    parser.add_argument('--max_iters', type=int, default=100) # 10
    parser.add_argument('--num_steps_per_iter', type=int, default=10000) # 10000
    parser.add_argument('--traj_set_size', type=int, default=5000)
    parser.add_argument('--traj_optimizer', type=str, default='PI2')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()

    assert args.num_eval_episodes >= args.worker_num

    import warnings
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    experiment('gym-experiment', variant=vars(args))
