import numpy as np
import torch
from tqdm import tqdm


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        eval_mode=True
):
    if eval_mode:
        model.eval()
    else: # TODO: add action noise
        model.train()
    model.to(device=device)

    traj_states, traj_actions, traj_rwds, traj_dones = [], [], []

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    traj_states.append(state.copy())
    # get info for local trajectory optimization
    init_state = state.copy()
    init_pos_vel = [env.data.qpos.flat.copy(), env.data.qvel.flat.copy()]
    u_init = []

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        u_init.append(action)
        state, reward, done, _, _ = env.step(action)

        traj_states.append(state.copy())
        traj_actions.append(action.copy())
        traj_rwds.append(reward)
        traj_dones.append(done)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        # states = torch.cat([states, cur_state], dim=0)
        states = cur_state
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length, init_state, init_pos_vel, np.array(u_init), traj_states[:-1], traj_actions, traj_rwds, traj_dones


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        eval_mode=True
    ):

    if eval_mode:
        model.eval()
    else: # TODO: add action noise
        model.train()
    model.to(device=device)

    traj_states, traj_actions, traj_rwds, traj_dones = [], [], [], []

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    traj_states.append(state.copy())
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # get info for local trajectory optimization
    init_state = state.copy()
    init_pos_vel = [env.data.qpos.flat.copy(), env.data.qvel.flat.copy()]
    u_init = []

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    # print(actions.shape, rewards.shape, target_return.shape) # torch.Size([0, 6]) torch.Size([0]) torch.Size([1, 1])

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        u_init.append(action)
        state, reward, done, _, _ = env.step(action)

        traj_states.append(state.copy())
        traj_actions.append(action.copy())
        traj_rwds.append(reward)
        traj_dones.append(done)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length, init_state, init_pos_vel, np.array(u_init), traj_states[:-1], traj_actions, traj_rwds, traj_dones
