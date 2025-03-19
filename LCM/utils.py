import numpy as np

class LinearScheduler:
    def __init__(self, start_value, end_value, num_intervals):
        self.start_value = start_value
        self.end_value = end_value
        self.num_intervals = num_intervals
        self.interval_value_drop = (start_value - end_value) / num_intervals
        self.current_value = start_value
        self.current_interval = 0

    def step(self):
        if self.current_interval < self.num_intervals:
            self.current_value -= self.interval_value_drop
            self.current_interval += 1
        else:
            self.current_value = self.end_value
        return self.current_value

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

def process_new_trajectories(mode, new_trajectories):
    # save all path information into separate lists
    returns = []
    for path in new_trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        returns.append(path['rewards'].sum())
    returns = np.array(returns)

    info = {"local_trajs/mean": np.mean(returns), "local_trajs/std": np.std(returns),
            "local_trajs/max": np.max(returns), "local_trajs/min": np.min(returns)}
    print('=' * 50)
    for k, v in info.items():
        print(f'{k}: {v:.2f}')
    print('=' * 50)

    return info

def process_trajectories(pct_traj, trajectories):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0) # (10000, 17)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    return num_trajectories, p_sample, sorted_inds, state_mean.astype(np.float32), state_std.astype(np.float32)
 