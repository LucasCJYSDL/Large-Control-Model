import numpy as np
import gymnasium as gym
import time, ray, torch
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from decision_transformer.training.value_func import ValueFunction

def parse_env_state(env):
    env_qpos = env.data.qpos.flat.copy()
    env_qvel = env.data.qvel.flat.copy()
    return env_qpos, env_qvel

def mppi(init_state, init_pos_vel, u_init, env, envs, num_envs, total_steps, num_samples, 
         horizon, lam, state_dim, action_dim, action_low, action_high, value_function, state_mean, state_std):
    """
    MPPI algorithm implementation.
    """
    traj = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
    # important schedules
    u = u_init.copy()[:horizon]
    noise_covariance = np.array([np.eye(action_dim) for _ in range(horizon)])

    env_state, _ = env.reset()
    ## This initial state should come from the center agent.
    if init_pos_vel is not None:
        env.set_state(init_pos_vel[0], init_pos_vel[1])
        env_state = init_state
    
    env_return = 0.0

    if state_mean is not None:
        state_mean = torch.tensor(state_mean, dtype=torch.float32)
        state_std = torch.tensor(state_std, dtype=torch.float32)

    # Main loop
    print("MPPI is going on ......")
    for t_step in tqdm(range(total_steps)):
        traj['observations'].append(env_state.copy())
        noise = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, horizon, action_dim))
        ## danger: generate noise based on the covirance
        # noise = []
        # for h_idx in range(horizon):
        #     noise.append(np.random.multivariate_normal(mean=np.zeros(action_dim), cov=noise_covariance[h_idx], size=1000)) # (1000, 6)
        # noise = np.stack(noise, axis=1) # (1000, 20, 6)

        actions = u[None, :, :] + noise  # Shape: (num_samples, horizon, action_dim)
        ## Clip actions to action space
        actions = np.clip(actions, action_low, action_high)
        noise = actions - u[None, :, :] # TODO: using this
        ## print(u.shape, actions.shape, noise.shape) # (20, 6) (1000, 20, 6) (1000, 20, 6)

        # start the rollout
        assert num_samples % num_envs == 0
        num_batches = num_samples // num_envs
        env_qpos, env_qvel = parse_env_state(env)
        ## some import quantities to track
        total_cost = np.zeros(num_samples, dtype=np.float32)
        dones = np.zeros(num_samples, dtype=np.float32)
        final_states = np.zeros((num_samples, state_dim), dtype=np.float32)

        for batch_idx in range(num_batches):
            ## Reset environments and set initial states
            envs.reset()
            envs.env_method('set_state', env_qpos, env_qvel)
            # datas = envs.get_attr('data')

            mask = np.ones(num_envs, dtype=bool)
            s_id, e_id = batch_idx * num_envs, (batch_idx+1) * num_envs
            batch_actions = actions[s_id : e_id]
            for h in range(horizon):
                a = batch_actions[:, h, :]
                s, r, d, _ = envs.step(a) # (num_envs, state_dim), (num_envs, ), (num_envs, )
                ## important updates
                final_states[s_id : e_id][mask] = s[mask]
                total_cost[s_id : e_id][mask] -= r[mask]
                dones[s_id : e_id][mask] = d[mask]
                ## once an env terminates, it would be masked util the end of the horizon
                mask = np.logical_and(mask, np.logical_not(d))
                if not mask.any():
                    break
            
        ## There should be an optional step to update the total_cost based on final_states, dones, and the (negative) value function.
        if value_function is not None:
            final_states = (torch.from_numpy(final_states) - state_mean) / state_std
            values = value_function.predict(final_states)
            total_cost -= (1.0 - dones) * values

        # Compute weights
        beta = np.min(total_cost)
        weights = np.exp(-1 / lam * (total_cost - beta))
        weights /= np.sum(weights)

        # Update control sequence
        weighted_noise = np.einsum('i,ijk->jk', weights, noise) # (1000,) (1000, 20, 6) (20, 6)
        u += weighted_noise # TODO: clip this
        u = np.clip(u, action_low, action_high)
        ## danger
        # covariance = np.einsum('...i,...j->...ij', noise, noise) # (1000, 20, 6, 6)
        # weighted_covariance = np.tensordot(weights, covariance, axes=1) # (20, 6, 6)
        
        # Execute first action
        action_to_take = u[0]
        env_state, env_reward, env_done, _, _ = env.step(action_to_take)
        env_return += env_reward
        traj['actions'].append(action_to_take.copy())
        traj['rewards'].append(env_reward)
        traj['dones'].append(env_done)
        if env_done:
            break

        # Shift control sequence
        u = np.roll(u, -1, axis=0)
        if t_step + horizon < len(u_init):
            u[-1] = u_init[t_step + horizon]
        else:
            u[-1] = 0.0  # danger
        ## danger
        # noise_covariance = np.roll(weighted_covariance, -1, 0)
        # noise_covariance[-1] = np.eye(action_dim)

        # Optional: Render one of the environments
        # if t_step % 10 == 0:
        #     envs.env_method('render', indices=0)
    
    print("The final return: ", env_return)

    # post process of the trajectory
    # if t_step >= total_steps - 1:
    #     traj['dones'][-1] = True # danger: a little hacky since we view the timeout case as done
    for k in traj:
        traj[k] = np.array(traj[k])
    
    return traj

@ray.remote
class mppi_runner:

    def __init__(self):
        # Parameters
        self.num_envs = 100        # Number of parallel environments
        self.horizon = 20          # Prediction horizon
        self.num_samples = 1000    # Number of action sequences to sample
        self.lam = 2.0             # Temperature parameter for MPPI
        self.value_function = None
    
    def set_value_function(self, env_id, center_value_function):
        env = gym.make(env_id)
        state_dim = env.observation_space.shape[0]
        self.value_function = ValueFunction(state_dim)
        self.value_function.update_model(center_value_function)
        env.close()
    
    def run(self, env_id, init_state=None, init_pos_vel=None, u_init=None, state_mean=None, state_std=None):
        # Create vectorized environments
        def make_env():
            def _init():
                env = gym.make(env_id)
                return env
            return _init

        # envs = SubprocVecEnv([make_env() for _ in range(self.num_envs)], start_method="fork")
        envs = DummyVecEnv([make_env() for _ in range(self.num_envs)]) # faster when using multiple cores

        # Get action and state dimensions
        env = gym.make(env_id)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
        total_steps = env.spec.max_episode_steps # Total steps to run the simulation
        # print(total_steps, state_dim, action_dim, action_low, action_high) # 1000 17 6 [-1. -1. -1. -1. -1. -1.] [1. 1. 1. 1. 1. 1.]

        # Initialize control sequence
        u_init_pad = np.zeros((total_steps, action_dim))
        if u_init is not None:
            u_init_pad[:len(u_init)] += u_init.copy()
        u_init = u_init_pad

        traj = mppi(init_state, init_pos_vel, u_init, env, envs, self.num_envs, total_steps, self.num_samples, \
                    self.horizon, self.lam, state_dim, action_dim, action_low, action_high, self.value_function, state_mean, state_std)

        env.close()
        envs.close()
        self.value_function = None

        return traj
    

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    ray.init()
    test_runner = mppi_runner.remote()
    new_trajectories = ray.get([test_runner.run.remote(env_id='HalfCheetah-v4')])
    print(new_trajectories[0]["rewards"])
