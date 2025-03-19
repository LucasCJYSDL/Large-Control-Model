import ray
import torch, time
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from collections import defaultdict

from decision_transformer.training.pi_func import PolicyFunction
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from utils import discount_cumsum


def update_policy_after_exploration(exploration_results, policy_function, lam):
    rwd_trajectories = []
    parameter_dict = defaultdict(list)
    parameter_diff_dict = defaultdict(list)

    for rwd_trajs, parameter_dicts, parameter_diff_dicts in exploration_results:
        rwd_trajectories.extend(rwd_trajs)
        for _parameter_dict in parameter_dicts:
            for k, v in _parameter_dict.items():
                parameter_dict[k].append(v)
        ## danger
        for _parameter_diff_dict in parameter_diff_dicts:
            for k, v in _parameter_diff_dict.items():
                parameter_diff_dict[k].append(torch.square(v))

    for k in parameter_dict:
        parameter_dict[k] = torch.stack(parameter_dict[k], dim=0)
        ## danger
        parameter_diff_dict[k] = torch.stack(parameter_diff_dict[k], dim=0)
    
    max_length = -1
    member_num = len(rwd_trajectories)
    rtg_trajectories = []
    traj_rets = []
    for idx in range(member_num):
        rtg_trajectories.append(discount_cumsum(np.array(rwd_trajectories[idx]), gamma=1.0))
        traj_rets.append(rtg_trajectories[-1][0])
        if len(rwd_trajectories[idx]) > max_length:
            max_length = len(rwd_trajectories[idx])
    
    print("During exploration, the returns of the trajectories have a mean {}, a std {}, a max {}, a min {}.".format(np.mean(traj_rets), np.std(traj_rets), np.max(traj_rets), np.min(traj_rets)))
    
    # get the weights based on the PI rule
    P = np.zeros((max_length, member_num), dtype=np.float32)
    for i in range(max_length):
        min_v, max_v = np.inf, -np.inf
        for k in range(member_num):
            if len(rtg_trajectories[k]) > i:
                if rtg_trajectories[k][i] < min_v:
                    min_v = rtg_trajectories[k][i]
                if rtg_trajectories[k][i] > max_v:
                    max_v = rtg_trajectories[k][i]
        for k in range(member_num):
            if len(rtg_trajectories[k]) > i:
                P[i][k] = np.exp((rtg_trajectories[k][i]-min_v)/(max_v-min_v+1e-3)/lam)

        P[i] = P[i] / (np.sum(P[i]) + 1e-6)
        
    # get the average weights for the policy net
    weights_for_time_step = np.array([max_length-i for i in range(max_length)], dtype=np.float32)
    weights_for_time_step = weights_for_time_step / np.sum(weights_for_time_step)
    final_weights = torch.tensor(P * weights_for_time_step[:, np.newaxis], dtype=torch.float32)

    # print(final_weights.sum(dim=1), final_weights.sum(dim=1).shape)
    for k in parameter_dict:
        parameter_dict[k] = torch.sum(torch.matmul(parameter_dict[k].T, final_weights.T).T, dim=0)
        ## danger
        parameter_diff_dict[k] = torch.sum(torch.matmul(parameter_diff_dict[k].T, final_weights.T).T, dim=0)

    set_policy_parameters(policy_function, parameter_dict)

    return parameter_diff_dict
    

def set_policy_parameters(policy_function, parameter_dict):
    with torch.no_grad():
        for name, param in policy_function.policy_network.named_parameters():
            param.copy_(parameter_dict[name])


def collect_trajectory(env_id, actor, state_mean, state_std):
    env = gym.make(env_id)
    max_epi_length = env.spec.max_episode_steps
    state, _ = env.reset()
    rwd_trajectory = []

    done = False
    i = 0
    while not done and i < max_epi_length:
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
        action = actor.predict((state_tensor-state_mean)/state_std)
        next_state, reward, done, _, _ = env.step(action[0])
        rwd_trajectory.append(reward)
        state = next_state
        i += 1

    env.close()
    return rwd_trajectory


def collect_trajectories(num_envs, envs, actor, state_mean, state_std, max_epi_length):
    trajs = [{'observations': [], 'actions': [], 'rewards': [], 'dones': []} for _ in range(num_envs)]

    state_mean = torch.tensor(state_mean, dtype=torch.float32)
    state_std = torch.tensor(state_std, dtype=torch.float32)

    states = envs.reset()
    mask = np.ones(num_envs, dtype=bool)
    l = 0
    while True:
        actions = actor.predict((torch.tensor(states, dtype=torch.float32)-state_mean)/state_std)    
        next_states, rewards, dones, _ = envs.step(actions) 

        for i in range(num_envs):
            if mask[i]:
                trajs[i]['observations'].append(states[i])
                trajs[i]['actions'].append(actions[i])
                trajs[i]['rewards'].append(rewards[i])
                trajs[i]['dones'].append(dones[i])  

        mask = np.logical_and(mask, np.logical_not(dones))
        if not mask.any():
            break
        
        l += 1
        if l >= max_epi_length:
            break

        states = next_states
    
    for traj in trajs:
        for k in traj:
            traj[k] = np.array(traj[k])
    
    return trajs


def get_new_trajectories(new_trajectories_list):
    new_trajectories = []
    for t_list in new_trajectories_list:
        new_trajectories.extend(t_list)
    return new_trajectories


def get_parameter_diff(current_parameters, initial_parameters):
    return {name: param - initial_parameters[name] for name, param in current_parameters.items()}


@ray.remote
class pi2_runner:
    def __init__(self):
        # Parameters
        self.noise_scale = 0.1
        self.policy_function = None
    
    def set_policy_function(self, env_id, center_policy_function):
        env = gym.make(env_id)
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.policy_function = PolicyFunction(state_dim, act_dim)
        self.policy_function.update_model(center_policy_function)
        env.close()
    
    def get_policy_parameters(self):
        return {name: param.clone() for name, param in self.policy_function.policy_network.named_parameters()}
    
    def exploration(self, env_id, state_mean, state_std, num_samples, noise_scheme):
        # Store the initial parameters of the actor network
        initial_parameters = self.get_policy_parameters()

        # Collect several trajectories
        rwd_trajectories, parameter_dicts, parameter_diff_dicts = [], [], []
        for _ in tqdm(range(num_samples)):
            # Perturb the actor network
            with torch.no_grad():
                # for param in self.policy_function.policy_network.parameters():
                #     param += torch.randn_like(param) * self.noise_scale  ## Example perturbation
                ## danger
                for name, param in self.policy_function.policy_network.named_parameters():
                    if noise_scheme is None:
                        param += torch.randn_like(param) * self.noise_scale
                    else:
                        param += torch.randn_like(param) * torch.sqrt(noise_scheme[name])
            # Collect a trajectory
            rwd_trajectory = collect_trajectory(env_id, self.policy_function, state_mean, state_std)
            rwd_trajectories.append(rwd_trajectory)
            parameter_dicts.append(self.get_policy_parameters())
            parameter_diff_dicts.append(get_parameter_diff(parameter_dicts[-1], initial_parameters))
            # Reset the actor network to its initial parameters
            set_policy_parameters(self.policy_function, initial_parameters)

        return rwd_trajectories, parameter_dicts, parameter_diff_dicts

    def run(self, env_id, state_mean, state_std, num_samples):
        # Create vectorized environments
        def make_env():
            def _init():
                env = gym.make(env_id)
                return env
            return _init

        envs = DummyVecEnv([make_env() for _ in range(num_samples)])
        env = gym.make(env_id)
        max_epi_length = env.spec.max_episode_steps
        trajectories = collect_trajectories(num_samples, envs, self.policy_function, state_mean, state_std, max_epi_length)
        
        env.close()
        envs.close()
        self.policy_function = None

        return trajectories

