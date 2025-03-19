import pickle
import gymnasium as gym
import numpy as np

with open('initial_trajs.pkl', 'rb') as file:
    new_trajectories = pickle.load(file)

print(new_trajectories[1])

env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False)
ori_state, _ = env.reset()
init_state = new_trajectories[1]['observations'][0]
# init_pos = np.concatenate([ori_state[:1], init_state[:8]], axis=0)
init_pos = init_state[:9]
init_vel = init_state[9:]

env.set_state(init_pos, init_vel)
for i in range(len(new_trajectories[1]['actions'])//2):
    act = new_trajectories[1]['actions'][i]
    ns, r, d, _, _ = env.step(act)
    print(ns, r, d)
    if i+1 < len(new_trajectories[1]['observations']):
        print(new_trajectories[1]['observations'][i+1])
    print(new_trajectories[1]['rewards'][i])