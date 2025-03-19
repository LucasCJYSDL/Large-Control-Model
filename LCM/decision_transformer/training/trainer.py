import numpy as np
import torch
from tqdm import tqdm
import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, policy_function=None, value_function=None, state_mean=None, state_std=None, num_steps=1000, iter_num=0, print_logs=False):
        # preparation
        train_losses = []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _ in tqdm(range(num_steps)):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        init_states, init_pos_vels, u_inits, traj_states, traj_actions, traj_rwds, traj_dones = [], [], [], [], [], [], []
        for eval_fn in self.eval_fns:
            outputs, _init_state, _init_pos_vels, _u_inits, _traj_states, _traj_actions, _traj_rwds, _traj_dones = eval_fn(self.model, eval_mode=True)
            init_states.extend(_init_state)
            init_pos_vels.extend(_init_pos_vels)
            u_inits.extend(_u_inits)
            traj_states.extend(_traj_states)
            traj_actions.extend(_traj_actions)
            traj_rwds.extend(_traj_rwds)
            traj_dones.extend(_traj_dones)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        # train the value function
        if value_function is not None or policy_function is not None:
            for eval_fn in self.eval_fns: # TODO: using train mode data only
                _, _, _, _, _traj_states, _traj_actions, _traj_rwds, _traj_dones = eval_fn(self.model, eval_mode=False)
                traj_states.extend(_traj_states)
                traj_actions.extend(_traj_actions)
                traj_rwds.extend(_traj_rwds)
                traj_dones.extend(_traj_dones)
        
        if value_function is not None:
            v_loss = value_function.train({'observations': traj_states, 'rewards': traj_rwds, 'dones': traj_dones}, state_mean, state_std)
            logs['value_loss'] = v_loss
        if policy_function is not None:
            p_loss = policy_function.train({'observations': traj_states, 'actions': traj_actions}, state_mean, state_std)
            logs['policy_loss'] = p_loss

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs, init_states, init_pos_vels, u_inits

    def train_step(self):
        pass
