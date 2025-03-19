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

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

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
        init_states, init_pos_vels, u_inits = [], [], []
        for eval_fn in self.eval_fns:
            outputs, _init_state, _init_pos_vels, _u_inits = eval_fn(self.model)
            init_states.extend(_init_state)
            init_pos_vels.extend(_init_pos_vels)
            u_inits.extend(_u_inits)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

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
