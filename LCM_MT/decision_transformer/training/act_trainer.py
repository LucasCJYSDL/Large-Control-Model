import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class ActTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, _, attention_mask, action_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        act_dim = action_preds.shape[-1]
        action_preds = (action_preds * action_mask.reshape(-1, act_dim))[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward() # TODO: apply gradient clip
        self.optimizer.step()

        return loss.detach().cpu().item()
