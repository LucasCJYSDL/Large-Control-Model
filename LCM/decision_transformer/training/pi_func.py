import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from decision_transformer.models.actor_net import ActorNetwork

class PolicyFunction:
    def __init__(self, state_dim, action_dim, device=None, learning_rate=1e-3):
        self.policy_network = ActorNetwork(state_dim, action_dim)
        if device is not None:
            self.policy_network.to(device)
            self.device = device
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def _collect_data(self, trajectories, state_mean, state_std):
        all_states = []
        all_actions = []
        state_mean = torch.tensor(state_mean, dtype=torch.float32)
        state_std = torch.tensor(state_std, dtype=torch.float32)

        states = trajectories['observations']
        actions = trajectories['actions']

        for i in range(len(states)):
            all_states.extend(states[i])
            all_actions.extend(actions[i])

        # Convert lists to torch tensors
        all_states = (torch.tensor(all_states, dtype=torch.float32) - state_mean) / state_std
        all_actions = torch.tensor(all_actions, dtype=torch.float32)
        # print(all_states.shape, state_mean.shape, state_std.shape, all_actions.shape)

        return all_states, all_actions
    
    def train(self, trajectories, state_mean, state_std, total_iterations=10, batch_size=640):
        self.policy_network.train()

        # Collect all states and returns
        all_states, all_actions = self._collect_data(trajectories, state_mean, state_std)

        # Create batches
        dataset_size = len(all_states)
        
        print("Trainig the policy function ......")
        pi_loss = 0.0
        train_times = 0
        for _ in tqdm(range(total_iterations)):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get batch of states and returns
                states_batch = all_states[batch_indices].to(self.device)
                actions_batch = all_actions[batch_indices].to(self.device)

                # Forward pass to get predicted values
                predicted_actions = self.policy_network(states_batch)

                # Compute the loss
                loss = self.criterion(predicted_actions, actions_batch)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pi_loss += loss.item()
                train_times += 1
        
        return pi_loss / float(train_times)

    def predict(self, states):
        return self.policy_network(states).detach().cpu().numpy()
    
    def update_model(self, new_model):
        self.policy_network.load_state_dict(new_model.state_dict())