import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from decision_transformer.models.value_net import ValueNetwork

class ValueFunction:
    def __init__(self, state_dim, device=None, learning_rate=1e-3):
        self.value_network = ValueNetwork(state_dim)
        if device is not None:
            self.value_network.to(device)
            self.device = device
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def _compute_returns(self, rewards, dones, gamma):
        returns = []
        g = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                g = 0
            g = reward + gamma * g
            returns.insert(0, g)
        return returns
    
    def _collect_data(self, trajectories, state_mean, state_std, gamma):
        all_states = []
        all_returns = []
        state_mean = torch.tensor(state_mean, dtype=torch.float32)
        state_std = torch.tensor(state_std, dtype=torch.float32)

        states = trajectories['observations']
        rewards = trajectories['rewards']
        dones = trajectories['dones']

        for i in range(len(states)):
            # Compute returns for the current trajectory
            returns = self._compute_returns(rewards[i], dones[i], gamma)

            all_states.extend(states[i])
            all_returns.extend(returns.copy())

        # Convert lists to torch tensors
        all_states = (torch.tensor(all_states, dtype=torch.float32) - state_mean) / state_std
        all_returns = torch.tensor(all_returns, dtype=torch.float32)
        # print(all_states.shape, state_mean.shape, state_std.shape, all_returns.shape)

        return all_states, all_returns
    
    def train(self, trajectories, state_mean, state_std, total_iterations=10, batch_size=640, gamma=1.0):
        self.value_network.train()

        # Collect all states and returns
        all_states, all_returns = self._collect_data(trajectories, state_mean, state_std, gamma)

        # Create batches
        dataset_size = len(all_states)
        
        print("Trainig the value function ......")
        v_loss = 0.0
        train_times = 0
        for _ in tqdm(range(total_iterations)):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get batch of states and returns
                states_batch = all_states[batch_indices].to(self.device)
                returns_batch = all_returns[batch_indices].to(self.device)

                # Forward pass to get predicted values
                predicted_values = self.value_network(states_batch).squeeze()

                # Compute the loss
                loss = self.criterion(predicted_values, returns_batch)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                v_loss += loss.item()
                train_times += 1
        
        return v_loss / float(train_times)

    def predict(self, states):
        return self.value_network(states).squeeze().detach().cpu().numpy()    
    
    def update_model(self, new_model):
        self.value_network.load_state_dict(new_model.state_dict())

            