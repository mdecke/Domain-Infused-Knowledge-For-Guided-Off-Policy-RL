import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

  
class DDPGMemory:
    def __init__(self, state_dim:int, action_dim:int, buffer_length:int, device:str='cpu'):
        self.memory_buffer_length = buffer_length
        self.ptr = 0
        self.size = 0
        self.device = device

        self.states = np.zeros((buffer_length, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_length, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_length, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_length, 1), dtype=np.float32)

    def add_sample(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.ptr = (self.ptr + 1) % self.memory_buffer_length
        self.size = min(self.size + 1, self.memory_buffer_length)
    
    def sample_memory(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32,device=self.device),
            torch.tensor(self.actions[indices], dtype=torch.float32,device=self.device),
            torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        )    

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, policy_lr=1e-3, action_lim=2.0,device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    
        self.fc1 = nn.Sequential(nn.Linear(state_dim, 400), nn.LayerNorm(400))
        self.fc2 = nn.Sequential(nn.Linear(400, 300), nn.LayerNorm(300))
        self.action_layer = nn.Linear(300, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), policy_lr)
        if type(action_lim) is not torch.Tensor:
            action_lim = torch.tensor(action_lim, dtype=torch.float32)
        self.action_lim = action_lim
        self.to(device)
        
    def forward(self, inputs):
        x = functional.relu(self.fc1(inputs))
        x = functional.relu(self.fc2(x)) 
        return self.action_lim * torch.tanh(self.action_layer(x))

class Value(nn.Module):
    def __init__(self, state_dim, action_dim, value_lr=1e-3, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Sequential(nn.Linear(state_dim+action_dim, 400), nn.LayerNorm(400))
        self.fc2 = nn.Sequential(nn.Linear(400, 300), nn.LayerNorm(300))
        self.fc3 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), value_lr)
        self.to(device)
    
    def forward(self, inputs):
        x = functional.relu(self.fc1(inputs))
        x = functional.relu(self.fc2(x))
        return self.fc3(x).squeeze()

def init_model_weights(model:nn.Module, low=0.0, high=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "weight" in name:
                nn.init.uniform_(param, a=low, std=high)
            elif "bias" in name:
                nn.init.constant_(param, val=low)

class DDPG:
    def __init__(self, policy_network:Policy,value_network:Value, 
                 discount_factor:float, seed=None, device='cpu'):
        
        self.pi = policy_network.to(device=device)
        self.pi_t = Policy(state_dim=self.pi.state_dim, action_dim=self.pi.action_dim, device=device)
        self.pi_t.load_state_dict(self.pi.state_dict())

        self.q = value_network.to(device=device)
        self.q_t = Value(state_dim=self.q.state_dim, action_dim=self.q.action_dim, device=device)
        self.q_t.load_state_dict(self.q.state_dict())

        self.gamma = discount_factor
        self.pi_loss = []
        self.q_loss = []
        self.device = device
        self.seed = seed
    
    def soft_update(self,target_network, network, tau):
        for target_param, source_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


    def train(self,memory_buffer:DDPGMemory, batch_size:int, epochs:int):

        models = [self.pi, self.pi_t, self.q, self.q_t]
        for model in models:
            model.train()
        
        for epoch in range(epochs):
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = memory_buffer.sample_memory(batch_size)

            sampled_states = sampled_states.to(self.device)
            sampled_actions = sampled_actions.to(self.device)
            sampled_rewards = sampled_rewards.view(-1).to(self.device)
            sampled_next_states = sampled_next_states.to(self.device)
            sampled_dones = sampled_dones.view(-1).to(self.device)
            
            # compute target values
            with torch.no_grad():
                next_actions = self.pi_t.forward(sampled_next_states)
                next_state_action_pairs = torch.cat([sampled_next_states, next_actions], dim=1)
                target_q_values = self.q_t.forward(next_state_action_pairs)
                y = sampled_rewards + self.gamma * (1 - sampled_dones) * target_q_values

            # compute critic loss
            state_action_pairs = torch.cat([sampled_states, sampled_actions], dim=1)
            critic_values = self.q.forward(state_action_pairs)
            critic_loss = functional.mse_loss(critic_values, y)

            # optimization step (critic)
            self.q.optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.q.optimizer.step()

            # compute policy (actor) loss
            actions = self.pi.forward(sampled_states)
            state_action_pairs = torch.cat([sampled_states, actions], dim=1)
            critic_values = self.q.forward(state_action_pairs)

            policy_loss = -critic_values.mean()

            # optimization step (policy)
            self.pi.optimizer.zero_grad()
            policy_loss.backward()
            # nn.utils.clip_grad_norm_(self.pi.parameters(), 5.0)
            self.pi.optimizer.step()

            # update target networks
            self.soft_update(self.pi_t, self.pi, tau=0.005)
            self.soft_update(self.q_t, self.q, tau=0.005)

        
        self.pi_loss.append(policy_loss.detach().cpu().numpy().item())
        self.q_loss.append(critic_loss.detach().cpu().numpy().item())

