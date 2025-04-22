import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

import pandas as pd
import gymnasium as gym

from gop_rl.utils import data_processor, prepare_data


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


# class ActionMLE(nn.Module):
#     def __init__(self, state_dim:int, action_dim:int,action_lim:float=2.0):
#         super(ActionMLE, self).__init__()
        
#         self.state_dim = state_dim
#         self.action_dim = action_dim

#         if type(action_lim) is not torch.Tensor:
#             action_lim = torch.tensor(action_lim, dtype=torch.float32)
#         self.action_lim = action_lim
        
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Linear(256, 128),
#             nn.GELU(),
#             nn.Linear(128, 32),
#             nn.GELU()
#         )
#         self.mu_head = nn.Linear(32, action_dim)
#         self.log_sigma_head = nn.Linear(32, action_dim)  

    
#     def forward(self, model_input:torch.Tensor):
#         x = self.fc(model_input)
#         mu = self.action_lim * torch.tanh(self.mu_head(x))
#         log_sigma = self.log_sigma_head(x)
#         return mu, log_sigma
    
#     def sample(self, state:torch.Tensor): 
#         mu, log_sigma = self.forward(state)
#         sigma = torch.exp(log_sigma) + 1e-9  # Ensure sigma is positive
#         dist = torch.distributions.Normal(mu, sigma)
#         action = dist.sample()
#         return action, mu, log_sigma

class ActionMLE(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim=2.0):
        super(ActionMLE, self).__init__()

        if type(action_lim) is not torch.Tensor:
            action_lim = torch.tensor(action_lim, dtype=torch.float32)
        self.action_lim = action_lim
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(32, action_dim)
        self.log_sigma_head = nn.Linear(32, action_dim)  

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma
    
    def sample(self, state):
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma) + 1e-9  # Ensure sigma is positive
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        return action, mu, log_sigma


def gaussian_nll_loss(mu:torch.Tensor, log_std:torch.Tensor, target:torch.Tensor):
    std = torch.exp(log_std)
    variance = std ** 2
    log_variance = 2 * log_std
    
    nll = 0.5 * (
        log_variance + 
        ((target - mu) ** 2) / variance + 
        torch.log(2 * torch.tensor(np.pi))
    )
    return nll.mean()


def train_model(args,csv_file_path:str):
    
    all_data = pd.read_csv(csv_file_path)
    episodes = all_data['episode'].unique()

    episodes_array = np.array(episodes)
    np.random.seed(args.seed if hasattr(args, 'seed') else 42)  # Set seed for reproducibility
    np.random.shuffle(episodes_array)
    
    split_train_idx = int(len(episodes) * 0.8)
    split_val_idx = split_train_idx + int(len(episodes) * 0.1)
    
    train_episodes = episodes_array[:split_train_idx]
    val_episodes = episodes_array[split_train_idx:split_val_idx]
    test_episodes = episodes_array[split_val_idx:]

    train_data = all_data[all_data['episode'].isin(train_episodes)].copy()
    val_data = all_data[all_data['episode'].isin(val_episodes)].copy()
    test_data = all_data[all_data['episode'].isin(test_episodes)].copy()

    processed_train_data, state_mean, state_std, actions_mean, actions_std = data_processor(train_data, args)
    augmented_val_data, _, _, _, _ = data_processor(val_data, args)
    
    
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high[0]

    if args.input_type == 'state':
        model_name = 'P(a|s)'
        input_dim = obs_dim
    elif args.input_type == 'state_action':
        model_name = 'P(a|s,a-1)'
        input_dim = obs_dim + action_dim
    elif args.input_type == 'prev_state_action':
        model_name = 'P(a|s,a-1,s-1)'
        input_dim = 2*obs_dim + action_dim
    elif args.input_type == 'state_prev_state':
        model_name = 'P(a|s,s-1)'
        input_dim = 2*obs_dim
    else:
        raise ValueError("Invalid input type. Must be 'state', 'state_action' or 'state_prev_state'")
    
    model = ActionMLE(input_dim, action_dim, action_lim=action_high)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=5,)
    expert_model_path = os.path.join(args.data_dir, f'{args.model_type}_{model_name}.pth')
    early_stopping = EarlyStopping(patience=args.early_stopping, min_delta=0.0, path=expert_model_path)

    X_train, Y_train = prepare_data(processed_train_data, args.input_type)
    X_val, Y_val = prepare_data(augmented_val_data, args.input_type)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=args.device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=args.device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=args.device)
    Y_val = torch.tensor(Y_val, dtype=torch.float32, device=args.device)

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False)

    history = {'train_nll': [], 'val_nll': [], 'train_mse': [], 'val_mse': []}

    for epoch in range(1, args.epochs+1):
        # --- train ---
        model.train()
        total_train_nll = 0.0
        total_train_mse = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            optimizer.zero_grad()
            mu, log_s = model(x_batch)
            loss = gaussian_nll_loss(mu, log_s, y_batch)
            mse_loss = nn.MSELoss()(mu, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_nll += loss.item() * x_batch.size(0)
            total_train_mse += mse_loss.item() * x_batch.size(0)
        avg_train_nll = total_train_nll / len(train_loader.dataset)
        avg_train_mse = total_train_mse / len(train_loader.dataset)

        # --- validate ---
        model.eval()
        total_val_nll = 0.0
        total_val_mse = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                mu, log_s = model(xb)
                loss = gaussian_nll_loss(mu, log_s, yb)
                mse_loss = nn.MSELoss()(mu, yb.view(-1, 1) if len(yb.shape) == 1 else yb)
                total_val_nll += loss.item() * xb.size(0)
                total_val_mse += mse_loss.item() * xb.size(0)
        avg_val_nll = total_val_nll / len(val_loader.dataset)
        avg_val_mse = total_val_mse / len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs} |"
              f" Train: {avg_train_nll:.4f} | Val: {avg_val_nll:.4f}")

        history['train_nll'].append(avg_train_nll)
        history['train_mse'].append(avg_train_mse)
        history['val_nll'].append(avg_val_nll)
        history['val_mse'].append(avg_val_mse)

        scheduler.step(avg_val_nll)
        early_stopping(avg_val_nll, model)
        if early_stopping.early_stop:
            print("→ Early stopping triggered")
            break

    # 11) Load best‐model & return
    model.load_state_dict(torch.load(expert_model_path))
    return model, history


def plot_metrics(history, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history['train_nll'], label='Train NLL', color='blue', linestyle='--')
    ax[0].plot(history['val_nll'], label='Validation NLL', color='orange', linestyle='-.')
    ax[0].set_ylim(bottom=-3.5,top=1)   
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Negative Log Likelihood')
    ax[0].set_title('Training and Validation NLL')
    ax[0].legend()
    ax[1].plot(history['train_mse'], label='Train MSE', color='blue', linestyle='--')
    ax[1].plot(history['val_mse'], label='Validation MSE', color='orange', linestyle='-.')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Mean Squared Error')
    ax[1].set_title('Training and Validation MSE for Mean Action Prediction')
    ax[1].legend()
    
    plt.savefig(save_path)
    plt.close()

def test_model(model, test_data, test_labels,args):
    model.eval()
    with torch.no_grad():
        sampled_actions, mu, log_s= model.sample(test_data)
        mu = mu.cpu().numpy()
        sampled_actions = sampled_actions.cpu().numpy().flatten()
        nll = gaussian_nll_loss(mu, log_s, test_labels).item()
        mse = nn.MSELoss()(mu, test_labels).item()

    # Plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_data[:, 0], test_data[:, 1], sampled_actions, color='blue', label='True')
    ax.scatter(test_data[:, 0], test_data[:, 1], test_labels, marker='x', color='red', label='Pred')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_zlabel('Action')
    ax.set_title(f'MLE Model of {args.input_type} -  Prediction vs True Actions - Test Set')
    plt.legend(loc='best')
    plt.savefig(f'{args.output_dir}/{args.input_type}_mle_fit.svg')
    plt.show()
    return nll, mse
