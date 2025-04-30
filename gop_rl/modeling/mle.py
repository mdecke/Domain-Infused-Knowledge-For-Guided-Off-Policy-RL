import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import pandas as pd
import gymnasium as gym
import math

from gop_rl.utils import data_processor, prepare_data
from gop_rl.modeling import EarlyStopping


_LOG_2PI_HALF = 0.5*math.log(2 * math.pi)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
class ActionMLE(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super(ActionMLE, self).__init__()

        if type(action_lim) is not torch.Tensor:
            action_lim = torch.tensor(action_lim, dtype=torch.float32)
        self.action_lim = action_lim
        
        self.fc = nn.Sequential(nn.Linear(state_dim, 256),
                                # nn.LayerNorm(256),
                                nn.ReLU(),
                                nn.Dropout(0.15),

                                nn.Linear(256, 256),
                                # nn.LayerNorm(256),
                                nn.ReLU(),
                                nn.Dropout(0.15),

                                nn.Linear(256, 64),
                                #nn.LayerNorm(64),
                                nn.ReLU())
        self.mu_head = nn.Linear(64, action_dim)
        self.log_sigma_head = nn.Linear(64, action_dim)

    def forward(self, state):
        x = self.fc(state)
        mu = torch.clamp(self.mu_head(x), min=-self.action_lim, max=self.action_lim)#self.action_lim*torch.tanh(self.mu_head(x))
        log_sigma = torch.clamp(self.log_sigma_head(x), min=-5.0, max=2.0) #clamps sigma to be in [0.01, 7.0]
        return mu, log_sigma
    
    def sample(self, state):
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma) + 1e-6  # Ensure sigma is positive
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        return action, mu , sigma


def gaussian_nll_loss(mu:torch.Tensor, log_std:torch.Tensor, target:torch.Tensor):
    var = torch.exp(2*log_std)
    return nn.GaussianNLLLoss(full=True, reduction='mean')(mu, target, var)


def process_batch(model, episodes_subset, data, args):
        batch_nll = 0.0
        batch_mse = 0.0
        for _ in range(args.nb_traj):
            ep = np.random.choice(episodes_subset)
            df = data[data['episode'] == ep]

            states = df['states'].to_numpy()
            states = [np.array(s, dtype=np.float32) for s in states]
            states = np.stack(states)

            previous_actions = df['prev_actions'].to_numpy()
            if args.input_type != 'state':
                previous_actions = [np.array(a, dtype=np.float32) for a in previous_actions]
                previous_actions = np.stack(previous_actions)
                
            prev_states = df['prev_states'].to_numpy()
            prev_states = [np.array(s, dtype=np.float32) for s in prev_states]
            prev_states = np.stack(prev_states)
            

            batch_states = torch.tensor(states, dtype=torch.float32, device=args.device)
            targets = df['targets'].to_numpy()
            
            if args.env_name == 'Pendulum-v1':
                labels = torch.tensor(targets, dtype=torch.float32, device=args.device).reshape(-1, 1)
                previous_actions = previous_actions.reshape(-1, 1)
            else:
                targets = [np.array(t, dtype=np.float32) for t in targets]
                targets = np.stack(targets)
                labels = torch.tensor(targets, dtype=torch.float32, device=args.device)

            if args.input_type == 'state':
                model_input = batch_states
            elif args.input_type == 'state_action':
                batch_previous_actions = torch.tensor(previous_actions, dtype=torch.float32)
                model_input = torch.cat((batch_states, batch_previous_actions), dim=1)
            elif args.input_type == 'prev_state_action':
                batch_previous_states = torch.tensor(prev_states, dtype=torch.float32)
                batch_previous_actions = torch.tensor(previous_actions, dtype=torch.float32)
                model_input = torch.cat((batch_states, batch_previous_actions, batch_previous_states), dim=1)
            elif args.input_type == 'state_prev_state':
                batch_previous_states = torch.tensor(prev_states, dtype=torch.float32)
                model_input = torch.cat((batch_states, batch_previous_states), dim=1)
            else:
                raise ValueError("Invalid data type. Must be 'states' or 'states_action' or 'prev_states_action'")
            mu, log_std = model(model_input)
            nll = gaussian_nll_loss(mu, log_std, labels)
            mse = nn.functional.mse_loss(mu,labels)

            batch_nll += nll
            batch_mse += mse

        return batch_nll / args.nb_traj, batch_mse / args.nb_traj



def train(args,csv_file_path:str):
    
    dummy_env = gym.make(args.env_name)
    args.obs_dim = dummy_env.observation_space.shape[0]
    args.act_dim = dummy_env.action_space.shape[0]
    action_high = dummy_env.action_space.high[0]
    dummy_env.close()
    
    if args.env_name == 'Pendulum-v1':
        args.obs_dim = 2

    all_data = pd.read_csv(csv_file_path)
    episodes = all_data['episode'].unique()

    episodes_array = np.array(episodes)
    np.random.seed(args.seed if hasattr(args, 'seed') else 42)  # Set seed for reproducibility
    np.random.shuffle(episodes_array)
    
    split_train_idx = int(len(episodes) * 0.7)
    split_val_idx = split_train_idx + int(len(episodes) * 0.20)
    
    train_episodes = episodes_array[:split_train_idx]
    val_episodes = episodes_array[split_train_idx:split_val_idx]
    test_episodes = episodes_array[split_val_idx:]

    train_data = all_data[all_data['episode'].isin(train_episodes)].copy()
    val_data = all_data[all_data['episode'].isin(val_episodes)].copy()
    test_data = all_data[all_data['episode'].isin(test_episodes)].copy()

    processed_train_data, state_mean, state_std, actions_mean, actions_std = data_processor(train_data, args)
    
    args.state_mean = state_mean
    args.state_std = state_std
    args.actions_mean = actions_mean
    args.actions_std = actions_std
    
    processed_val_data, *_ = data_processor(val_data, args)
    processed_test_data, *_ = data_processor(test_data, args, test=True)
    

    if args.input_type == 'state':                  input_dim = args.obs_dim
    elif args.input_type == 'state_action':         input_dim = args.obs_dim + args.act_dim
    elif args.input_type == 'prev_state_action':    input_dim = 2*args.obs_dim + args.act_dim
    elif args.input_type == 'state_prev_state':     input_dim = 2*args.obs_dim
    else: raise ValueError("Invalid input type. Must be 'state', 'state_action' or 'state_prev_state'")
    
    model = ActionMLE(input_dim, args.act_dim, action_lim=action_high)
    model.apply(initialize_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5,)
    expert_model_path = os.path.join(args.data_dir, f'{args.model_type}',f'{args.input_type}_{args.cycle}.pth')
    early_stopping = EarlyStopping(patience=args.early_stopping,min_delta=0.0, path=expert_model_path)

    
    

    history = {'train_nll':[], 'train_mse':[],'val_nll':[], 'val_mse':[]}
    best_vals = {           # values at the epoch that beat every previous Val-NLL
        "train_nll": None,
        "train_mse": None,
        "val_nll"  : float("inf"),   # start higher than anything you’ll see
        "val_mse"  : None,
    }

    print('train starts')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_nlls, train_mses = [], []

        for step in range(1, args.n_grad_steps + 1):
            optimizer.zero_grad()
            nll_loss, mse_loss = process_batch(model,train_episodes, processed_train_data, args)
            nll_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_nlls.append(nll_loss.item())
            train_mses.append(mse_loss.item())

            # if step % 10 == 0:
                # print(
                #     f"[Epoch {epoch}] Step {step}/{args.n_grad_steps}  "
                #     f"train_nll={nll_loss.item():.4f}  train_mse={mse_loss.item():.4f}"
                # )

        mean_train_nll = float(np.mean(train_nlls))
        mean_train_mse = float(np.mean(train_mses))

        # ─── validation ───────────────────────────────────────────────────────────
        model.eval()
        val_nlls, val_mses = [], []
        with torch.no_grad():
            for _ in range(max(1, args.n_grad_steps // 4)):
                vnll, vmse = process_batch(model, val_episodes, processed_val_data, args)
                val_nlls.append(vnll.item())
                val_mses.append(vmse.item())

        mean_val_nll = float(np.mean(val_nlls))
        mean_val_mse = float(np.mean(val_mses))

        print(f"→ Epoch {epoch}/{args.epochs}  "
                f"Train NLL: {mean_train_nll:.4f}  Train MSE: {mean_train_mse:.4f}  "
                f"Val NLL: {mean_val_nll:.4f}    Val MSE: {mean_val_mse:.4f}")

        history['train_nll'].append(mean_train_nll)
        history['train_mse'].append(mean_train_mse)
        history['val_nll'].append(mean_val_nll)
        history['val_mse'].append(mean_val_mse)

        scheduler.step(mean_val_nll)
        if mean_val_nll < best_vals["val_nll"]:
            best_vals["train_nll"] = mean_train_nll
            best_vals["train_mse"] = mean_train_mse
            best_vals["val_nll"]   = mean_val_nll
            best_vals["val_mse"]   = mean_val_mse
        early_stopping(mean_val_nll, model)
        if early_stopping.early_stop:
            print("Early stopping triggered at epoch", epoch)
            break
    # ─── 5) Load the best weights & return ───────────────────────────────────────
    model.load_state_dict(torch.load(expert_model_path, weights_only=True))
    
    print(
          f"  Train NLL : {best_vals['train_nll']:.4f}\n"
          f"  Train MSE : {best_vals['train_mse']:.4f}\n"
          f"  Val   NLL : {best_vals['val_nll']:.4f}\n"
          f"  Val   MSE : {best_vals['val_mse']:.4f}")

    return model, history, processed_test_data, test_episodes


def plot_metrics(history,args):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history['train_nll'], label='Train NLL', color='blue', linestyle='--')
    ax[0].plot(history['val_nll'], label='Validation NLL', color='orange', linestyle='-.')
    nll_min = min(min(history['train_nll']), min(history['val_nll']))
    nll_max = max(max(history['train_nll']), max(history['val_nll']))
    y_padding = (nll_max - nll_min) * 0.05  # 5% padding
    ax[0].set_ylim(bottom=nll_min-y_padding, top=nll_max+y_padding)
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
    
    plt.savefig(f'{args.output_dir}/{args.model_type}/{args.input_type}_metrics_cycle_{args.cycle}.svg')
    plt.close()

def test_model(model, test_episodes, test_data, args):
    X_test, y_test = prepare_data(test_data, args.input_type)
    if args.input_type == 'state':
        states_std = (X_test - args.state_mean) / args.state_std
    elif args.input_type == 'state_action':
        current_states_std = (X_test[:,:args.obs_dim] - args.state_mean) / args.state_std
        prev_action_std = (X_test[:,args.obs_dim:] - args.actions_mean) / args.actions_std
        states_std = np.concatenate([current_states_std, prev_action_std], axis=1)
    elif args.input_type == 'prev_state_action':
        current_states_std = (X_test[:,:args.obs_dim] - args.state_mean) / args.state_std
        prev_action_std = (X_test[:,args.obs_dim:args.obs_dim+args.act_dim] - args.actions_mean) / args.actions_std
        prev_state_std = (X_test[:,args.obs_dim+args.act_dim:] - args.state_mean) / args.state_std
        states_std = np.concatenate([current_states_std, prev_action_std, prev_state_std], axis=1)
    elif args.input_type == 'state_prev_state':
        current_states_std = (X_test[:,:args.obs_dim] - args.state_mean) / args.state_std
        prev_state_std = (X_test[:,args.obs_dim:] - args.state_mean) / args.state_std
        states_std = np.concatenate([current_states_std, prev_state_std], axis=1)
    else:
        raise ValueError("Invalid input type. Must be 'state', 'state_action' or 'state_prev_state'")
    
    # model_input = torch.from_numpy(states_std).float().to(args.device)
    model.eval()
    nlls, mses = [], []
    with torch.no_grad():
        for _ in range(max(1, args.n_grad_steps // 4)):
            vnll, vmse = process_batch(model, test_episodes, test_data, args)
            nlls.append(vnll.item())
            mses.append(vmse.item())

        # mean_action, log_s = model(model_input)
        # raw_actions_tensor = torch.from_numpy(raw_actions).float().to(args.device)

        # nll = gaussian_nll_loss(mean_action,log_s,raw_actions_tensor).item()
        # mse = nn.MSELoss()(mean_action, raw_actions_tensor.view(-1, 1) if args.act_dim == 1 else raw_actions_tensor).item()
    nll = float(np.mean(nlls))
    mse = float(np.mean(mses))
    with torch.no_grad():
        model_input = torch.tensor(states_std, dtype=torch.float32, device=args.device)
        mean_action,_ = model(model_input)
        raw_actions_tensor = torch.from_numpy(y_test).float().to(args.device)

    mean_action = mean_action.cpu().numpy()

    if args.env_name == 'Pendulum-v1':
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_test[:10000,0],X_test[:10000,1], y_test.flatten()[:10000],
                color='blue', marker='o', label='True Actions', alpha=0.4)
        ax.scatter(X_test[:10000,0], X_test[:10000,1], mean_action.flatten()[:10000],
                color='red',  marker='x', label='Predicted Mean Actions', alpha=0.6)
        ax.set_xlabel('State dim 0 (raw)')
        ax.set_ylabel('State dim 1 (raw)')
        ax.set_zlabel('Action')
        ax.set_title(f'{args.input_type} MLE: Pred vs True on test set')
        ax.legend(loc='best')
    else:
        action_dim = y_test.shape[1]
        n_row = int(np.ceil(action_dim/2))
        fig, ax = plt.subplots(nrows=n_row, ncols=2, figsize=(10,10))
        ax = ax.flatten()

        for i in range(action_dim):
            ax[i].scatter(y_test[:5000,i], mean_action[:5000,i], marker='x', color='red', label='Predicted Mean Actions', alpha=0.6)
            ax[i].set_xlabel(f'true action dim{i}')
            ax[i].set_ylabel(f'predicted action dim {i}')
            ax[i].grid(True)
            x_min, x_max = ax[i].get_xlim()
            x=np.linspace(x_min,x_max,100)
            ax[i].plot(x,x,color='k',label='x=y')
            ax[i].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{args.model_type}/{args.input_type}_fit_cycle_{args.cycle}.svg')
    # plt.show()
    plt.close()

    return nll, mse