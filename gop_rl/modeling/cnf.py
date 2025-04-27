import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from gop_rl.utils import data_processor, prepare_data
from gop_rl.modeling import EarlyStopping

from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

class ConditionalBase(nn.Module):
    def __init__(self, condition_dim, latent_dim, action_lim=1.0):
        super().__init__()
        if not isinstance(action_lim, torch.Tensor):
            action_lim = torch.tensor(action_lim, dtype=torch.float32)
        self.action_lim = action_lim
        
        self.fc = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.log_sigma_head = nn.Linear(128, latent_dim)

    def forward(self, condition):
        h = self.fc(condition)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        return mu, log_sigma

class ConditionalAffineLayer(nn.Module):
    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.param_net = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )

    def forward(self, a, condition):
        batch, dim = a.shape
        params = self.param_net(condition).view(batch, dim, 2)
        s = params[..., 0]
        t = params[..., 1]
        scale = torch.exp(s)
        z = (a - t) / scale
        log_det = -torch.log(scale).sum(-1)
        return z, log_det

    def inverse(self, z, condition):
        batch, dim = z.shape
        params = self.param_net(condition).view(batch, dim, 2)
        s = params[..., 0]
        t = params[..., 1]
        scale = torch.exp(s)
        a = scale * z + t
        log_det = torch.log(scale).sum(-1)
        return a, log_det

class ConditionalSplineLayer(nn.Module):
    def __init__(self, condition_dim, latent_dim,
                 hidden_dim=256, num_bins=8, tail_bound=3.0):
        super().__init__()
        
        self.transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=latent_dim,
            hidden_features=hidden_dim,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            context_features=condition_dim,
        )

    def forward(self, a, condition):
        z, log_det = self.transform(a, context=condition)
       
        return z, log_det.sum(-1) if log_det.dim() > 1 else log_det

    def inverse(self, z, condition):
        a, log_det = self.transform.inverse(z, context=condition)
        return a, log_det.sum(-1) if log_det.dim() > 1 else log_det

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self,
                 condition_dim,
                 latent_dim,
                 n_flows,
                 action_lim=1.0,
                 affine_first=True,
                 spline_bins=8,
                 spline_bound=3.0,
                 hidden_dim=256):
        super().__init__()
        self.layers = nn.ModuleList()
        # Build alternating affine & spline blocks
        for _ in range(n_flows):
            self.layers.append(ConditionalAffineLayer(condition_dim, latent_dim))
            self.layers.append(ConditionalSplineLayer(condition_dim, latent_dim,
                                                    hidden_dim=hidden_dim,
                                                    num_bins=spline_bins,
                                                    tail_bound=spline_bound))
        self.conditional_base = ConditionalBase(condition_dim, latent_dim, action_lim)

    def forward(self, a, condition):
        log_det_total = 0.0
        z = a
        for layer in self.layers:
            z, ld = layer(z, condition)
            log_det_total += ld
        return z, log_det_total

    def inverse(self, z, condition):
        log_det_total = 0.0
        a = z
        for layer in reversed(self.layers):
            a, ld = layer.inverse(a, condition)
            log_det_total += ld
        return a, log_det_total

    def log_prob(self, a, condition):
        z, log_det = self.forward(a, condition)
        mu, log_sigma = self.conditional_base(condition)
        std = torch.exp(log_sigma)
        base = torch.distributions.Normal(mu, std)
        log_base = base.log_prob(z).sum(-1)
        return log_base + log_det

    def sample(self, condition):
        mu, log_sigma = self.conditional_base(condition)
        std = torch.exp(log_sigma)
        base = torch.distributions.Normal(mu, std)
        z = base.rsample()
        a, _ = self.inverse(z, condition)
        return a, mu



def train(args:dict,file_name:str):
    
    dummy_env = gym.make(args.env_name)
    args.obs_dim = dummy_env.observation_space.shape[0]
    args.act_dim = dummy_env.action_space.shape[0]
    action_high = dummy_env.action_space.high[0]
    dummy_env.close()

    if args.env_name == 'Pendulum-v1':
        args.obs_dim = 2

    all_data = pd.read_csv(file_name)
    episodes = all_data['episode'].unique()

    episodes_array = np.array(episodes)
    np.random.seed(args.seed if hasattr(args, 'seed') else 42)  # Set seed for reproducibility
    np.random.shuffle(episodes_array)
    
    split_train_idx = int(len(episodes) * 0.70)
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
    
    X_train, y_train = prepare_data(processed_train_data, args.input_type)
    X_val, y_val = prepare_data(processed_val_data, args.input_type)
    X_test, y_test = prepare_data(processed_test_data, args.input_type)
    
    if args.env_name == 'Pendulum-v1':
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.float32))
    
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                torch.tensor(y_val, dtype=torch.float32))


    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

    model = ConditionalNormalizingFlow(condition_dim=input_dim, n_flows=args.n_flows, latent_dim=args.act_dim, action_lim=action_high)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5,)
    expert_model_path = os.path.join(args.data_dir, f'{args.model_type}',f'{args.input_type}_{args.cycle}.pth')
    early_stopping = EarlyStopping(patience=args.early_stopping,min_delta=0.0, path=expert_model_path)

    train_nll, val_nll = [], []
    train_mse, val_mse = [], []
    print('train starts')
    for epoch in range(args.epochs):
        model.train()
        tot_nll, tot_mse = 0.0, 0.0 
        count = 0
        for input_batch, actions_batch in train_loader:
            count+=1
            # print('in batx: ',count)
            input_batch = input_batch.to(args.device)
            actions_batch = actions_batch.to(args.device)
            # print('nb uodates til nan')
            optimizer.zero_grad()
            # Compute negative log likelihood.
            log_prob = model.log_prob(actions_batch, input_batch)
            loss = -log_prob.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_nll += loss.item() * actions_batch.size(0)
            with torch.no_grad():
                base_mean, _ = model.conditional_base(input_batch)
                actions_pred,_ = model.inverse(base_mean, input_batch)   
                tot_mse += nn.MSELoss()(actions_pred, actions_batch).item()*actions_batch.size(0)
        train_nll.append(tot_nll/len(train_loader.dataset))
        train_mse.append(tot_mse/len(train_loader.dataset))
        

        model.eval()
        tot_nll, tot_mse = 0.0, 0.0
        with torch.no_grad():
            model.eval()
            for input_batch, actions_batch in val_loader:
                input_batch = input_batch.to(args.device)
                actions_batch = actions_batch.to(args.device)
                log_prob = model.log_prob(actions_batch, input_batch)
                loss = -log_prob.mean()
                tot_nll += loss.item() * actions_batch.size(0)
                base_mean, _ = model.conditional_base(input_batch)
                actions_pred,_ = model.inverse(base_mean, input_batch)
                tot_mse += nn.MSELoss()(actions_pred, actions_batch).item()*actions_batch.size(0)
        avg_val_loss = tot_nll/len(val_loader.dataset)
        val_nll.append(avg_val_loss)
        val_mse.append(tot_mse/len(val_loader.dataset))
        
        # if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_nll[-1]:.4f} || Val Loss: {avg_val_loss:.4f} || train_mse={train_mse[-1]:.4f} || val_mse={val_mse[-1]:.4f}")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered at epoch", epoch)
            break
        model.train()
    
    model.load_state_dict(torch.load(expert_model_path, weights_only=True))
    return model, (train_nll, train_mse), (val_nll, val_mse), X_test, y_test

def plot_metrics(train_losses, val_losses, args):
    tr_nll, tr_mse = train_losses
    vl_nll, vl_mse = val_losses
    epochs = list(range(1,len(tr_nll)+1))
    epochs = list(range(1,len(tr_nll)+1))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, tr_nll, '--', label='train_nll')
    plt.plot(epochs, vl_nll, '-.', label='val_nll')
    plt.legend(); plt.title('NLL')
    plt.subplot(1,2,2)
    plt.plot(epochs, tr_mse, '--', label='train_mse')
    plt.plot(epochs, vl_mse, '-.', label='val_mse')
    plt.legend(); plt.title('MSE')
    plt.savefig(f"{args.output_dir}/{args.model_type}/{args.input_type}_metrics_cycle_{args.cycle}.svg")
    plt.close()


def test_model(model, raw_states, raw_actions, args):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(raw_states,dtype=torch.float32,device=args.device)
        yb = torch.tensor(raw_actions,dtype=torch.float32,device=args.device)
        lp = model.log_prob(yb, xb)
        test_nll = -lp.mean().item()
        base_mean, _ = model.conditional_base(xb)
        actions_pred,_ = model.inverse(base_mean, xb)
        test_mse = nn.functional.mse_loss(actions_pred, yb).item()


    mean_action = actions_pred.cpu().numpy()

    # 1) Pendulum 3D scatter
    if args.env_name == 'Pendulum-v1':
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(raw_states[:1000,0],
                   raw_states[:1000,1],
                   raw_actions.flatten()[:1000],
                   color='blue', marker='o',
                   label='True Actions', alpha=0.5)
        ax.scatter(raw_states[:1000,0],
                   raw_states[:1000,1],
                   mean_action.flatten()[:1000],
                   color='red',  marker='x',
                   label='Predicted Mean', alpha=0.6)
        ax.set_xlabel('State 0')
        ax.set_ylabel('State 1')
        ax.set_zlabel('Action')
        ax.set_title(f'{args.input_type} input – Pred vs True (test)')
        ax.legend(loc='best')

    # 2) General multi-dim plot
    else:
        action_dim = raw_actions.shape[1]
        n_row = int(np.ceil(action_dim/2))
        fig, axes = plt.subplots(n_row, 2, figsize=(10,5*n_row))
        axes = axes.flatten()
        for i in range(action_dim):
            ax = axes[i]
            ax.scatter(raw_actions[:1000,i],
                       mean_action[:1000,i],
                       marker='x', color='red',
                       alpha=0.6, label='Pred Mean')
            ax.plot(*ax.get_xlim(), *ax.get_xlim(), color='k', label='ideal')
            ax.set_xlabel(f'True a[{i}]')
            ax.set_ylabel(f'Pred a[{i}]')
            ax.grid(True)
            ax.legend(loc='best')

        for j in range(action_dim, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.model_type}/{args.input_type}_fit_cycle_{args.cycle}.svg")
    plt.close()

    return test_nll, test_mse