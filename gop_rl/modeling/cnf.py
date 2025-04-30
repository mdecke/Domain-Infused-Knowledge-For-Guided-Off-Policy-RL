import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from gop_rl.utils import data_processor, prepare_data
from gop_rl.modeling import EarlyStopping


from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
# from nflows.transforms.nonlinearities import Tanh


    
# class ConditionalBase(nn.Module):
#     def __init__(self, condition_dim, latent_dim, action_lim=1.0):
#         super().__init__()
#         if not isinstance(action_lim, torch.Tensor):
#             action_lim = torch.tensor(action_lim, dtype=torch.float32)
#         self.action_lim = action_lim
        
#         self.fc = nn.Sequential(nn.Linear(condition_dim, 32),
#                                 # nn.LayerNorm(256),
#                                 nn.ReLU(),
#                                 # nn.Dropout(0.15),

#                                 nn.Linear(32, 64),
#                                 # nn.LayerNorm(256),
#                                 nn.ReLU(),
#                                 # nn.Dropout(0.15),

#                                 nn.Linear(64, 16),
#                                 #nn.LayerNorm(64),
#                                 nn.ReLU())
#         self.mu_head = nn.Linear(16, latent_dim)
#         self.log_sigma_head = nn.Linear(16, latent_dim)

#     def forward(self, condition):
#         x = self.fc(condition)
#         mu = self.mu_head(x),#self.action_lim*torch.tanh(self.mu_head(x))
#         log_sigma = self.log_sigma_head(x)#torch.clamp(self.log_sigma_head(x), min=-6.0, max=2.0) #clamps sigma to be in [0.01, 7.0]
#         return mu, log_sigma

# class ConditionalAffineLayer(nn.Module):
#     def __init__(self, condition_dim, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim
        
#         self.param_net = nn.Sequential(
#             nn.Linear(condition_dim, 32),
#             # nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             # nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Linear(16, latent_dim * 2),
#         )

#     def forward(self, a, condition):
#         batch, dim = a.shape
#         params = self.param_net(condition).view(batch, dim, 2)
#         s = params[..., 0]
#         t = params[..., 1]
#         scale = torch.exp(s)
#         z = (a - t) / scale
#         log_det = -torch.log(scale).sum(-1)
#         return z, log_det

#     def inverse(self, z, condition):
#         batch, dim = z.shape
#         params = self.param_net(condition).view(batch, dim, 2)
#         s = params[..., 0]
#         t = params[..., 1]
#         scale = torch.exp(s)
#         a = scale * z + t
#         log_det = torch.log(scale).sum(-1)
#         return a, log_det

# class ConditionalSplineLayer(nn.Module):
#     def __init__(self, condition_dim, latent_dim,
#                  hidden_dim=64, num_bins=8, tail_bound=3.0):
#         super().__init__()
        
#         self.transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
#             features=latent_dim,
#             hidden_features=hidden_dim,
#             num_bins=num_bins,
#             tails="linear",
#             tail_bound=tail_bound,
#             context_features=condition_dim,
#         )

#     def forward(self, a, condition):
#         z, log_det = self.transform(a, context=condition)
       
#         return z, log_det.sum(-1) if log_det.dim() > 1 else log_det

#     def inverse(self, z, condition):
#         a, log_det = self.transform.inverse(z, context=condition)
#         return a, log_det.sum(-1) if log_det.dim() > 1 else log_det

# class ConditionalNormalizingFlow(nn.Module):
#     """
#     * **generate**:   forward(latent, cond)   ℝ → action space
#     * **normalise**:  inverse(action, cond)   action space → ℝ
#     """

#     def __init__(self,
#                  condition_dim,
#                  latent_dim,
#                  n_flows,
#                  action_lim   = 1.0,
#                  spline_bins  = 8,
#                  spline_bound = 3.0,
#                  hidden_dim   = 64):
#         super().__init__()

#         self.action_lim = float(action_lim)

#         # ---------- build flow ---------------------------------------
#         flow_layers: list[nn.Module] = []
#         for _ in range(n_flows):
#             flow_layers.append(
#                 ConditionalAffineLayer(condition_dim, latent_dim)
#             )
#             # flow_layers.append(
#             #     ConditionalSplineLayer(condition_dim, latent_dim,
#             #                            hidden_dim  = hidden_dim,
#             #                            num_bins    = spline_bins,
#             #                            tail_bound  = spline_bound)
#             # )
        
#         self.flow = nn.ModuleList(flow_layers)

#         # base network (outputs mean, log_std)
#         self.base_net = ConditionalBase(condition_dim,
#                                         latent_dim,
#                                         action_lim = self.action_lim)

#     # -------------- GENERATIVE --------------------------------------
#     def forward(self, action, condition):
#         """a --> z"""
#         log_det_total = 0.0
#         z = action
#         for layer in self.flow:                 # latent → … → action
#             z, log_det = layer(z, condition)
#             log_det_total  += log_det
        
#         return z, log_det_total

#     # -------------- NORMALISING -------------------------------------
#     def inverse(self, z, condition):
#         """latent --> action"""
#         log_det_total = 0.0
#         a = z
#         for layer in reversed(self.flow):       # action → … → latent
#             a, log_det = layer.inverse(a, condition)
#             log_det_total  += log_det
#         return a, log_det_total

#     # -------------- API helpers -------------------------------------
#     def log_prob(self, action, condition):
#         z, log_det_flow = self.forward(action, condition)
#         mean, log_std = self.base_net(condition)
#         base_dist = torch.distributions.Normal(mean, torch.exp(log_std))
#         log_prob_base = base_dist.log_prob(z)
#         if len(log_prob_base.shape) >1:
#             log_prob_base = log_prob_base.sum(1)
#         if len(log_det_flow) != len(log_prob_base):
#             log_det_flow = log_det_flow.view(log_prob_base)
#         return log_prob_base + log_det_flow

#     def sample(self, condition):
#         mean, log_std = self.base_net(condition)
#         latent = torch.distributions.Normal(mean, torch.exp(log_std)).rsample()
#         action, _ = self.forward(latent, condition)
#         return action


# def variance_regulariser(log_std, min_val=-4.0, max_val=1.0):
#     return (log_std.clamp(min=min_val, max=max_val) ** 2).mean()

class ConditionalBase(nn.Module):
    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2 * latent_dim)  # outputs concatenated mean and log_std
        )
    
    def forward(self, condition):
        params = self.net(condition)
        latent_dim = params.shape[1] // 2
        mean = params[:, :latent_dim]
        log_std = params[:, latent_dim:]
        return mean, log_std
    

class ConditionalAffineLayer(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  
        )
    
    def forward(self, a, condition):
        params = self.net(condition)  
        s = params[:, 0:1]            # log-scale parameter
        t = params[:, 1:2]            # translation
        scale = torch.exp(s)          # ensure scale is positive
        # Forward transform: a -> latent z.
        z = (a - t) / scale
        log_det = -torch.log(scale).squeeze(1)
        return z, log_det
    
    def inverse(self, z, condition):
        params = self.net(condition)
        s = params[:, 0:1]
        t = params[:, 1:2]
        scale = torch.exp(s)
        # Inverse transform: latent z -> a.
        a = scale * z + t
        log_det = torch.log(scale).squeeze(1)
        return a, log_det
    
class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, condition_dim, n_flows, latent_dim=1):
        super().__init__()
        self.n_flows = n_flows
        self.layers = nn.ModuleList([ConditionalAffineLayer(condition_dim) for _ in range(n_flows)])
        self.conditional_base = ConditionalBase(condition_dim, latent_dim)
    
    def forward(self, a, condition):
        # Map action a to latent variable z.
        log_det_total = 0.0
        z = a
        for layer in self.layers:
            z, log_det = layer(z, condition)
            log_det_total += log_det
        return z, log_det_total
    
    def inverse(self, z, condition):
        
        log_det_total = 0.0
        a = z
        for layer in reversed(self.layers):
            a, log_det = layer.inverse(a, condition)
            log_det_total += log_det
        return a, log_det_total
    
    def log_prob(self, a, condition):
        z, log_det = self.forward(a, condition)
        
        base_mean, base_log_std = self.conditional_base(condition)
        base_std = torch.exp(base_log_std)
        base_dist = torch.distributions.Normal(base_mean, base_std)
        
        log_base = base_dist.log_prob(z)
        if len(log_base.shape) > 1:
            log_base = log_base.sum(1)  # Sum across all dimensions except batch dim
        
        # Make sure log_det has the same shape
        if len(log_det.shape) != len(log_base.shape):
            # Reshape log_det to match log_base
            log_det = log_det.view(log_base.shape)
            
        return log_base + log_det
    
    def sample(self, num_samples, condition):
        # Sample latent variable from the conditional base.
        base_mean, base_log_std = self.conditional_base(condition)
        base_std = torch.exp(base_log_std)
        base_dist = torch.distributions.Normal(base_mean, base_std)
        z = base_dist.rsample()  # reparameterized sample; shape: (num_samples, latent_dim)
        a, _ = self.inverse(z, condition)
        return a


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

    model = ConditionalNormalizingFlow(condition_dim=input_dim, n_flows=args.n_flows, latent_dim=args.act_dim)#, action_lim=action_high)
    # model.base_net.load_state_dict(torch.load('data/ant/mle/state_1.pth',weights_only=True))
    # print('mle base loaded')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5,)
    expert_model_path = os.path.join(args.data_dir, f'{args.model_type}',f'{args.input_type}_{args.cycle}.pth')
    early_stopping = EarlyStopping(patience=args.early_stopping,min_delta=0.0, path=expert_model_path)

    train_nll, val_nll = [], []
    train_mse, val_mse = [], []

    best_vals = {           # values at the epoch that beat every previous Val-NLL
        "train_nll": None,
        "train_mse": None,
        "val_nll"  : None,   # start higher than anything you’ll see
        "val_mse"  : None,
        "val_loss" : float("inf"),
    }


    print('train starts')
    # for epoch in range(args.epochs):
    #     model.train()
    #     tot_nll, tot_mse = 0.0, 0.0 
    #     for input_batch, actions_batch in train_loader:
    #         # print('in batx: ',count)
    #         input_batch = input_batch.to(args.device)
    #         actions_batch = actions_batch.to(args.device)
    #         optimizer.zero_grad()

    #         nll_loss = -model.log_prob(actions_batch, input_batch).mean()
    #         tot_nll += nll_loss.item() * actions_batch.size(0)

    #         base_mean, log_std = model.base_net(input_batch)        # (grad-tracked)
    #         actions_pred, _ = model.inverse(base_mean, input_batch) # mean action

    #         mse_loss  = nn.functional.mse_loss(actions_pred, actions_batch)
    #         tot_mse  += mse_loss.item() * actions_batch.size(0)

    #         # sigma_reg = variance_regulariser(log_std)

    #         # excess     = torch.clamp(actions_pred.abs() - (model.action_lim - 0.2), min=0.0)
    #         # bound_pen  = (excess ** 2).mean()

    #         loss = (
    #             nll_loss
    #         #    + 0.10 * mse_loss     # imitation
    #         #   + 0.01 * sigma_reg    # variance regulariser
    #         #   + 0.05 * bound_pen    # stay inside bounds
    #         )

    #         loss.backward()
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()
    #         # print('nb uodates til nan')
    #         # optimizer.zero_grad()
    #         # # Compute negative log likelihood.
    #         # log_prob = model.log_prob(actions_batch, input_batch)
    #         # nll_loss = -log_prob.mean()
    #         # tot_nll += nll_loss.item() * actions_batch.size(0)
    #         # with torch.no_grad():
    #         #     base_mean, _ = model.base_net(input_batch)
    #         #     actions_pred,_ = model.forward(base_mean, input_batch)
    #         # mse_loss = nn.MSELoss()(actions_pred, actions_batch)
    #         # sigma_reg = (log_std.clamp(min=-4.0, max=1.0) ** 2).mean()
    #         # tot_mse += mse_loss.item()*actions_batch.size(0)
    #         # loss = nll_loss + 0.1*mse_loss+0.01*bound_penalty
    #         # loss.backward()
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         # optimizer.step()
    #     train_mean_nll = tot_nll/len(train_loader.dataset)
    #     train_mean_mse = tot_mse/len(train_loader.dataset)
    #     train_nll.append(train_mean_nll)
    #     train_mse.append(train_mean_mse)
        

    #     model.eval()
    #     tot_nll, tot_mse = 0.0, 0.0
    #     with torch.no_grad():
    #         model.eval()
    #         for input_batch, actions_batch in val_loader:
    #             input_batch = input_batch.to(args.device)
    #             actions_batch = actions_batch.to(args.device)
    #             log_prob = model.log_prob(actions_batch, input_batch)
    #             loss = -log_prob.mean()
    #             tot_nll += loss.item() * actions_batch.size(0)
    #             base_mean, _ = model.base_net(input_batch)
    #             actions_pred,_ = model.inverse(base_mean, input_batch)
    #             tot_mse += nn.MSELoss()(actions_pred, actions_batch).item()*actions_batch.size(0)
    #     avg_val_loss = tot_nll/len(val_loader.dataset)
    #     val_mean_mse = tot_mse/len(val_loader.dataset)
    #     val_nll.append(avg_val_loss)
    #     val_mse.append(val_mean_mse)

    train_losses = []

    val_losses = []

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_train_mse = 0.0
        for input_batch, actions_batch in train_loader:
            input_batch = input_batch.to(args.device)
            actions_batch = actions_batch.to(args.device)
            optimizer.zero_grad()
            # Compute negative log likelihood.
            log_prob = model.log_prob(actions_batch, input_batch)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            base_mean, _ = model.conditional_base(input_batch)
            actions_pred,_ = model.inverse(base_mean, input_batch)
            total_train_mse += nn.functional.mse_loss(actions_pred, actions_batch).item()
            total_loss += loss.item() * actions_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        

        total_val_loss = 0.0
        total_val_mse = 0.0
        with torch.no_grad():
            for input_batch, actions_batch in val_loader:
                input_batch = input_batch.to(args.device)
                actions_batch = actions_batch.to(args.device)
                log_prob = model.log_prob(actions_batch, input_batch)
                loss = -log_prob.mean()
                total_val_loss += loss.item() * actions_batch.size(0)
            val_loss = total_val_loss / len(val_loader.dataset)
            val_losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_loss:.4f} || Val Loss: {val_loss:.4f}")

        
        # if (epoch + 1) % 10 == 0:
        # print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_nll[-1]:.4f} || Val Loss: {avg_val_loss:.4f} || train_mse={train_mse[-1]:.4f} || val_mse={val_mse[-1]:.4f}")
        
        scheduler.step(val_loss)
        # if avg_val_loss < best_vals["val_loss"]:
        #     best_vals["train_nll"] = train_mean_nll
        #     best_vals["train_mse"] = train_mean_mse
        #     best_vals["val_nll"]   = avg_val_loss
        #     best_vals["val_mse"]   = val_mean_mse
        early_stopping(val_loss, model)
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
        ax.scatter(raw_states[:10000,0],
                   raw_states[:10000,1],
                   raw_actions.flatten()[:10000],
                   color='blue', marker='o',
                   label='True Actions', alpha=0.5)
        ax.scatter(raw_states[:10000,0],
                   raw_states[:10000,1],
                   mean_action.flatten()[:10000],
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
        fig, ax = plt.subplots(nrows=n_row, ncols=2, figsize=(10,10))
        ax = ax.flatten()

        for i in range(action_dim):
            ax[i].scatter(raw_actions[:5000,i], mean_action[:5000,i], marker='x', color='red', label='Predicted Mean Actions', alpha=0.6)
            ax[i].set_xlabel(f'true action dim{i}')
            ax[i].set_ylabel(f'predicted action dim {i}')
            ax[i].grid(True)
            x_min, x_max = ax[i].get_xlim()
            x=np.linspace(x_min,x_max,100)
            ax[i].plot(x,x,color='k',label='x=y')
            ax[i].legend(loc='best')
        # action_dim = raw_actions.shape[1]
        # n_row = int(np.ceil(action_dim/2))
        # fig, axes = plt.subplots(n_row, 2, figsize=(10,5*n_row))
        # axes = axes.flatten()
        # for i in range(action_dim):
        #     ax = axes[i]
        #     ax.scatter(raw_actions[:1000,i],
        #                mean_action[:1000,i],
        #                marker='x', color='red',
        #                alpha=0.6, label='Pred Mean')
        #     ax.plot(*ax.get_xlim(), *ax.get_xlim(), color='k', label='ideal')
        #     ax.set_xlabel(f'True a[{i}]')
        #     ax.set_ylabel(f'Pred a[{i}]')
        #     ax.grid(True)
        #     ax.legend(loc='best')

        # for j in range(action_dim, len(axes)):
        #     axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.model_type}/{args.input_type}_fit_cycle_{args.cycle}.svg")
    plt.close()

    return test_nll, test_mse