import torch
from torch.distributions import Normal
import pandas as pd
import numpy as np
from gop_rl.modeling import mle, cnf



class OrnsteinUhlenbeckNoise:
    def __init__(self,theta: float,sigma: float,base_scale: float,mean: float = 0,std: float = 1) -> None:
        super().__init__()
        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.distribution = Normal(loc=torch.tensor(mean, dtype=torch.float32),
                                   scale=torch.tensor(std, dtype=torch.float32))

    def sample(self, size:torch.Size = torch.Size([1,1])) -> torch.Tensor:
        if hasattr(self.state, "shape") and self.state.shape != torch.Size(size):
            self.state = 0
        self.state += -self.state * self.theta + self.sigma * self.distribution.sample(size)

        return self.base_scale * self.state

    

def chose_exploration(args): #handle different size inputs?

    if args.input_type == 'state':
        input_dim = args.state_dim
    elif args.input_type == 'state_action':
        input_dim = args.state_dim + args.action_dim
    elif args.input_type == 'prev_state_action':
        input_dim = (2*args.state_dim) + args.action_dim
    elif args.input_type == 'state_prev_state':
        input_dim = args.state_dim * 2
    else:
        raise ValueError('unacceptable input type. Must be state, state_action, prev_state_action, state_prev_state')
    
    if args.env_name == 'Pendulum-v1':
        # Original state dimension is 3, but transformed it becomes 2 (angle, angular velocity)
        # Adjust the input dimensions accordingly
        if args.input_type == 'state':
            input_dim = 2  # Theta, angular velocity
        elif args.input_type == 'state_action':
            input_dim = 2 + args.action_dim  # Theta, angular velocity + action
        elif args.input_type == 'prev_state_action':
            input_dim = (2*2) + args.action_dim  # 2 states (current + previous) + action
        elif args.input_type == 'state_prev_state':
            input_dim = 2 * 2  # 2 states (current + previous)


    match args.exploration_type:
        case 'gaussian':
            if args.env_name =='Pendulum':
                explorator = Normal(loc=0, scale=0.2)
            else:
                explorator = Normal(loc=0.0, scale=0.1)
            noise = True
        case 'ou':
            explorator = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.1)
            noise = True
        case 'mle':
            print('input dim is: ',input_dim)
            explorator = mle.ActionMLE(state_dim=input_dim, action_dim=args.action_dim, action_lim=args.action_high).to(args.device)
            explorator.load_state_dict(torch.load('prev_state_action_3.pth'))
            noise = False
        case 'cnf':
            explorator = cnf.ConditionalNormalizingFlow(condition_dim=input_dim, n_flows=args.n_flows, latent_dim=args.action_dim).to(args.device)
            explorator.load_state_dict(torch.load('state_1.pth'))
            noise = False
        case _:
            raise ValueError(f"Unknown exploration type: {args.exploration_type}, must be 'gaussian', 'ou', 'mle' or 'cnf'")
    return explorator, noise



def handle_input(obs,prev_obs,prev_action,args):
    parts = []
    # you can choose your own keywords here
    if args.env_name == 'Pendulum-v1':
        obs_theta = np.arctan2(obs[:, 1], obs[:, 0]).reshape(-1, 1)
        obs_combined = np.hstack((obs_theta, obs[:, 2].reshape(-1, 1)))
        
        prev_obs_theta = np.arctan2(prev_obs[:, 1], prev_obs[:, 0]).reshape(-1, 1)
        prev_obs_combined = np.hstack((prev_obs_theta, prev_obs[:, 2].reshape(-1, 1)))
        
        obs = obs_combined
        prev_obs = prev_obs_combined
    
    use_scaling = args.scale == True or args.scale == 1
    
    if use_scaling:
        # Check if scalers exist, otherwise initialize them
        if not hasattr(args, 'obs_scaler') or not hasattr(args, 'act_scaler'):
            raise ValueError("Scaling requested but scalers not initialized. Initialize obs_scaler and act_scaler first.")
        scaled_state = args.obs_scaler.transform(obs)
        scaled_prev_states = args.obs_scaler.transform(prev_obs)
        scaled_prev_action = args.act_scaler.transform(prev_action)
    else:
        # If not scaling, use original values
        scaled_state = obs
        scaled_prev_states = prev_obs
        scaled_prev_action = prev_action
    
    if args.input_type == 'state':
        x = scaled_state
    elif args.input_type == 'state_action':
        x = np.concatenate((scaled_state, scaled_prev_action), axis=1)
    elif args.input_type == 'prev_state_action':
        x = np.concatenate((scaled_state, scaled_prev_states,scaled_prev_action), axis=1)
    elif args.input_type == 'state_prev_state':
        x = np.concatenate((scaled_state, scaled_prev_states), axis=1)
    else:
        raise ValueError('Unacceptable input type. Must be state, state_action, prev_state_action, state_prev_state')
    
    # Convert to tensor and return
    return torch.tensor(x, dtype=torch.float32, device=args.device)






