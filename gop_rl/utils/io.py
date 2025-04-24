import torch
from torch.distributions import Normal
import pandas as pd
import numpy as np
# from gop_rl.modeling import mle, cnf



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

    

def handle_exploration(args): #handle different size inputs?
    match args.exploration_type:
        case 'gaussian':
            explorator = Normal(loc=0, scale=0.1)
            noise = True
        case 'ou':
            explorator = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.1)
            noise = True
        case 'mle':
            explorator = mle.MLE(state_dim=args.state_dim, action_dim=args.action_dim, act_lim=args.act_lim).to(args.device)
            noise = False
        case 'cnf':
            explorator = cnf.CNF(state_dim=args.state_dim, action_dim=args.action_dim, act_lim=args.act_lim).to(args.device)
            noise = False
        case _:
            raise ValueError(f"Unknown exploration type: {args.exploration_type}, must be 'gaussian', 'ou', 'mle' or 'cnf'")
    return explorator, noise

def handle_input():
    pass

def make_episode_batches(df:pd.DataFrame, batch_size:int, train_episodes:np.ndarray) -> list:
    episode_batch = np.random.choice(train_episodes, size=batch_size, replace=False)
    pass



