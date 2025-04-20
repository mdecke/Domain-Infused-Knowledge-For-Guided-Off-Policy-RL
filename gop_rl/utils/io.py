import torch
import numpy as np
from torch.distributions import Normal
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


def set_seeds(seed:int, nb_training_cycles:int = 1):
    np.random.seed(seed)
    return np.random.randint(0, 2**32 - 1, size=nb_training_cycles )
    

def handle_exploration(exploration_type:str, state_dim:int, action_dim:int, act_lim:float,device:str): #handle different size inputs?
    match exploration_type:
        case 'gaussian':
            explorator = Normal(loc=0, scale=0.2)
        case 'ou':
            explorator = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.1)
        case 'mle':
            explorator = mle.MLE(state_dim=state_dim, action_dim=action_dim, act_lim=act_lim).to(device)
        case 'cnf':
            explorator = cnf.CNF(state_dim=state_dim, action_dim=action_dim, act_lim=act_lim).to(device)
        case _:
            raise ValueError(f"Unknown exploration type: {exploration_type}")
    return explorator
