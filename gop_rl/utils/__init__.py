import numpy as np
import pandas as pd
import ast
from typing import Union
import gym

def set_seeds(seed:int, nb_training_cycles:int = 1):
    np.random.seed(seed)
    return np.random.randint(0, 2**32 - 1, size=nb_training_cycles )

def augment_data(data:pd.DataFrame):
    data['prev_actions'] = data['actions'].shift(1, fill_value=0)
    data['prev_states'] = data['states'].shift(1, fill_value=0)

    mask = data['episode'] != data['episode'].shift(1)
    data = data[~mask].reset_index(drop=True)
    return data


def prepare_data(processed_data:pd.DataFrame, input_type:str):
    """Prepare inputs and targets for model training based on input type"""
    inputs = []
    targets = []
    
    for _, row in processed_data.iterrows():
        
        state = np.array(row.states, dtype=np.float32)
        target = np.array(row.targets, dtype=np.float32)
        
        if input_type == 'state':
            x = state
        elif input_type == 'state_action':
            prev_action = np.array(row.prev_actions, dtype=np.float32)
            x = np.concatenate([state, prev_action.reshape(-1)], axis=0)
        elif input_type == 'prev_state_action':
            prev_state = np.array(row.prev_states, dtype=np.float32)
            prev_action = np.array(row.prev_actions, dtype=np.float32)
            x = np.concatenate([state, prev_action.reshape(-1), prev_state], axis=0)
        elif input_type == 'state_prev_state':
            prev_state = np.array(row.prev_states, dtype=np.float32)
            x = np.concatenate([state, prev_state], axis=0)
        else:
            raise ValueError("Invalid input type. Choose from 'state', 'state_action', 'prev_state_action', or 'state_prev_state'.")
        
        inputs.append(x)
        targets.append(target)
    return np.stack(inputs), np.stack(targets)


def data_processor(data:Union[str, pd.DataFrame], args:dict, test:bool=False):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be a file path or a pandas DataFrame.")
    
    if isinstance(df['states'].iloc[0], str):
        try:
            df['states'] = df['states'].apply(lambda x: ast.literal_eval(x))
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing states: {e}")
            print(f"Sample state string: {df['states'].iloc[0]}")
            raise

    if isinstance(df['actions'].iloc[0], str):
        try:
            df['actions'] = df['actions'].apply(lambda x: ast.literal_eval(x) if '[' in x else float(x))
            actions_mat = np.stack([np.array([a]) if isinstance(a, (int, float)) else np.array(a) for a in df['actions']])
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing actions: {e}")
            print(f"Sample action string: {df['actions'].iloc[0]}")
            raise
    else:
        actions_mat = np.array(df['actions']).reshape(-1, 1)  # Ensure 2D
    
    df['targets'] = df['actions'].copy()
    
    if test:
        return augment_data(df), None, None, None, None
    
    state_mean, state_std = None, None
    actions_mean, actions_std = None, None
    
    if args.standardize:
        if hasattr(args, 'state_mean') and args.state_mean is not None:
            state_mean = args.state_mean
            state_std  = args.state_std
        else:
            state_mat = np.stack([np.array(s) for s in df['states'].to_numpy()])
            state_mean = state_mat.mean(axis=0)
            state_std  = state_mat.std(axis=0) + 1e-8
        
        df['states'] = df['states'].apply(lambda x: (np.array(x) - state_mean) / state_std)

        if hasattr(args, 'actions_mean') and args.actions_mean is not None:
            actions_mean = args.actions_mean
            actions_std  = args.actions_std
        else:
            actions_mean = actions_mat.mean(axis=0)
            actions_std  = actions_mat.std(axis=0) + 1e-8
        
        df['actions'] = df['actions'].apply(
            lambda x: (x - actions_mean) / actions_std if isinstance(x, (int, float))
            else (np.array(x) - actions_mean) / actions_std
        )
    else:
        state_mean = np.zeros_like(df['states'].iloc[0])
        state_std = np.ones_like(df['states'].iloc[0])
        actions_mean = np.zeros_like(df['actions'].iloc[0])
        actions_std = np.ones_like(df['actions'].iloc[0])
    
    augmented_data = augment_data(df)
    return augmented_data, state_mean, state_std, actions_mean, actions_std




class SparseRewardWrapper(gym.Wrapper):
    
    def __init__(self, env: gym.Env, threshold: float = 0.0):
        super().__init__(env)
        self.threshold = threshold

    def step(self, action):
        obs, r_dense, done, info = None, None, None, None
        # Gymnasium API:
        if hasattr(self.env, "step"):
            obs, r_dense, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        else:
            obs, r_dense, done, info = self.env.step(action)

        # apply sparsification
        r_sparse = 1.0 if r_dense >= self.threshold else 0.0

        # repackage
        if hasattr(self.env, "step") and terminated is not None:
            return obs, r_sparse, terminated, truncated, info
        else:
            return obs, r_sparse, done, info


class ParamOverrideWrapper(gym.Wrapper):
    """
    Wraps any Gym env and, at construction time, sets
    any attributes on `env.unwrapped` given by kwargs.
    E.g. ParamOverrideWrapper(env, m=2.0, l=0.5) will do
         env.unwrapped.m = 2.0; env.unwrapped.l = 0.5
    """
    def __init__(self, env: gym.Env, **overrides):
        super().__init__(env)
        base = env.unwrapped
        for key, val in overrides.items():
            if hasattr(base, key):
                setattr(base, key, val)
            else:
                raise AttributeError(f"{env.spec.id!r} has no attribute {key}")
