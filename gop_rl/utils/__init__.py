import numpy as np
import pandas as pd
import ast
from typing import Union

def set_seeds(seed:int, nb_training_cycles:int = 1):
    np.random.seed(seed)
    return np.random.randint(0, 2**32 - 1, size=nb_training_cycles )

def augment_data(data:pd.DataFrame):
    data['prev_actions'] = data['actions'].shift(1)
    data['prev_states'] = data['states'].shift(1)

    mask = data['episode'] != data['episode'].shift(1)
    data = data[~mask].reset_index(drop=True)
    return data


def data_processor(data:Union[str, pd.DataFrame], args:dict):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be a file path or a pandas DataFrame.")
    df['states'] = df['states'].apply(lambda x: ast.literal_eval(x))
    
    if isinstance(df['actions'].iloc[0], str):
        df['actions'] = df['actions'].apply(lambda x: ast.literal_eval(x))
        actions_mat = np.stack(df['actions'].to_numpy())
    else:
        actions_mat = np.array(df['actions'])
    df['targets'] = df['actions'].copy()
    state_mean, state_std = None, None
    actions_mean, actions_std = None, None
    if args.standardize:
        state_mat = np.stack(df['states'].to_numpy())
        state_mean = state_mat.mean(axis=0)
        state_std  = state_mat.std(axis=0) + 1e-8
        df['states'] = df['states'].apply(lambda x: (x - state_mean) / state_std)
        actions_mean = actions_mat.mean(axis=0)
        actions_std  = actions_mat.std(axis=0) + 1e-8
        df['actions'] = df['actions'].apply(lambda x: (x - actions_mean) / actions_std)
    
    augmented_data = augment_data(df)
    # augmented_data['prev_states'] = augmented_data['prev_states'].apply(lambda x: np.array(x))
    return augmented_data, state_mean, state_std, actions_mean, actions_std


class RunningStandardizer:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0  # For calculating variance
        
    def update(self, x):
        # If x is multi-dimensional, flatten it
        if hasattr(x, 'shape') and len(x.shape) > 1:
            x = x.flatten()
        
        # For each value in x
        for val in x:
            self.count += 1
            delta = val - self.mean
            self.mean += delta / self.count
            delta2 = val - self.mean
            self.M2 += delta * delta2
    
    def standardize(self, x):
        if self.count < 2:
            return np.zeros_like(x)
        
        # Calculate standard deviation
        variance = self.M2 / self.count
        std = np.sqrt(variance)
        
        # Avoid division by zero
        if std == 0:
            return np.zeros_like(x)
        
        return (x - self.mean) / std
    
def prepare_data(processed_data:pd.DataFrame, input_type:str):
    """Prepare inputs and targets for model training based on input type"""
    inputs = []
    targets = []
    
    for _, row in processed_data.iterrows():
        state = np.array(row.states, dtype=np.float32)
        action = np.array(row.targets, dtype=np.float32)
        
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
        targets.append(action)
    
    return np.stack(inputs), np.stack(targets)