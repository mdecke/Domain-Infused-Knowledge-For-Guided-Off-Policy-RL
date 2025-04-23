import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from gop_rl.modeling import mle, cnf

def main():
    parser = argparse.ArgumentParser(description="train expert model with expert data.")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Name of the environment.")
    parser.add_argument("--model_type", type=str, default="mle", help="Type of model to train.")
    parser.add_argument("--input_type", type=str, default="state", help="Type of input for the model.")
    parser.add_argument("--epochs", type=int, default=100, help="of gradient steps.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--early_stopping", type=int, default=10, help="How long we look for over fitting.")
    parser.add_argument("--data_dir", type=str, default="data/pendulum", help="Directory to save expert data.")
    parser.add_argument("--output_dir", type=str, default="outputs/pendulum", help="Directory to save plots.")
    parser.add_argument("--action_limit", type=float, default=2.0, help="Action limit for the environment.")
    parser.add_argument("--standardize", type=bool, default=True, help="Z-score standardization of model inputs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    if args.env_name == 'Pendulum-v1':
        csv_file_name = f'data/pendulum/elqr_5000_episodes.csv'
    elif args.env_name == 'Ant-v4':
        csv_file_name = f'data/ant/ppo_250_episodes.csv'
    elif args.env_name == 'Walker2d-v4':
        csv_file_name = f'data/walker/ppo_250_episodes.csv'
    else:  
        raise ValueError("Invalid environment name. Must be 'Pendulum-v1', 'Ant-v4' or 'Walker2d-v4'")
    

    if args.model_type == 'mle':
        model,history,test_inputs_np, test_labels_np= mle.train_model(args,csv_file_name)
        mle.plot_metrics(history, f'{args.output_dir}/{args.input_type}_mle_metrics.svg')
        test_nll, test_mse = mle.test_model(model,raw_states=test_inputs_np, raw_actions=test_labels_np,args=args)
        print(f'[INFO] Test NLL: {test_nll:.4f}, Test MSE: {test_mse:.4f}')
    elif args.model_type == 'cnf':
        cnf.train_model(args,csv_file_name)
    else:
        raise ValueError("Invalid model type. Must be 'mle' or 'cnf'")
    
    print('[INFO] Model trained and saved.')
    
   
if __name__ == "__main__":
    main()