import os
import argparse
import pandas as pd
import numpy as np

from gop_rl.modeling import mle, cnf
from gop_rl.utils import set_seeds



def main():
    parser = argparse.ArgumentParser(description="train expert model with expert data.")
    parser.add_argument("--n_cycles", type=int, default=5, help="Number for training cycles.")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Name of the environment.")
    parser.add_argument("--model_type", type=str, default="mle", help="Type of model to train.")
    parser.add_argument("--input_type", type=str, default="state", help="Type of input for the model.")
    parser.add_argument("--epochs", type=int, default=2, help="of gradient steps.")
    parser.add_argument("--n_grad_steps", type=int, default=32, help="Number of gradient steps per update.")
    parser.add_argument("--nb_traj", type=int, default=20, help="Number of trajectories to sample.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for CNF training.")
    parser.add_argument("--n_flows", type=int, default=10, help="Number of flows for CNF training.")
    parser.add_argument("--early_stopping", type=int, default=5, help="How long we look for over fitting.")
    parser.add_argument("--data_dir", type=str, default="data/pendulum", help="Directory to save expert data.")
    parser.add_argument("--output_dir", type=str, default="outputs/pendulum", help="Directory to save plots.")
    parser.add_argument("--standardize", type=bool, default=False, help="Z-score standardization of model inputs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training.")
    args = parser.parse_args()

    seeds = set_seeds(args.seed, args.n_cycles)
    print(f'[INFO] Seeds for cycles: {seeds}')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/{args.model_type}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{args.model_type}', exist_ok=True)

    # paths with respect to run file
    if args.env_name == 'Pendulum-v1':      csv_file_name = f'{args.data_dir}/elqr_5000_episodes.csv'
    elif args.env_name == 'Ant-v4':         csv_file_name = f'{args.data_dir}/ppo_250_episodes.csv'
    elif args.env_name == 'Walker2d-v4':    csv_file_name = f'data/walker/ppo_250_episodes.csv'
    else: raise ValueError("Invalid environment name. Must be 'Pendulum-v1', 'Ant-v4' or 'Walker2d-v4'")
    
    all_records = []
    all_metrics = []
    
    for i in range(args.n_cycles):
        args.seed = seeds[i]
        args.cycle = i+1
        print(f'[INFO] Cycle {i+1}/{args.n_cycles}')
        if args.model_type == 'mle':
            print('[INFO] Training MLE model...')
            model,history, test_inputs_np, test_labels_np= mle.train(args,csv_file_name)
            print([f'[INFO] Model saved to {args.output_dir}/{args.model_type}/{args.input_type}_model_cycle{i+1}.pt'])
            mle.plot_metrics(history, args)
            print(f'[INFO] Test metrics saved to {args.output_dir}/{args.model_type}/{args.input_type}_model_cycle{i+1}.pt')
            test_nll, test_mse = mle.test_model(model,raw_states=test_inputs_np, raw_actions=test_labels_np,args=args)
            print(f'[INFO] Test NLL: {test_nll:.4f}, Test MSE: {test_mse:.4f}')
            all_records.append({
                'model'      : args.model_type,
                'cycle'      : i + 1,
                'train_nll':  history['train_nll'],
                'val_nll_last':    history['val_nll'],
                'train_mse'  : history['train_mse'],
                'val_mse'    : history['val_mse'],
                'test_nll'   : test_nll,
                'test_mse'   : test_mse,
            })
            print(f'[INFO] Cycle {i+1} records: {all_records[-1]}')
        elif args.model_type == 'cnf':
            print('[INFO] Training CNF model...')
            model, train_losses, val_losses, X_test, y_test = cnf.train(args,csv_file_name)
            print(f'[INFO] Model saved to {args.output_dir}/{args.model_type}/{args.input_type}_model_cycle{i+1}.pt')
            cnf.plot_metrics(train_losses, val_losses, args)
            print(f'[INFO] Test metrics saved to {args.output_dir}/{args.model_type}/{args.input_type}_model_cycle{i+1}.pt')
            test_nll, test_mse = cnf.test_model(model,X_test,y_test, args)
            print(f'[INFO] Test NLL: {test_nll:.4f}, Test MSE: {test_mse:.4f}')
            all_metrics.append({
                'model'      : args.model_type,
                'cycle'      : i + 1,
                'train_nll':  train_losses,
                'val_nll':    val_losses,
                'train_mse'  : train_losses,
                'val_mse'    : val_losses,
                'test_nll'   : test_nll,
                'test_mse'   : test_mse,
            })
        else:
            raise ValueError("Invalid model type. Must be 'mle' or 'cnf'")

    if args.model_type == 'mle':
        df = pd.DataFrame(all_records)
    else:  # cnf
        df = pd.DataFrame(all_metrics)
        
    out_path = f"{args.data_dir}/{args.model_type}/{args.input_type}_all_cycles_history.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote full-cycle history to {out_path}")

if __name__ == "__main__":
    main()