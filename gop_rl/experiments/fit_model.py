import os
import argparse
import pandas as pd

from gop_rl.modeling import mle, cnf
from gop_rl.utils import set_seeds



def main():
    parser = argparse.ArgumentParser(description="train expert model with expert data.")
    parser.add_argument("--n_cycles", type=int, default=5, help="Number for training cycles.")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Name of the environment.")
    parser.add_argument("--model_type", type=str, default="mle", help="Type of model to train.")
    parser.add_argument("--input_type", type=str, default="state", help="Type of input for the model.")
    parser.add_argument("--epochs", type=int, default=100, help="of gradient steps.")
    parser.add_argument("--n_grad_steps", type=int, default=32, help="Number of gradient steps per update.")
    parser.add_argument("--nb_traj", type=int, default=20, help="Number of trajectories to sample.")
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
    
    for i in range(args.n_cycles):
        args.seed = seeds[i]
        args.cycle = i+1
        print(f'[INFO] Cycle {i+1}/{args.n_cycles}')
        if args.model_type == 'mle':
            print('[INFO] Training MLE model...')
            model,history, test_inputs_np, test_labels_np= mle.train(args,csv_file_name)
            print([f'[INFO] Model saved to {args.output_dir}/{args.model_type}/{args.input_type}_model_cycle{i+1}.pt'])
            mle.plot_metrics(history, f'{args.output_dir}/{args.model_type}/{args.input_type}_metrics_cycle{i+1}.svg')
            print(f'[INFO] Test metrics saved to {args.output_dir}/{args.model_type}/{args.input_type}_model_cycle{i+1}.pt')
            test_nll, test_mse = mle.test_model(model,raw_states=test_inputs_np, raw_actions=test_labels_np,args=args)
            print(f'[INFO] Test NLL: {test_nll:.4f}, Test MSE: {test_mse:.4f}')
        elif args.model_type == 'cnf':
            print('[INFO] Training CNF model...')
            cnf.train(args,csv_file_name)
        else:
            raise ValueError("Invalid model type. Must be 'mle' or 'cnf'")

        for step in range(len(history['train_nll'])):
            all_records.append({
                'model'      : args.model_type,
                'cycle'      : i + 1,
                'step'       : step,
                'train_nll'  : history['train_nll'][step],
                'val_nll'    : history['val_nll'][step],
                'train_mse'  : history['train_mse'][step],
                'val_mse'    : history['val_mse'][step],
                'test_nll'   : test_nll,
                'test_mse'   : test_mse,
            })
        
    
    df = pd.DataFrame(all_records)
    out_path = f"{args.data_dir}/{args.model_type}all_cycles_history.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote full-cycle history to {out_path}")
if __name__ == "__main__":
    main()