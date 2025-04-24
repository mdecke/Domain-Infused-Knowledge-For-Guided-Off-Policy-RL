import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from typing import List, Dict

from gop_rl.modeling import mle, cnf
from gop_rl.utils import set_seeds



def plot_aggregate(mean_train, std_train, mean_val, std_val,
                   epochs, test_nlls, test_mses, outdir, label):
    """Learning-curve ribbons + test metrics bars."""
    # ribbon plot
    plt.figure(figsize=(9, 4.5))
    for m, s, c, name in [(mean_train, std_train, "tab:blue", "Train NLL"),
                          (mean_val,   std_val,   "tab:orange", "Val NLL")]:
        plt.plot(epochs, m, color=c, label=name)
        plt.fill_between(epochs, m - s, m + s, color=c, alpha=0.25)
    plt.xlabel("Epoch");  plt.ylabel("NLL");  plt.title("NLL mean ± std over cycles")
    plt.legend();  plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_nll_ribbon.svg")); plt.close()

    # bar plot for test metrics
    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    for i, (vals, title) in enumerate([(test_nlls, "Test NLL"),
                                       (test_mses, "Test MSE")]):
        ax[i].bar(0, np.mean(vals), yerr=np.std(vals), color="tab:green",
                  capsize=6, width=0.6)
        ax[i].set_xticks([]); ax[i].set_title(title)
    plt.suptitle(f"{label}: mean ± std over {len(test_nlls)} cycles")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(outdir, f"{label}_test_bars.svg")); plt.close()

def aggregate_histories(histories: List[Dict[str, List[float]]]):
    """Stacks variable-length histories → mean & std (ignores trailing NaNs)."""
    max_ep = max(len(h["train_nll"]) for h in histories)
    def stack(key):
        arr = np.full((len(histories), max_ep), np.nan, dtype=np.float32)
        for i, h in enumerate(histories):
            arr[i, : len(h[key])] = h[key]
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr,  axis=0)
        return mean, std
    mean_train, std_train = stack("train_nll")
    mean_val,   std_val   = stack("val_nll")
    return mean_train, std_train, mean_val, std_val, np.arange(1, len(mean_train)+1)


def plot_aggregate(mean_train, std_train, mean_val, std_val,
                   epochs, test_nlls, test_mses, outdir, label):
    """Learning-curve ribbons + test metrics bars."""
    # ribbon plot
    plt.figure(figsize=(9, 4.5))
    for m, s, c, name in [(mean_train, std_train, "tab:blue", "Train NLL"),
                          (mean_val,   std_val,   "tab:orange", "Val NLL")]:
        plt.plot(epochs, m, color=c, label=name)
        plt.fill_between(epochs, m - s, m + s, color=c, alpha=0.25)
    plt.xlabel("Epoch");  plt.ylabel("NLL");  plt.title("NLL mean ± std over cycles")
    plt.legend();  plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_nll_ribbon.svg")); plt.close()

    # bar plot for test metrics
    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    for i, (vals, title) in enumerate([(test_nlls, "Test NLL"),
                                       (test_mses, "Test MSE")]):
        ax[i].bar(0, np.mean(vals), yerr=np.std(vals), color="tab:green",
                  capsize=6, width=0.6)
        ax[i].set_xticks([]); ax[i].set_title(title)
    plt.suptitle(f"{label}: mean ± std over {len(test_nlls)} cycles")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(outdir, f"{label}_test_bars.svg")); plt.close()


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
    
    histories = []
    test_nlls = []
    test_mses = []
    for i in range(args.n_cycles):
        args.seed = seeds[i]
        args.cycle = i+1
        print(f'[INFO] Cycle {i+1}/{args.n_cycles}')
        if args.model_type == 'mle':
            print('[INFO] Training MLE model...')
            model,history, test_inputs_np, test_labels_np= mle.train_model(args,csv_file_name)
            print(f'[INFO] Saving model to {args.output_dir}/{args.model_type}/{args.input_type}_model_cycle{i+1}.pt')
            mle.plot_metrics(history, f'{args.output_dir}/{args.model_type}/{args.input_type}_metrics_cycle{i+1}.svg')
            test_nll, test_mse = mle.test_model(model,raw_states=test_inputs_np, raw_actions=test_labels_np,args=args)
            print(f'[INFO] Test NLL: {test_nll:.4f}, Test MSE: {test_mse:.4f}')
        elif args.model_type == 'cnf':
            print('[INFO] Training CNF model...')
            cnf.train_model(args,csv_file_name)
        else:
            raise ValueError("Invalid model type. Must be 'mle' or 'cnf'")
        histories.append(history)
        test_nlls.append(test_nll)
        test_mses.append(test_mse)
    mean_tr, std_tr, mean_val, std_val, epochs = aggregate_histories(histories)
    plot_aggregate(mean_tr, std_tr, mean_val, std_val,
                   epochs, test_nlls, test_mses,
                   outdir=f"{args.output_dir}/{args.model_type}",
                   label=args.input_type)

    print('[INFO] Model trained and saved.')
    
   
if __name__ == "__main__":
    main()