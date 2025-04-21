import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import time
import math
import matplotlib.pyplot as plt
import ast

from gop_rl.expert_controllers import elqr, sbl3_ppo

def main():
    parser = argparse.ArgumentParser(description="Generate expert data for a given environment.")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Name of the environment.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode - check gym of episode length.")
    parser.add_argument("--data_dir", type=str, default="data/pendulum", help="Directory to save expert data.")
    parser.add_argument("--output_dir", type=str, default="outputs/pendulum", help="Directory to save plots.")
    parser.add_argument("--action_limit", type=float, default=2.0, help="Action limit for the environment.")
    parser.add_argument("--kp", type=float, default=1.0, help="Proportional gain for the controller.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.env_name == 'Ant-v4':
        sbl3_ppo.sim(args)
        print('Videos and data saved.')
    if args.env_name == 'Pendulum-v1':
        elqr.sim(args)
        print('Plotting...')
        csv_file_name = f'{args.data_dir}/elqr_{args.num_episodes}_episodes.csv'
        data = pd.read_csv(csv_file_name)
        data['states'] = data['states'].apply(lambda x: ast.literal_eval(x))
        data['angle_state'] = data['states'].apply(lambda x: np.arctan2(x[1], x[0]))
        N_EPISODES = data['episode'].nunique()
        
        n_traj_to_plot = 6
        selected_eps = np.random.choice(a=N_EPISODES, size=n_traj_to_plot,replace=False)

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        for i in selected_eps:
            episode_data = data[data['episode'] == i].reset_index(drop=True)
            time_steps   = episode_data.index

            angles       = episode_data['angle_state']
            angular_vels = episode_data['states'].apply(lambda x: x[2])
            actions      = episode_data['actions']

            ax[0].plot(time_steps, angles,       label=f'Episode {i}')
            ax[1].plot(time_steps, angular_vels, label=f'Episode {i}')
            ax[2].plot(time_steps, actions,      label=f'Episode {i}')

        for a in ax:
            a.grid()
            a.legend(loc='upper right', fontsize='small', ncol=2, framealpha=0.5)

        ax[0].set_title(f"Performance for Episodes {selected_eps}")
        ax[0].set_ylabel("Angle [rad]")
        ax[1].set_ylabel("Angular Velocity [rad/s]")
        ax[2].set_ylabel("Torque [Nm]")
        ax[2].set_xlabel("Time Steps")

        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/elqr_episodes.svg')
        plt.show()

if __name__ == "__main__":
    main()