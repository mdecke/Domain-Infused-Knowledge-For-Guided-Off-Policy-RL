import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import time
import math
import matplotlib.pyplot as plt
import ast

from gop_rl.expert_controllers import elqr

def main():
    parser = argparse.ArgumentParser(description="Generate expert data for a given environment.")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Name of the environment.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode - check gym of episode length.")
    parser.add_argument("--data_dir", type=str, default="data/pendulum", help="Directory to save expert data.")
    parser.add_argument("--output_dir", type=str, default="outputs/pendulum/elqr", help="Directory to save plots.")
    parser.add_argument("--action_limit", type=float, default=2.0, help="Action limit for the environment.")
    parser.add_argument("--kp", type=float, default=1.0, help="Proportional gain for the controller.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    env = gym.make(args.env_name)
    
    if args.env_name == 'Pendulum-v1':
        pendulum_params = {"mass": env.unwrapped.m,
                           "rod_length": env.unwrapped.l,
                           "gravity": 10.0,
                           "action_limits": (env.action_space.low, env.action_space.high),
                           'dt': env.unwrapped.dt}
        ANGLE_SWITCH_THRESHOLD_DEG = 18
        EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5 # deg
    else:
        raise ValueError(f"Unsupported ELQR for: {args.env_name}")
        

    energy_controller = elqr.EnergyShapingController(**pendulum_params)
    lqr_controller = elqr.LQRController(**pendulum_params)
    
    duration_episodes = []
    collected_data = []
    
    for i in range(args.num_episodes):
        # print(f'Episode {i}')
        obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
        done = False
        state = obs.squeeze().copy()
        upright_angle_buffer = []
        ctrl_type = None
        time_start_episode = time.time()

        while not done:
            angle = np.arctan2(obs[1], obs[0])
            pos_vel = np.array([angle, obs[2]]).squeeze()

            if abs(angle) < np.deg2rad(ANGLE_SWITCH_THRESHOLD_DEG):
                action = lqr_controller.compute_control(pos_vel)
                ctrl_type = 'LQR'
            else:
                action = energy_controller.get_action(pos_vel)
                ctrl_type = 'EnergyShaping'

            obs ,_ ,_ ,_, _ = env.step(action)
            next_angle = np.arctan2(obs[1], obs[0])
            if (abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG)) and (abs(next_angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG)):
                upright_angle_buffer.append(angle)
            if len(upright_angle_buffer) > 40:
                done = True

            collected_data.append([i,
                                         action.squeeze(),
                                         state.tolist(),
                                         ctrl_type])
            
            state = obs.squeeze().copy() # use .copy() for arrays because of the shared memory issues

        time_end_episode = time.time()
        duration_episodes.append(time_end_episode - time_start_episode)
    env.close()
    
    print('It took total of', sum(duration_episodes), 'seconds to run', args.num_episodes, 'episodes')

    print('... Saving data ...')
    col_names = ['episode', 'actions', 'states', 'ctrl_type']
    df = pd.DataFrame(collected_data, columns=col_names)

    csv_file_name = f'{args.data_dir}/elqr_{args.num_episodes}_episodes.csv'
    df.to_csv(csv_file_name, index=False)

    # Plot state.
    
    print('Plotting...')
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