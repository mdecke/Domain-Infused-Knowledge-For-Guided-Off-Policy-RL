import os

import numpy as np
import pandas as pd
import control
import gymnasium as gym
import time

class LQRController:
    def __init__(self,
                 mass, rod_length, gravity, dt,
                 action_limits:tuple,
                 Q = np.diag([50, 0.5]),
                 R = np.array([[0.2]])):
        """
        LQR Controller for the Pendulum environment.
        Parameters:
        mass: mass of the pendulum
        rod_length: length of the pendulum
        gravity: gravity
        action_limits: limits of the action space
        dt: time step
        Q: state cost matrix
        R: control effort cost
        """
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2) * 1/3
        self.action_limits = action_limits
        self.dt = dt

        self.Q = Q
        self.R = R

        self.A = np.array([[0, 1],
                           [3 * self.g / (self.l * 2), 0.0]])
        self.B = np.array([[0], [1 / self.I]])

        self.K = control.lqr(self.A, self.B, self.Q, self.R)[0]

    def compute_control(self, state: np.ndarray, state_d: np.ndarray = np.array([0, 0])):
        """
        Compute the control action using the LQR feedback law.
        """
        u = - self.K @ (state - state_d)
        return np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])

class EnergyShapingController:
    def __init__(self,
                 mass: float,
                 rod_length: float,
                 gravity: float,
                 dt: float,
                 action_limits: tuple):
        """
        Energy Shaping Controller for the Pendulum environment.
        See https://underactuated.mit.edu/acrobot.html#section6
        Parameters:
            mass: mass of the pendulum
            rod_length: length of the pendulum
            gravity: gravity
            action_limits: limits of the action space
            dt: time step
        """
        self.m = mass
        self.l = rod_length
        self.g = gravity
        self.I = self.m * (self.l ** 2) * 1/3
        self.dt = dt
        self.action_limits = action_limits

    def compute_total_energy(self, state:np.ndarray):
        kinetic_energy = 1/2 * self.I * (state[1] ** 2)
        potential_energy = self.m * self.g * self.l/2 * np.cos(state[0])
        E = kinetic_energy + potential_energy
        return kinetic_energy, potential_energy, E

    def compute_desired_energy(self):
        E_d = self.m * self.g * self.l/2
        return E_d

    def get_action(self, state:np.ndarray, kp:float = 1.0):
        E_d = self.compute_desired_energy()
        _, _, E = self.compute_total_energy(state)

        E_err = E - E_d # Energy error.
        u = - kp * state[1] * E_err
        return np.clip(u, a_min=self.action_limits[0], a_max=self.action_limits[1])

def sim(args):
    env = gym.make(args.env_name)

    pendulum_params = {
        "mass": env.unwrapped.m,
        "rod_length": env.unwrapped.l,
        "gravity": 10.0,
        "action_limits": (env.action_space.low, env.action_space.high),
        'dt': env.unwrapped.dt
    }

    ANGLE_SWITCH_THRESHOLD_DEG = 18
    EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5

    energy_controller = EnergyShapingController(**pendulum_params)
    lqr_controller = LQRController(**pendulum_params)

    collected_data = []
    duration_episodes = []

    for i in range(args.num_episodes):
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

            obs, _, _, _, _ = env.step(action)
            next_angle = np.arctan2(obs[1], obs[0])

            if (abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG)) and \
               (abs(next_angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG)):
                upright_angle_buffer.append(angle)
            if len(upright_angle_buffer) > 40:
                done = True

            collected_data.append([
                i,
                action.squeeze(),
                state.tolist(),
                ctrl_type
            ])
            
            state = obs.squeeze().copy()

        duration_episodes.append(time.time() - time_start_episode)

    env.close()
    print(f'Total simulation time: {sum(duration_episodes):.2f} s for {args.num_episodes} episodes')

    # Save data
    col_names = ['episode', 'actions', 'states', 'ctrl_type']
    df = pd.DataFrame(collected_data, columns=col_names)

    os.makedirs(args.data_dir, exist_ok=True)
    csv_file_name = f'{args.data_dir}/elqr_{args.num_episodes}_episodes.csv'
    df.to_csv(csv_file_name, index=False)

    print(f'Data saved at: {csv_file_name}')
