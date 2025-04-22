import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class DDPGMetrics:
    def __init__(self, data: pd.DataFrame = None, file_path: str = None, 
                 show: bool = True, title: str = None, smooth: int = None):
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        elif data is not None:
            self.df = data
        else:
            raise ValueError("Either `data` or `file_path` must be provided.")

        self.show = show
        self.title = title
        self.smooth = smooth

    def split_losses(self):
        cycles = self.df['cycle'].unique()
        self.policy_losses = pd.DataFrame()
        self.q_losses = pd.DataFrame()
        self.episodic_returns = pd.DataFrame()
        for cycle in cycles:
            cycle_policy_loss = self.df[self.df['cycle'] == cycle]['policy_loss'].reset_index(drop=True)
            cycle_q_loss = self.df[self.df['cycle'] == cycle]['q_loss'].reset_index(drop=True)
            cycle_returns = self.df[self.df['cycle'] == cycle]['return'].dropna().reset_index(drop=True)

            self.policy_losses[f'cycle {cycle}'] = cycle_policy_loss
            self.q_losses[f'cycle {cycle}'] = cycle_q_loss
            self.episodic_returns[f'cycle {cycle}'] = cycle_returns

        # Compute mean/std/upper/lower for policy
        self.policy_losses['mean'] = self.policy_losses.mean(axis=1)
        self.policy_losses['std'] = self.policy_losses.std(axis=1)
        self.policy_losses['upper'] = self.policy_losses['mean'] + self.policy_losses['std']
        self.policy_losses['lower'] = self.policy_losses['mean'] - self.policy_losses['std']

        # Compute mean/std/upper/lower for q
        self.q_losses['mean'] = self.q_losses.mean(axis=1)
        self.q_losses['std'] = self.q_losses.std(axis=1)
        self.q_losses['upper'] = self.q_losses['mean'] + self.q_losses['std']
        self.q_losses['lower'] = self.q_losses['mean'] - self.q_losses['std']

        # Compute mean/std/upper/lower for returns
        self.episodic_returns['mean'] = self.episodic_returns.mean(axis=1)
        self.episodic_returns['std'] = self.episodic_returns.std(axis=1)
        self.episodic_returns['upper'] = self.episodic_returns['mean'] + self.episodic_returns['std']
        self.episodic_returns['lower'] = self.episodic_returns['mean'] - self.episodic_returns['std']

    def smoothing_function(self, data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='same')

    def smooth_data(self):
        window = self.smooth
        # Smooth policy losses
        self.policy_losses['mean'] = self.smoothing_function(self.policy_losses['mean'].to_numpy(), window)
        self.policy_losses['upper'] = self.smoothing_function(self.policy_losses['upper'].to_numpy(), window)
        self.policy_losses['lower'] = self.smoothing_function(self.policy_losses['lower'].to_numpy(), window)
        # Smooth Q losses
        self.q_losses['mean'] = self.smoothing_function(self.q_losses['mean'].to_numpy(), window)
        self.q_losses['upper'] = self.smoothing_function(self.q_losses['upper'].to_numpy(), window)
        self.q_losses['lower'] = self.smoothing_function(self.q_losses['lower'].to_numpy(), window)
        # Smooth returns
        self.episodic_returns['mean'] = self.smoothing_function(self.episodic_returns['mean'].to_numpy(), window)
        self.episodic_returns['upper'] = self.smoothing_function(self.episodic_returns['upper'].to_numpy(), window)
        self.episodic_returns['lower'] = self.smoothing_function(self.episodic_returns['lower'].to_numpy(), window)

        # Recompute the x-axis to match the new length if needed
        self.trainings_steps = np.linspace(0, len(self.policy_losses['mean']), len(self.policy_losses['mean']))
        self.n_episodes = np.linspace(0, len(self.policy_losses['mean']), len(self.episodic_returns['mean']))

    def plot_losses(self, ax=None, plot_p_q: bool = False):
        """
        By default, only the episodic returns are plotted (a single subplot).
        If plot_p_q=True, three subplots are made: returns, policy loss, and Q loss.
        """
        # Prepare data
        self.split_losses()
        self.trainings_steps = np.linspace(0, len(self.policy_losses['mean']), len(self.policy_losses['mean']))
        self.n_episodes = np.linspace(0, len(self.policy_losses['mean']), len(self.episodic_returns['mean']))
        
        # Apply smoothing if desired
        if self.smooth:
            self.smooth_data()

        if not plot_p_q:
            # SINGLE SUBPLOT (just returns)
            if ax is None:
                fig, ax = plt.subplots(1, 1, sharex=False, figsize=(10, 8))

            ax.plot(self.n_episodes, self.episodic_returns['mean'], label=f'{self.title}')
            ax.fill_between(self.n_episodes,
                            self.episodic_returns['lower'],
                            self.episodic_returns['upper'],
                            alpha=0.1)
            ax.set_ylabel('Return')
            ax.set_xlabel('Training Steps')
            ax.set_title(f'Mean Episodic Return (smoothed = {self.smooth})')

            # Draw zero line + top limit
            ax.axhline(y=0, color='k', linewidth=2)
            ax.set_ylim(top=100)

            ax.set_facecolor('lightgrey')
            ax.grid(True)
            ax.legend(loc='upper left', fontsize='x-small')

        else:
            # THREE SUBPLOTS (returns, policy loss, Q loss)
            if ax is None:
                fig, ax = plt.subplots(3, 1, sharex=False, figsize=(10, 8))

            # Episodic returns in ax[0]
            ax[0].plot(self.n_episodes, self.episodic_returns['mean'], label=f'{self.title}')
            # ax[0].fill_between(self.n_episodes,
            #                    self.episodic_returns['lower'],
            #                    self.episodic_returns['upper'],
            #                    alpha=0.1)
            ax[0].set_ylabel('Return')
            ax[0].set_xlabel('Training Steps')
            ax[0].set_title(f'Mean Episodic Return (smoothed = {self.smooth})')
            ax[0].axhline(y=0, color='k', linewidth=2)
            ax[0].set_ylim(top=100)

            # Policy loss in ax[1]
            ax[1].plot(self.trainings_steps, self.policy_losses['mean'], label=f'{self.title}')
            ax[1].set_ylabel('Policy Loss')
            ax[1].set_xlabel('Training Steps')
            ax[1].set_title(f'Mean Policy Loss (smoothed = {self.smooth})')

            # Q loss in ax[2]
            ax[2].plot(self.trainings_steps, self.q_losses['mean'], label=f'{self.title}')
            ax[2].set_ylabel('Value Loss')
            ax[2].set_xlabel('Training Steps')
            ax[2].set_title(f'Mean Q-function Loss (smoothed = {self.smooth})')

            # Styling for all three
            for sub_ax in ax:
                sub_ax.set_facecolor('0.1')
                sub_ax.grid(True)
                sub_ax.legend(loc='upper left', fontsize='x-small')

        if self.show and ax is None:
            # If we created the figure ourselves, show it
            plt.tight_layout()
            plt.show()
        