import os
import argparse
import gymnasium as gym

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from gop_rl.utils.io import set_seeds, handle_exploration
# from gop_rl.utils.rl_metrics_plotter import Plotter
from gop_rl.agents import ddpg as agent

def main():
    
    parser = argparse.ArgumentParser(description='Run RL agent')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='Environment name')
    parser.add_argument('--agent', type=str, default='ddpg', help='Agent name')
    parser.add_argument('--exploration', type=str, default='gaussian', help='Exploration type')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_cycles', type=int, default=5, help='Number of training cycles')
    parser.add_argument('--training_steps', type=int, default=15_000, help='Number of training time steps')
    parser.add_argument('--max_episode_length', type=int, default=200, help='Maximum steps per episode')
    parser.add_argument('--n_grad_steps', type=int, default=1, help='Number of gradient steps per update')
    parser.add_argument('--warm_up', type=int, default=0, help='Warm up steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=1_000_000, help='Replay buffer size')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--render', type=bool, default=False, help='Render the environment')
    parser.add_argument('--data_dir', type=str, default='data/pendulum', help='Directory to save data')
    parser.add_argument('--output_dir', type=str, default='outputs/pendulum', help='Directory to save figures')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(args.env, render_mode='human' if args.render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high


    explorator = handle_exploration(args.exploration, action_dim, action_low, action_high, args.device)
    print(f"[INFO] Exploration type: {args.exploration}")
    
    seeds = set_seeds(args.seed, args.n_cycles)
    print(f"[INFO] Cycle seeds: {seeds}")

    list_of_all_the_data = []
    best_episodic_reward = -np.inf

    for cycle_idx in range(args.n_cycles):
        print(f"[INFO] Cycle {cycle_idx+1}/{args.n_cycles}")
        torch.manual_seed(seeds[cycle_idx])
        
        behavior_policy = agent.Policy(state_dim=state_dim,action_dim=action_dim,
                                    action_lim=action_high,device=args.device)
        
        behavior_Q_fct = agent.Value(state_dim=state_dim,action_dim=action_dim,
                                    device=args.device)
        
        rl_agent = agent.DDPG(behavior_policy, behavior_Q_fct,discount_factor=args.discount,
                              seed=seeds[cycle_idx], device=args.device)
        
        memory = agent.DDPGMemory(buffer_length=args.buffer_size, state_dim=state_dim,
                                        action_dim=action_dim, device=args.device)
        
        obs, _ = env.reset(seed=int(seeds[cycle_idx]),options={'x_init': np.pi, 'y_init': 8.0})
        episodic_returns = []
        cumulative_reward = 0

        progress_bar = tqdm(range(args.training_steps), desc=f"Cycle {cycle_idx+1}/{args.n_cycles}", unit="step")
        
        for t in progress_bar:
            if t < args.warm_up:
                clipped_action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device=args.device))
                    expl_noise = explorator.sample(action.shape).cpu().numpy()
                noisy_action = action.cpu().numpy() + expl_noise
                clipped_action = np.clip(noisy_action, a_min=action_low, a_max=action_high)

            obs_, reward, terminated, truncated, _ = env.step(clipped_action)
            
            done = terminated or truncated

            cumulative_reward += reward
            memory.add_sample(state=obs, action=clipped_action, reward=reward, next_state=obs_, done=done)

            if (t >= args.warm_up) and (memory.size >= args.batch_size):
                rl_agent.train(memory_buffer=memory, batch_size=args.batch_size, epochs=args.n_grad_steps)

            if done:
                episodic_returns.append(cumulative_reward)
                if cumulative_reward > best_episodic_reward:
                    best_episodic_reward = cumulative_reward
                    torch.save(behavior_policy.state_dict(), f"{args.output_dir}/best_{args.agent}_model_{args.exploration}.pth")

                cumulative_reward = 0
                obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
            else:
                obs = obs_.copy()

        # Collect stats
        for i in range(len(rl_agent.pi_loss)):
            list_of_all_the_data.append({
                'cycle': cycle_idx + 1,
                'step': i,
                'policy_loss': rl_agent.pi_loss[i],
                'q_loss': rl_agent.q_loss[i],
                'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
            })

    env.close()

    df = pd.DataFrame(list_of_all_the_data)
    csv_filename = f"{args.agent}_{args.exploration}_metrics.csv"
    df.to_csv(os.path.join(args.data_dir, csv_filename), index=False)
    print(f"[INFO] Metrics saved to: {os.path.join(args.data_dir, csv_filename)}")

if __name__ == "__main__":
    main()

        