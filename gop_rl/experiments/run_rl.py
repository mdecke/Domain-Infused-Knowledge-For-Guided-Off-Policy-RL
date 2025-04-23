import os
import argparse
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from gymnasium.vector import SyncVectorEnv

from gop_rl.utils import set_seeds
from gop_rl.utils.io import handle_exploration, handle_input
from gop_rl.agents import ddpg as agent

def make_env(env_name: str, seed: int):
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env
    return _init

def evaluate_policy(env_name:str, policy: agent.Policy, device: torch.device, n_episodes:int=100, seed:int=0) -> float:
    """Run exactly one episode per sub-env (no exploration noise)."""
    policy.eval()
    env = gym.make(env_name)
    act_lim_high = env.action_space.high
    act_lim_low = env.action_space.low
    episode_rewards = []
    for n in range(n_episodes):
        obs, _ = env.reset(seed=seed+n)
        done = False
        cumulative_reward = 0
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                action = policy(obs_tensor).cpu().numpy()
            clipped_action = np.clip(action, a_min=act_lim_low, a_max=act_lim_high)

            obs_,r,trunc ,term, _ = env.step(clipped_action)
            cumulative_reward += r
            done = trunc or term
            obs = obs_.copy()
            if done:
                episode_rewards.append(cumulative_reward)
    env.close()
    policy.train()
    return float(np.array(episode_rewards).mean())


def main():
    parser = argparse.ArgumentParser(description='Run RL agent on parallel environments')
    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='Environment name')
    parser.add_argument('--agent', type=str, default='ddpg', help='Agent name')
    parser.add_argument('--exploration', type=str, default='gaussian', help='Exploration type')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_cycles', type=int, default=5, help='Number of training cycles')
    parser.add_argument('--training_steps', type=int, default=15000, help='Number of training time steps')
    parser.add_argument('--max_episode_length', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--n_grad_steps', type=int, default=1, help='Gradient steps per update')
    parser.add_argument('--warm_up', type=int, default=0, help='Warm up steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=1_000_000, help='Replay buffer size')
    parser.add_argument('--eval_freq', type=int, default= 1500, help='frequncy of evaluation usually 0.1*train steps')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--data_dir', type=str, default='data/pendulum', help='Directory to save data')
    parser.add_argument('--output_dir', type=str, default='outputs/pendulum', help='Directory to save models')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    # Determine dimensions from a dummy env
    dummy = gym.make(args.env_name)
    state_dim  = dummy.observation_space.shape[0]
    action_dim = dummy.action_space.shape[0]
    action_low = dummy.action_space.low
    action_high= dummy.action_space.high
    dummy.close()

    explorator, noise = handle_exploration(
        args.exploration, action_dim, action_low, action_high, args.device
    )
    print(f"[INFO] Exploration type: {args.exploration}")

    seeds = set_seeds(args.seed, 2*args.n_cycles)
    train_seeds, test_seeds = np.split(seeds, 2)
    print(f"[INFO] Cycle train and eval seeds: {seeds}")

    loss_records   = []
    return_records = []
    eval_records   = []
    BEST_SO_FAR    = -np.inf

    for cycle_idx in range(args.n_cycles):
        print(f"[INFO] Cycle {cycle_idx+1}/{args.n_cycles}")
        seed = int(train_seeds[cycle_idx])
        torch.manual_seed(seed)
        np.random.seed(seed)

        # initialize policy and Q-function
        behavior_policy = agent.Policy(state_dim=state_dim, action_dim=action_dim,action_lim=action_high, device=device)
        behavior_Q_fct = agent.Value(state_dim=state_dim, action_dim=action_dim,device=device)
        
        agent.init_model_weights(behavior_policy, low=0.0, high=0.001, seed=seed)
        agent.init_model_weights(behavior_Q_fct,  low=0.0, high=0.001, seed=seed)

        rl_agent = agent.DDPG(behavior_policy, behavior_Q_fct,
                              discount_factor=args.discount,
                              seed=seed, device=device)
        
        memory = agent.DDPGMemory(buffer_length=args.buffer_size, state_dim=state_dim,
                                  action_dim=action_dim, device=device)

        # create vectorized envs for this cycle
        train_envs = SyncVectorEnv([make_env(args.env_name, seed = seed + i) for i in range(args.num_envs)])
        
        obs, _ = train_envs.reset()
        cumulative_reward = np.zeros(args.num_envs, dtype=np.float32)
        episode_counter = np.zeros(args.num_envs, dtype=int)
        prev_action = np.zeros((args.num_envs, action_dim))
        prev_state = np.zeros_like(obs)

        progress_bar = tqdm(range(args.training_steps),desc=f"Cycle {cycle_idx+1}/{args.n_cycles}",unit="step")

        for t in progress_bar:
            # action selection
            if memory.size < args.warm_up:
                clipped_action = np.stack([train_envs.single_action_space.sample() for _ in range(args.num_envs)])
            else:
                with torch.no_grad():
                    state = torch.tensor(obs, dtype=torch.float32, device=device)
                    action = behavior_policy(state)
                    if noise:
                        exploration = explorator.sample(action.shape).to(device)
                    else:
                        # explorator_input = handle_input(args.input_type)
                        exploration = explorator.sample(state).to(device)
                    raw_action = action + exploration
                clipped_action = np.clip(raw_action.cpu().numpy(),a_min=action_low,a_max=action_high)
            
            # step envs
            next_obs, rewards, terminated, truncated, _ = train_envs.step(clipped_action)
            dones = np.logical_or(terminated, truncated)
            cumulative_reward += rewards
            # store transitions
            for i in range(args.num_envs):
                memory.add_sample(state=obs[i,:],
                                  action=clipped_action[i,:],
                                  reward=rewards[i],
                                  next_state=next_obs[i,:],
                                  done=dones[i].astype(np.float32))

            # training update
            if memory.size >= args.warm_up and memory.size >= args.batch_size:
                rl_agent.train(memory_buffer=memory,
                               batch_size=args.batch_size,
                               epochs=args.n_grad_steps)
                loss_records.append({'cycle': cycle_idx + 1,
                                     'step': t,
                                     'policy_loss': rl_agent.pi_loss[-1],
                                     'q_loss': rl_agent.q_loss[-1]})
            
            if t > 0 and t % args.eval_freq == 0:
                avg_r = evaluate_policy(env_name=args.env_name, policy=rl_agent.pi, device=device, seed=int(test_seeds[cycle_idx]))
                eval_records.append({'cycle': cycle_idx+1,
                                     'step': t,
                                     'avg_return': avg_r})
                
                print(f"[EVAL] cycle {cycle_idx+1}, step {t} → avg_return {avg_r:.2f}")
            
            prev_state = obs.copy()
            prev_action = clipped_action.copy()

            for i, done_flag in enumerate(dones):
                if done_flag:
                    episode_counter[i] +=1
                    if cumulative_reward[i] > BEST_SO_FAR:
                        BEST_SO_FAR = cumulative_reward[i]
                        torch.save(
                            behavior_policy.state_dict(),
                            f"{args.data_dir}/best_{args.agent}_model_{args.exploration}.pth"
                        )
                    return_records.append({'cycle': cycle_idx+1, 'env_idx': i,
                                           'episode': episode_counter[i],'return':  cumulative_reward[i]})
                    cumulative_reward[i] = 0.0
                    reset_obs, _ = train_envs.reset()
                    next_obs[i] = reset_obs[i]
                    prev_action[i] = np.zeros_like((action_dim,))
                    prev_state[i] = np.zeros_like((state_dim,))

            obs = next_obs.copy()

        train_envs.close()

        
    # save metrics
    df_loss   = pd.DataFrame(loss_records)
    df_ret    = pd.DataFrame(return_records)
    df_eval   = pd.DataFrame(eval_records)

    df_loss.to_csv(os.path.join(args.data_dir, f"losses_{args.exploration}.csv"),     index=False)
    df_ret .to_csv(os.path.join(args.data_dir, f"returns_{args.exploration}.csv"),    index=False)
    df_eval.to_csv(os.path.join(args.data_dir, f"eval_returns__{args.exploration}.csv"), index=False)
    print(f"[INFO] Metrics saved to {args.data_dir}")

if __name__ == "__main__":
    main()
