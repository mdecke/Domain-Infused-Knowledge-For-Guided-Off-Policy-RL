import os
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from moviepy.video.io import ImageSequenceClip

def sim(args):
    """
    Loads a trained PPO model + VecNormalize stats,
    runs several episodes in the Ant environment,
    saves step-by-step data (obs, action, reward, etc.) to a CSV,
    and only records a video for the episode with the highest total reward.
    """
    model_path=f"{args.data_dir}/ppo_best_model.zip"
    vecnormalize_path=os.path.join(f"{args.data_dir}","ppo_vecnormalize.pkl")
    episodes=args.num_episodes
    max_steps=args.max_steps
    csv_path=f"{args.data_dir}/{args.env_name}_ppo_{args.num_episodes}_episodes.csv"
    video_folder=f"{args.output_dir}/videos"
    video_prefix=f"{args.env_name}_ppo_expert"

    os.makedirs(video_folder, exist_ok=True)
    def make_env():
        return gym.make(args.env_name, render_mode="rgb_array")

    vec_env = DummyVecEnv([make_env])

    if os.path.isfile(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        # IMPORTANT: set to evaluation mode
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)
    print(f"Loaded model from {model_path}")

    step_records = []
    best_reward = -np.inf
    best_frames = []
    best_episode = -1

    for ep in range(episodes):
        obs = vec_env.reset()
        done, truncated = [False], [False]  # Because it's a vec env with 1 environment
        total_reward = 0.0
        step_count = 0
        frames = []
        
        frames.append(vec_env.envs[0].render())  # single-env, so index 0

        while not (done[0] or truncated[0]) and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = vec_env.step(action)

            # Record step info
            step_records.append({
                "episode": ep,
                "states": obs.flatten().tolist(),
                "actions": action.flatten().tolist()
            })

            obs = next_obs.copy()
            total_reward += reward[0]
            step_count += 1

            frame = vec_env.envs[0].render()  # returns an RGB array
            frames.append(frame)

        print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward}")

        # Check if this episode had the best reward so far
        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames.copy()  # Store the frames for the best episode
            best_episode = ep

    # After all episodes, save only the best one as a video
    if best_episode >= 0:
        video_path = os.path.join(video_folder, f"{video_prefix}_best.mp4")
        clip = ImageSequenceClip.ImageSequenceClip(best_frames, fps=30)
        clip.write_videofile(video_path, codec="libx264")
        print(f"Saved video for best episode (#{best_episode+1}) with reward {best_reward} to {video_path}")

    df = pd.DataFrame(step_records)
    df.to_csv(csv_path, index=False)
    print(f"Saved step data to {csv_path}")

    vec_env.close()

