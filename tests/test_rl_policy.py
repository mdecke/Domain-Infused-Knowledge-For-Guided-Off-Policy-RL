import gymnasium as gym
import torch
from gop_rl.agents.ddpg import Policy
import numpy as np
import os
from moviepy.video.io import ImageSequenceClip#.editor import ImageSequenceClipenv = gym.make('Walker2d-v4', render_mode='rgb_array')




file_loc = 'data/ant/best_ddpg_model_cnf.pth'

env = gym.make('Ant-v4', render_mode='rgb_array')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]



policy = Policy(state_dim=obs_dim, action_dim=act_dim, action_lim=1.0)
policy.load_state_dict(torch.load(file_loc,weights_only=True))
policy.eval()

episode_R = []
episode_frames = []

for n in range(15):
    print('episode ',n+1)
    frames = []
    cum_r = 0
    obs,_ = env.reset()
    for _ in range(1000):
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32)
            action = policy(state)
        clipped_act = np.clip(action.cpu().numpy(), -1.0, 1.0)
        obs,r,_,_,_ = env.step(clipped_act)
        cum_r += r
        frames.append(env.render())
    episode_R.append(cum_r)
    episode_frames.append(frames)
env.close()
best_episode_idx = np.argmax(episode_R)

current_dir = os.path.dirname(os.path.abspath('play.py'))
video_path = os.path.join(current_dir, f"dummy_video.mp4")
clip = ImageSequenceClip.ImageSequenceClip(episode_frames[best_episode_idx], fps=30)
clip.write_videofile(video_path, codec="libx264")
print(f"Saved video for episode")


