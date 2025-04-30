import os, math, torch, gymnasium as gym, numpy as np, matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gop_rl.modeling.mle import ActionMLE
from gop_rl.modeling.cnf import ConditionalNormalizingFlow

import os
from moviepy.video.io import ImageSequenceClip#.editor import ImageSequenceClipenv = gym.make('Walker2d-v4', render_mode='rgb_array')
# -------------------------------------------------------------------
SEED          = 42
N_EPISODES    = 100
ENV_NAME      = "Ant-v4"
INPUT_TYPE    = "prev_state_action"               # 'state' | 'state_action' | 'prev_state_action' | 'state_prev_state'
NUM_FLOWS_CNF = 20

# ----- paths -------------------------------------------------------
DATA_DIR   = "data/ant"
PPO_PATH   = f"{DATA_DIR}/ppo_best_model.zip"
VEC_PATH   = f"{DATA_DIR}/ppo_vecnormalize.pkl"
MLE_PATH   = f"{DATA_DIR}/mle/{INPUT_TYPE}_1.pth"
CNF_PATH   = f"{DATA_DIR}/cnf/{INPUT_TYPE}_1.pth"
# -------------------------------------------------------------------

# 1) sizes / env helpers -------------------------------------------
dummy = gym.make(ENV_NAME)
obs_dim, act_dim = dummy.observation_space.shape[0], dummy.action_space.shape[0]
act_low, act_high = dummy.action_space.low, dummy.action_space.high
dummy.close()

# 2) cond-dim -------------------------------------------------------
if   INPUT_TYPE == "state":             cond_dim = obs_dim
elif INPUT_TYPE == "state_action":      cond_dim = obs_dim + act_dim
elif INPUT_TYPE == "prev_state_action": cond_dim = obs_dim + act_dim + obs_dim
elif INPUT_TYPE == "state_prev_state": cond_dim = obs_dim + obs_dim
else: raise ValueError("Bad INPUT_TYPE")

# 3) load models ----------------------------------------------------
mle = ActionMLE(state_dim=cond_dim, action_dim=act_dim, action_lim=1.0)
mle.load_state_dict(torch.load(MLE_PATH, map_location="cpu", weights_only=True))
mle.eval()

cnf = ConditionalNormalizingFlow(condition_dim=cond_dim,
                                 latent_dim=act_dim,
                                 n_flows=NUM_FLOWS_CNF)
cnf.load_state_dict(torch.load(CNF_PATH, map_location="cpu", weights_only=True))
cnf.eval()

def _mk(): return gym.make(ENV_NAME)
vec = DummyVecEnv([_mk])
vec = VecNormalize.load(VEC_PATH, vec)
vec.training, vec.norm_reward = False, False
expert = PPO.load(PPO_PATH, env=vec)
obs_mean, obs_var, eps = vec.obs_rms.mean, vec.obs_rms.var, vec.epsilon
norm = lambda o: (o - obs_mean) / np.sqrt(obs_var + eps)

# 4) episode loop ---------------------------------------------------
returns_exp, returns_mle, returns_cnf = [], [], []
all_traj_exp, all_traj_mle, all_traj_cnf = [], [], []

best_ep = None
bestRc = -np.inf

for ep in range(N_EPISODES):
    # print('episode: ',ep)
    env_mle, env_cnf, env_exp= [gym.make(ENV_NAME) for _ in range(3)]
    obs_e,_ = env_exp.reset(seed=SEED+ep)
    obs_m,_ = env_mle.reset(seed=SEED+ep)
    obs_c,_ = env_cnf.reset(seed=SEED+ep)

    prev_a_m = np.zeros(act_dim, np.float32); prev_s_m = np.zeros(obs_dim, np.float32)
    prev_a_c = np.zeros(act_dim, np.float32); prev_s_c = np.zeros(obs_dim, np.float32)

    cum_e = cum_m = cum_c = 0.0
    done_e = done_m = done_c = False

    ep_traj_exp, ep_traj_mle, ep_traj_cnf = [], [], []

    while not (done_e and done_m):
        # expert ----------------------------------------------------------
        if not done_e:
            a_e,_   = expert.predict(norm(obs_e)[None,:], deterministic=True)
            a_e     = np.clip(a_e.squeeze(), act_low, act_high)
            # print('expert: ',a_e)
            obs_e, r_e, done_e, trunc_e, _ = env_exp.step(a_e)
            done_e = done_e or trunc_e
            cum_e += float(r_e)
            ep_traj_exp.append(a_e.copy())

        # mle -------------------------------------------------------------
        if not done_m:
            if INPUT_TYPE == "state":
                cond = obs_m
            elif INPUT_TYPE == "state_action":
                cond = np.concatenate([obs_m, prev_a_m])
            elif INPUT_TYPE == "prev_state_action":
                cond = np.concatenate([obs_m, prev_a_m, prev_s_m])
            elif INPUT_TYPE == "state_prev_state":
                cond = np.concatenate([obs_m, prev_s_m])
            with torch.no_grad():
                _,a,_ = mle.sample(torch.tensor(cond, dtype=torch.float32))
                a_m = np.clip(a.numpy(), -1.0, 1.0)

            # print('mle: ', a_m)
            obs_m, r_m, done_m, trunc_m, _ = env_mle.step(a_m)
            done_m = done_m or trunc_m
            cum_m += float(r_m)
            prev_a_m, prev_s_m = a_m.copy(), obs_m.copy()
            ep_traj_mle.append(a_m.copy())

        # cnf -------------------------------------------------------------
        if not done_c:
            
            if INPUT_TYPE == "state":
                cond = obs_c
            elif INPUT_TYPE == "state_action":
                cond = np.concatenate([obs_c, prev_a_c])
            elif INPUT_TYPE == "prev_state_action":
                cond = np.concatenate([obs_c, prev_a_c, prev_s_c])
            elif INPUT_TYPE == "state_prev_state":
                cond = np.concatenate([obs_c, prev_s_c])
            c = torch.tensor(cond, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # base, _ = cnf.conditional_base(c)
                # a_c, _  = cnf.inverse(base, c)
                _,a_c = cnf.sample(c)
            a_c   = np.clip(a_c.squeeze().cpu().numpy(), -0.8, 0.8)
            # print('cnf: ',a_c)
            obs_c, r_c, done_c, trunc_c, _ = env_cnf.step(a_c)
            done_c = done_c or trunc_c
            cum_c += float(r_c)
            prev_a_c, prev_s_c = a_c.copy(), obs_c.copy()
            ep_traj_cnf.append(a_c.copy())

    env_exp.close(); env_mle.close(); env_cnf.close()
    all_traj_exp.append(np.asarray(ep_traj_exp))
    all_traj_mle.append(np.asarray(ep_traj_mle))
    all_traj_cnf.append(np.asarray(ep_traj_cnf))
    returns_exp.append(cum_e); returns_mle.append(cum_m); returns_cnf.append(cum_c)
    print(f"Episode {ep+1}/{N_EPISODES}  |  R_exp={cum_e:.1f}  R_mle={cum_m:.1f}  R_cnf={cum_c:.1f}")
    if cum_c > bestRc:
        best_rc = cum_c
        best_ep = ep

# 5) averages -------------------------------------------------------
print(f"\nAverage cumulative reward over 50 episodes for {INPUT_TYPE}")
print(f"  Expert : {np.mean(returns_exp):.1f}  ± {np.std(returns_exp):.1f}")
print(f"  MLE    : {np.mean(returns_mle):.1f}  ± {np.std(returns_mle):.1f}")
print(f"  CNF    : {np.mean(returns_cnf):.1f}  ± {np.std(returns_cnf):.1f}")

traj_exp = all_traj_exp[best_ep]
traj_mle = all_traj_mle[best_ep]
traj_cnf = all_traj_cnf[best_ep]

t_e = np.arange(len(traj_exp))
t_m = np.arange(len(traj_mle))
t_c = np.arange(len(traj_cnf))

n_cols, n_rows = 2, math.ceil(act_dim/2)
plt.figure(figsize=(12, 3*n_rows))
for i in range(act_dim):
    plt.subplot(n_rows, n_cols, i+1)
    if len(t_e): plt.plot(t_e, traj_exp[:,i], label='Expert', color='blue', alpha=0.5)
    if len(t_m): plt.plot(t_m, traj_mle[:,i],'--', label='MLE', color='red', alpha=0.2)
    if len(t_c): plt.plot(t_c, traj_cnf[:,i],':', label='CNF', color='green')
    plt.title(f'Action dim {i}'); plt.grid(True)
    if i % n_cols == 0: plt.ylabel('Torque')
    if i >= act_dim - n_cols: plt.xlabel('Step')

plt.suptitle(f'Ant-v4 — best CNF episode #{best_ep+1} (Expert vs MLE vs CNF)', y=1.02)
plt.tight_layout()
plt.legend(loc='upper center', ncol=3)
os.makedirs('data/plots', exist_ok=True)
plt.savefig(f'data/plots/Ant_{INPUT_TYPE}_best_rollout.svg', dpi=300)
plt.show()