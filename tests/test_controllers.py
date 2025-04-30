import os
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from gop_rl.modeling.cnf import ConditionalNormalizingFlow
from gop_rl.modeling.mle import ActionMLE
from gop_rl.expert_controllers.elqr import LQRController, EnergyShapingController

SEED = 42

INPUT_TYPE = 'state_prev_state'  # 'state', 'state_action', or 'prev_state_action' or 'state_prev_state'
ANGLE_SWITCH_THRESHOLD_DEG = 18
device = 'cpu'


if __name__ == '__main__':
    np.random.seed(SEED)
    
    base_env = gym.make("Pendulum-v1")
    action_low, action_high = base_env.action_space.low, base_env.action_space.high
    action_dim = base_env.action_space.shape[0]
    pendulum_params = dict(
        mass        = base_env.unwrapped.m,
        rod_length  = base_env.unwrapped.l,
        gravity     = 10.0,
        action_limits = (action_low, action_high),
        dt          = base_env.unwrapped.dt,
    )
    base_env.close()

    NUM_EPISODES     = 1000
    PLOT_EPISODE_IDX = np.random.randint(0, 100)
    print('plot episode: ', PLOT_EPISODE_IDX)

    
    if INPUT_TYPE == 'state':
        input_dim = 2
        mle_name = 'P(a|s)'
        cnf_name = 'CNF(a|s)'
    elif INPUT_TYPE == 'state_action':
        input_dim = 3
        mle_name = 'P(a|s,a-1)'
        cnf_name = 'CNF(a|s,a-1)'
    elif INPUT_TYPE == 'prev_state_action':
        input_dim = 5
        mle_name = 'P(a|s,a-1,s-1)'
        cnf_name = 'CNF(a|s,a-1,s-1)'
    elif INPUT_TYPE == 'state_prev_state':
        input_dim = 4
        mle_name = 'P(a|s,a-1,s-1)'
        cnf_name = 'CNF(a|s,a-1,s-1)'
    else:
        raise ValueError(f"Invalid input type: {INPUT_TYPE}")

    energy_controller = EnergyShapingController(**pendulum_params)
    lqr_controller = LQRController(**pendulum_params)
    
   
    MLE_PATH  = f"data/pendulum/mle/{INPUT_TYPE}_1.pth"
    CNF_PATH  = f"data/pendulum/cnf/{INPUT_TYPE}_1.pth"   # adjust to your folder

    mle_sampler = ActionMLE(state_dim=input_dim, action_dim=action_dim,
                            action_lim=action_high)
    mle_sampler.load_state_dict(torch.load(MLE_PATH, map_location=device,
                                        weights_only=True))
    mle_sampler.eval()

    cnf_model = ConditionalNormalizingFlow(condition_dim=input_dim,
                                           n_flows=6,latent_dim=action_dim)          # ← use your config
    cnf_model.load_state_dict(torch.load(CNF_PATH, map_location=device,
                                        weights_only=True))
    cnf_model.eval()

    
    ret_expert, ret_mle, ret_cnf = [], [], []
    traj_expert, traj_mle, traj_cnf = {}, {}, {}

    mse_mle_episodes = []
    mse_cnf_episodes = []

    # helper to convert states to (θ, θ̇)
    def theta_theta_dot(state_vec):
        cos_t, sin_t, thdot = state_vec
        return np.arctan2(sin_t, cos_t), thdot

    
    for ep in range(NUM_EPISODES):
        print('episode: ', ep)
        env_exp = gym.make("Pendulum-v1")
        env_mle = gym.make("Pendulum-v1")
        env_cnf = gym.make("Pendulum-v1")

        init_opts = {"x_init": np.pi, "y_init": 8.0}
        obs_exp, _ = env_exp.reset(seed=SEED + ep, options=init_opts)
        obs_mle, _ = env_mle.reset(seed=SEED + ep, options=init_opts)
        obs_cnf, _ = env_cnf.reset(seed=SEED + ep, options=init_opts)

        assert np.allclose(obs_exp, obs_mle) and np.allclose(obs_exp, obs_cnf)

        # per-episode logs
        ep_ret_e = ep_ret_m = ep_ret_c = 0.0
        acts_e,  states_e  = [], []
        acts_m,  states_m  = [], []
        acts_c,  states_c  = [], []

        prev_a = np.zeros(action_dim, dtype=np.float32)
        prev_s = np.zeros(2,           dtype=np.float32)

        done_e = done_m = done_c = False
        while not (done_e and done_m and done_c):

            # ========================= Expert ===============================
            if not done_e:
                θ = np.arctan2(obs_exp[1], obs_exp[0])
                θ̇ = obs_exp[2]
                pv = np.array([θ, θ̇])
                act_e = (lqr_controller.compute_control(pv)
                        if abs(θ) < np.deg2rad(ANGLE_SWITCH_THRESHOLD_DEG)
                        else energy_controller.get_action(pv))
                nxt, r, term, trunc, _ = env_exp.step(act_e)
                done_e = term or trunc
                ep_ret_e += float(r)
                acts_e.append(float(act_e.item()))
                states_e.append(obs_exp.copy())
                obs_exp = nxt.copy()

            # ========================= MLE ==================================
            if not done_m:
                theta_mle = np.arctan2(obs_mle[1], obs_mle[0])
                theta_dot_mle= obs_mle[2]
                pv = np.array([theta_mle, theta_dot_mle])

                if   INPUT_TYPE == "state":                cond = pv
                elif INPUT_TYPE == "state_action":         cond = np.concatenate([pv, prev_a])
                elif INPUT_TYPE == "prev_state_action":    cond = np.concatenate([pv, prev_a, prev_s])
                elif INPUT_TYPE == "state_prev_state":     cond = np.concatenate([pv, prev_s])

                with torch.no_grad():
                    _, mu, _ = mle_sampler.sample(torch.tensor(cond, dtype=torch.float32))
                act_m = np.clip(mu.numpy(), action_low, action_high)

                nxt, r, term, trunc, _ = env_mle.step(act_m)
                done_m = term or trunc
                ep_ret_m += float(r)
                acts_m.append(float(act_m.item()))
                states_m.append(obs_mle.copy())

                prev_a, prev_s = act_m.copy(), pv.copy()
                obs_mle = nxt.copy()

            # ========================= CNF ==================================
            if not done_c:
                theta_cnf = np.arctan2(obs_cnf[1], obs_cnf[0])
                theta_dot_cnf= obs_cnf[2]
                pv = np.array([theta_cnf, theta_dot_cnf])

                if   INPUT_TYPE == "state":                cond = pv
                elif INPUT_TYPE == "state_action":         cond = np.concatenate([pv, prev_a])
                elif INPUT_TYPE == "prev_state_action":    cond = np.concatenate([pv, prev_a, prev_s])
                elif INPUT_TYPE == "state_prev_state":     cond = np.concatenate([pv, prev_s])

                cond_t = torch.tensor(cond, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    base_mean, _ = cnf_model.conditional_base(cond_t)
                    expert_action, _ = cnf_model.inverse(base_mean, cond_t)
                    predicted_action = expert_action.cpu().numpy().flatten()
                    # _,action = cnf_model.sample(cond_t)
                act_c = np.clip(predicted_action, action_low, action_high)

                nxt, r, term, trunc, _ = env_cnf.step(act_c)
                done_c = term or trunc
                ep_ret_c += float(r)
                acts_c.append(float(act_c.item()))
                states_c.append(obs_cnf.copy())

                prev_a, prev_s = act_c.copy(), pv.copy()
                obs_cnf = nxt.copy()

        # close envs
        env_exp.close(); env_mle.close(); env_cnf.close()
        print('deon episode: ', ep)

        mse_mle = np.mean((np.array(acts_m) - np.array(acts_e))**2)
        mse_cnf = np.mean((np.array(acts_c) - np.array(acts_e))**2)
        mse_mle_episodes.append(mse_mle)
        mse_cnf_episodes.append(mse_cnf)

        # aggregate
        ret_expert.append(ep_ret_e); ret_mle.append(ep_ret_m); ret_cnf.append(ep_ret_c)

        if ep == PLOT_EPISODE_IDX:
            traj_expert = {"states": np.array(states_e), "actions": np.array(acts_e)}
            traj_mle    = {"states": np.array(states_m), "actions": np.array(acts_m)}
            traj_cnf    = {"states": np.array(states_c), "actions": np.array(acts_c)}

   
    s1_exprt, s2_exprt = zip(*(theta_theta_dot(s) for s in traj_expert["states"]))
    s1_mle, s2_mle = zip(*(theta_theta_dot(s) for s in traj_mle["states"]))
    s1_cnf, s2_cnf = zip(*(theta_theta_dot(s) for s in traj_cnf["states"]))

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(s1_exprt, label="Expert", color='blue',alpha=0.5) 
    plt.plot(s1_mle,"--",label="MLE", color='red')
    plt.plot(s1_cnf,"-.",label="CNF", color='green')
    plt.ylabel("θ [rad]")
    plt.legend()
    plt.grid()
    ax = plt.gca()  # Get the current axes
    ax.set_facecolor('0.95')

    plt.subplot(3,1,2)
    plt.plot(s2_exprt, color='blue',alpha=0.5)
    plt.plot(s2_mle,"--", color='red') 
    plt.plot(s2_cnf,"-.", color='green')
    plt.ylabel("dθ/dt")
    plt.grid()
    ax = plt.gca()  # Get the current axes
    ax.set_facecolor('0.95')

    plt.subplot(3,1,3)
    plt.plot(traj_expert["actions"], color='blue',alpha=0.5)
    plt.plot(traj_mle["actions"],"--",color='red')
    plt.plot(traj_cnf["actions"],":", color='green')
    plt.ylabel("Torque")
    plt.xlabel("timestep")
    plt.grid()
    ax = plt.gca()  # Get the current axes
    ax.set_facecolor('0.95')

    plt.tight_layout()
    plt.savefig(f"outputs/pendulum/{INPUT_TYPE}_rollout_comp.svg")
    plt.show()

  
    print(f"Expert  avg return: {np.mean(ret_expert):.2f} ± {np.std(ret_expert):.2f}")
    print(f"MLE     avg return: {np.mean(ret_mle):.2f} ± {np.std(ret_mle):.2f}")
    print(f"CNF     avg return: {np.mean(ret_cnf):.2f} ± {np.std(ret_cnf):.2f}")
    print(f"MLE    action MSE: {np.mean(mse_mle_episodes):.4f} ± {np.std(mse_mle_episodes):.4f}")
    print(f"CNF    action MSE: {np.mean(mse_cnf_episodes):.4f} ± {np.std(mse_cnf_episodes):.4f}")
