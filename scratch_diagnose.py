import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path.append("/home/duc/Raptor_UAV/Virgo_RL")

import torch
from virgo_env_isaaclab import VirgoEnv, VirgoEnvCfg
from run_virgo_figure8_wind_test import ActorCritic

cfg = VirgoEnvCfg()
cfg.scene.num_envs = 1
cfg.is_eval = True
env = VirgoEnv(cfg=cfg)
agent = ActorCritic(35, 6).to(env.device)
ckpt = torch.load("/home/duc/Raptor_UAV/Virgo_RL/checkpoints/virgo_isaaclab_best.pt", map_location=env.device)
agent.load_state_dict(ckpt["model_state_dict"])
agent.eval()

obs_dict, _ = env.reset()
obs = obs_dict["policy"]
print("\n" + "="*60, flush=True)
print("--- STEP 0 DIAGNOSIS ---", flush=True)
print("pos_err_b (target - pos):", obs[0, 0:3].cpu().numpy(), flush=True)
print("vel_b:", obs[0, 3:6].cpu().numpy(), flush=True)
print("omega_b:", obs[0, 6:9].cpu().numpy(), flush=True)
print("target_vel_b:", obs[0, 9:12].cpu().numpy(), flush=True)
print("q_err:", obs[0, 12:16].cpu().numpy(), flush=True)
print("target_acc_b:", obs[0, 28:31].cpu().numpy(), flush=True)
print("lookahead_err_b:", obs[0, 31:34].cpu().numpy(), flush=True)
print("phase_feat:", obs[0, 34].item(), flush=True)

with torch.no_grad():
    actions = agent(obs)
print("\nPolicy Commanded Action [fx, fy, fz, tx, ty, tz]:", actions[0].cpu().numpy(), flush=True)

for s in range(15):
    obs_dict, _, _, _, _ = env.step(actions)
    obs = obs_dict["policy"]
    with torch.no_grad():
        actions = agent(obs)
    pos = env._robot.data.root_pos_w[0].cpu().numpy()
    vel = env._robot.data.root_lin_vel_w[0].cpu().numpy()
    target_pos = env._desired_pos_w[0].cpu().numpy()
    alphas = env._current_alphas[0].cpu().numpy() * 180.0 / 3.14159
    betas = env._current_betas[0].cpu().numpy() * 180.0 / 3.14159
    print(f"Step {s+1:2d} | Pos: ({pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}) | Des: ({target_pos[0]:5.2f}, {target_pos[1]:5.2f}, {target_pos[2]:5.2f}) | Act: {actions[0, :3].cpu().numpy()} | Alpha0: {alphas[0]:4.1f}° | Beta0: {betas[0]:4.1f}°", flush=True)

simulation_app.close()
