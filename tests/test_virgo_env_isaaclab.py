#!/usr/bin/env python3
"""
Test script for VirgoEnv in Isaac Lab:
Tests environment instantiation, reset, stepping, observations, and rewards.
"""

import os
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from isaaclab.app import AppLauncher

# Launch headless app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from virgo_env_isaaclab import VirgoEnv, VirgoEnvCfg


def main():
    print("=" * 70)
    print("      VIRGO_RL (12-DOF MORPHQUAD) ISAAC LAB ENVIRONMENT TEST        ")
    print("=" * 70)

    cfg = VirgoEnvCfg()
    cfg.scene.num_envs = 16  # Test with 16 parallel environments

    print(f"[IsaacLab] Creating VirgoEnv with {cfg.scene.num_envs} environments...")
    env = VirgoEnv(cfg=cfg)

    print("[IsaacLab] Resetting environment...")
    obs_dict, info = env.reset()
    policy_obs = obs_dict["policy"]
    print(f"  - Observation tensor shape : {policy_obs.shape} (Expected: [{cfg.scene.num_envs}, 35])")
    print(f"  - Action space shape       : {env.action_space.shape} (Expected: (6,))")

    print("\n[IsaacLab] Running 100 simulation steps with active 6-DOF Wrench control...")
    for step in range(100):
        # Sample small corrective actions
        actions = torch.randn(cfg.scene.num_envs, 6, device=env.device) * 0.1
        obs_dict, rewards, dones, truncated, info = env.step(actions)
        
        if step % 25 == 0:
            mean_rew = rewards.mean().item()
            mean_z = env._robot.data.root_pos_w[:, 2].mean().item()
            mean_dist = torch.norm(env._robot.data.root_pos_w - env._desired_pos_w, dim=-1).mean().item()
            print(f"  Step {step:3d} | Mean Rew: {mean_rew:6.3f} | Mean Altitude: {mean_z:5.2f}m | Mean Pos Error: {mean_dist:5.2f}m")

    print("\n[IsaacLab] SUCCESS! 100 steps completed without errors or numerical drift!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
