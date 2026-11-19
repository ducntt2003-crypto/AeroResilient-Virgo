#!/usr/bin/env python3
"""
Verification script: Spawns Virgo_RL in Isaac Lab and runs physics simulation for 100 steps.
"""

import os
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from isaaclab.app import AppLauncher

# Launch Isaac Sim headless
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from virgo_robot_cfg import VIRGO_CFG


@configclass
class VirgoSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, lights, and Virgo robot."""
    ground = sim_utils.GroundPlaneCfg()
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    robot: Articulation = VIRGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    print("[Virgo_RL] Initializing Isaac Lab InteractiveScene...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = VirgoSceneCfg(num_envs=4, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[Virgo_RL] Scene created successfully with 4 parallel environments!")
    robot: Articulation = scene["robot"]
    print(f"  - Number of joints: {robot.num_joints}")
    print(f"  - Joint names: {robot.joint_names}")
    print(f"  - Number of bodies: {robot.num_bodies}")
    print(f"  - Body names: {robot.body_names}")
    print(f"  - Root position at step 0: {robot.data.root_pos_w[0].cpu().numpy()}")

    # Step simulation
    for step in range(50):
        # Command servos to a test tilt angle (e.g. 0.3 rad)
        joint_targets = torch.zeros_like(robot.data.joint_pos)
        # 8 servo joints: tilt ARM_MOTOR to 0.2 rad, SHAFT_MOTOR to -0.3 rad
        for idx, name in enumerate(robot.joint_names):
            if "ARM_MOTOR" in name:
                joint_targets[:, idx] = 0.2
            elif "SHAFT_MOTOR" in name:
                joint_targets[:, idx] = -0.3

        robot.set_joint_position_target(joint_targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=0.01)

    print(f"[Virgo_RL] 50 simulation steps completed successfully!")
    print(f"  - Servo angles achieved on env 0: {robot.data.joint_pos[0, :8].cpu().numpy()}")
    print(f"  - Final root position: {robot.data.root_pos_w[0].cpu().numpy()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
