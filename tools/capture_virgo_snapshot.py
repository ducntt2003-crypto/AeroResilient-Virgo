#!/usr/bin/env python3
"""
Captures a high-resolution snapshot of Virgo_RL inside Isaac Sim / Isaac Lab.
"""

from isaaclab.app import AppLauncher

# Launch with rendering enabled (even in headless mode)
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

import torch
import torchvision.io
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from virgo_robot_cfg import VIRGO_CFG


@configclass
class VirgoSnapshotSceneCfg(InteractiveSceneCfg):
    ground = sim_utils.GroundPlaneCfg()
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    robot: Articulation = VIRGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Camera looking at the robot from 45 deg angle
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.2, -1.2, 2.0),
            rot=(0.88, 0.28, 0.11, 0.35), # looking down towards (0, 0, 1.5)
            convention="world"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 50.0)
        ),
        width=800,
        height=600,
    )


def main():
    save_dir = "/home/duc/Raptor_UAV/Virgo_RL/snapshots"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "virgo_isaaclab_render.png")

    print("[Virgo_RL] Initializing Isaac Lab Scene with TiledCamera...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = VirgoSnapshotSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    camera: TiledCamera = scene["camera"]

    # Step simulation 15 steps so materials and lighting converge
    for step in range(15):
        # Tilt servos slightly to showcase morphable mechanism
        joint_targets = torch.zeros_like(robot.data.joint_pos)
        for idx, name in enumerate(robot.joint_names):
            if "ARM_MOTOR" in name:
                joint_targets[:, idx] = 0.35
            elif "SHAFT_MOTOR" in name:
                joint_targets[:, idx] = -0.45

        robot.set_joint_position_target(joint_targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=0.01)

    # Capture frame from camera
    camera.update(dt=0.01)
    rgb_data = camera.data.output["rgb"][0] # [H, W, 3] or [H, W, 4]
    
    # Ensure RGB uint8 format [3, H, W] for torchvision
    if rgb_data.shape[-1] >= 3:
        rgb_tensor = rgb_data[..., :3].permute(2, 0, 1).cpu()
        if rgb_tensor.dtype != torch.uint8:
            rgb_tensor = rgb_tensor.to(torch.uint8)
        torchvision.io.write_png(rgb_tensor, out_path)
        print(f"\n[Virgo_RL] Snapshot successfully rendered and saved to: {out_path}")
    else:
        print("[Virgo_RL] Unexpected camera output shape:", rgb_data.shape)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
