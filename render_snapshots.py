#!/usr/bin/env python3
"""
Render high-resolution snapshots of the trained Virgo 12-DOF MorphQuad drone
in 4 omnidirectional flight configurations:
1. 0 deg Upright Hover
2. 45 deg Tilted Flight
3. 90 deg Vertical Wall-Facing Hover
4. 180 deg Inverted (Upside-Down) Flight
"""

import os
import sys
import math
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render Virgo Flight Snapshots")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torchvision.io
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.assets import Articulation
from virgo_robot_cfg import VIRGO_CFG
from virgo_env_isaaclab import BatchedVirgoKinematics


def main():
    snapshot_dir = "/home/duc/Raptor_UAV/Virgo_RL/snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("=" * 70)
    print("      RENDERING HIGH-DEFINITION VIRGO FLIGHT SNAPSHOTS       ")
    print("=" * 70)

    # 1. Setup Simulation Context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Ground & Dome Lighting
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    dome_light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    dome_light_cfg.func("/World/DomeLight", dome_light_cfg)

    dist_light_cfg = sim_utils.DistantLightCfg(intensity=3500.0, color=(1.0, 1.0, 1.0))
    dist_light_cfg.func("/World/DistLight", dist_light_cfg)

    # Spawn Camera prim
    sim_utils.create_prim("/World/CamOrigin", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CamOrigin/CameraSensor",
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, -1.8, 1.8),
            rot=(0.88, 0.28, 0.11, 0.35),
            convention="world"
        ),
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 50.0)
        ),
    )
    cam = Camera(cfg=camera_cfg)

    # Robot Asset
    robot_cfg = VIRGO_CFG.replace(prim_path="/World/VirgoRobot")
    robot = Articulation(robot_cfg)

    # Reset Context
    sim.reset()
    robot.reset()
    cam.reset()

    kinematics = BatchedVirgoKinematics(device=sim.device)
    servo_names = [f"ARM_MOTOR_{i}" for i in range(1, 5)] + [f"SHAFT_MOTOR_{i}" for i in range(1, 5)]
    servo_ids, _ = robot.find_joints(servo_names)

    modes = [
        ("virgo_hover_0deg.png", "0 deg Upright Hover", 0.0),
        ("virgo_tilt_45deg.png", "45 deg Tilted Flight", 45.0),
        ("virgo_wall_90deg.png", "90 deg Vertical Wall Hover", 90.0),
        ("virgo_inverted_180deg.png", "180 deg Inverted Upside-Down Flight", 180.0),
    ]

    for filename, title, pitch_deg in modes:
        print(f"\n--> Posing & Rendering: {title} (Pitch = {pitch_deg} deg)...")
        pitch_rad = math.radians(pitch_deg)
        sp = math.sin(pitch_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        q = torch.tensor([[cp, 0.0, sp, 0.0]], dtype=torch.float32, device=sim.device)

        # Body hover force: f_b = R_bw @ [0, 0, mg] = [-mg*sin(theta), 0, mg*cos(theta)]
        mg = 2.45 * 9.81
        f_b = torch.tensor([[-mg * math.sin(pitch_rad), 0.0, mg * math.cos(pitch_rad)]], dtype=torch.float32, device=sim.device)
        tau_b = torch.zeros(1, 3, dtype=torch.float32, device=sim.device)

        # Null-space tilt allocation
        _, alphas, betas = kinematics.allocate(f_b, tau_b)
        joint_targets = torch.cat([alphas, betas], dim=-1)

        # Set robot pose at 1.0m altitude
        root_state = torch.zeros(1, 13, device=sim.device)
        root_state[0, 2] = 1.0
        root_state[0, 3:7] = q[0]
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_targets, torch.zeros_like(joint_targets), joint_ids=servo_ids)
        robot.set_joint_position_target(joint_targets, joint_ids=servo_ids)

        # Step 20 frames to let rendering pipeline and shaders stabilize
        for _ in range(20):
            robot.permanent_wrench_composer.set_forces_and_torques(
                forces=f_b.unsqueeze(1),
                torques=tau_b.unsqueeze(1),
                body_ids=[0]
            )
            sim.step()

        # Update and grab rendered image
        cam.update(dt=sim.get_physics_dt())
        rgb_data = cam.data.output["rgb"][0] # [H, W, 3] or [H, W, 4]
        rgb_tensor = rgb_data[..., :3].permute(2, 0, 1).cpu()
        if rgb_tensor.dtype != torch.uint8:
            rgb_tensor = rgb_tensor.to(torch.uint8)

        out_path = os.path.join(snapshot_dir, filename)
        torchvision.io.write_png(rgb_tensor, out_path)
        print(f"    [OK] Saved snapshot: {out_path} ({rgb_tensor.shape[2]}x{rgb_tensor.shape[1]})")

    print("\n[IsaacLab] All flight mode snapshots successfully rendered and saved to /home/duc/Raptor_UAV/Virgo_RL/snapshots/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except:
            pass
        os._exit(0)
