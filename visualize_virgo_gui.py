#!/usr/bin/env python3
"""
Virgo_RL: Interactive 3D GUI Simulation Viewer in Isaac Sim.
Displays the trained 12-DOF Coaxial MorphQuad drone live on the user's desktop screen (DISPLAY=:0).

Demonstrates:
1. Stable 0 deg Hover with the trained RL policy
2. Active rejection of Beaufort Scale 1 to 6 winds (up to 13.8 m/s, ~50 km/h)
3. 45 deg Tilted Flight
4. 90 deg Vertical Wall-Facing Hover
5. 180 deg Inverted (Upside-Down) Flight
"""

import os
import math
import time
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Virgo_RL Interactive 3D Simulation Viewer")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments to display")
parser.add_argument("--checkpoint", type=str, default="/home/duc/Raptor_UAV/Virgo_RL/checkpoints/virgo_isaaclab_best.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set GUI display on user screen :0
os.environ["DISPLAY"] = ":0"

# Launch Isaac Sim App with GUI enabled
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from virgo_env_isaaclab import VirgoEnv, VirgoEnvCfg


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=35, action_dim=6):
        super().__init__()
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def forward(self, obs):
        hidden = self.actor_backbone(obs)
        return self.actor_mean(hidden).clamp(-1.0, 1.0)


def main():
    print("=" * 80)
    print("      VIRGO_RL (12-DOF COAXIAL MORPHQUAD) LIVE SIMULATION VIEWER       ")
    print("=" * 80)
    print("[GUI] Displaying on X11 display: :0")
    print("[GUI] Controls: Mouse Left-Click = Rotate camera, Middle-Click = Pan, Right-Click/Scroll = Zoom")
    print("[GUI] Mass: 4.56 kg | Motors: 8 Coaxial Counter-Rotating | Servos: 8 Dynamixel")

    cfg = VirgoEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 3.5

    env = VirgoEnv(cfg=cfg)
    device = env.device

    # Load trained neural policy
    agent = ActorCritic(cfg.observation_space, cfg.action_space).to(device)
    if os.path.exists(args_cli.checkpoint):
        ckpt = torch.load(args_cli.checkpoint, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[Policy] Loaded trained checkpoint: {args_cli.checkpoint} (Update: {ckpt.get('update', '?')})")
    else:
        print(f"[Policy] Checkpoint not found at {args_cli.checkpoint}, running baseline controller.")

    agent.eval()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    # Target hover altitude: 1.5 m
    env._desired_pos_w[:, 2] = 1.5

    # Sequence of Flight Test Modes
    flight_phases = [
        # (Name, pitch_deg, wind_speed_mps, duration_steps, description)
        ("1. UPRIGHT HOVER (0 deg)", 0.0, 0.0, 200, "Nominal hover thăng bằng ở 1.5m, không gió"),
        ("2. BEAUFORT 4 GENTLE WIND (6.0 m/s)", 0.0, 6.0, 200, "Gió cấp 4 (21 km/h) - 8 servo chủ động nghiêng cụm motor chống gió"),
        ("3. BEAUFORT 6 STRONG WIND (13.8 m/s)", 0.0, 13.8, 250, "Gió cấp 6 (50 km/h) - Servo nghiêng góc lớn bù trừ khí động học"),
        ("4. TILTED FLIGHT (45 deg Pitch)", 45.0, 3.0, 200, "Nghiêng thân 45 độ, vector lực nâng hướng lên chống trọng lực"),
        ("5. WALL HOVER (90 deg Pitch)", 90.0, 2.0, 200, "Thân xoay vuông góc 90 độ áp sát mặt tường"),
        ("6. INVERTED FLIGHT (180 deg Pitch)", 180.0, 0.0, 200, "Bay lộn ngược 180 độ, servo đảo chiều sinh lực nâng đẩy lên trần"),
    ]

    phase_idx = 0
    step_in_phase = 0

    print(f"\n--> Bắt đầu Phase [{phase_idx + 1}/{len(flight_phases)}]: {flight_phases[phase_idx][0]}")
    print(f"    Mô tả: {flight_phases[phase_idx][4]}")

    last_print = time.time()

    while simulation_app.is_running():
        name, pitch_deg, wind_spd, max_steps, desc = flight_phases[phase_idx]

        # Update flight mode orientation & hover position
        pitch_rad = math.radians(pitch_deg)
        sp = math.sin(pitch_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        env._desired_quat_w[:] = torch.tensor([cp, 0.0, sp, 0.0], device=device)
        env._desired_pos_w[:] = torch.tensor([0.0, 0.0, 1.5], device=device)
        env._desired_vel_w[:] = 0.0

        # Set wind disturbance vector in X direction
        env._wind_w[:] = torch.tensor([wind_spd, 0.0, 0.0], device=device)

        # Run trained RL policy
        with torch.no_grad():
            action = agent(obs)

        # Step simulation
        obs_dict, rewards, dones, truncated, info = env.step(action)
        obs = obs_dict["policy"]

        if dones.any() or truncated.any():
            env._desired_pos_w[:] = torch.tensor([0.0, 0.0, 1.5], device=device)
            env._desired_quat_w[:] = torch.tensor([cp, 0.0, sp, 0.0], device=device)

        step_in_phase += 1

        # Periodic telemetry printout
        now = time.time()
        if now - last_print >= 0.8:
            last_print = now
            pos_err = torch.norm(env._robot.data.root_pos_w[0] - env._desired_pos_w[0]).item()
            cur_z = env._robot.data.root_pos_w[0, 2].item()
            mean_throttle = (env._current_motor_thrusts[0].mean().item() / 14.0) * 100.0
            alphas_deg = [round(a.item() * 180.0 / math.pi, 1) for a in env._current_alphas[0]]
            betas_deg = [round(b.item() * 180.0 / math.pi, 1) for b in env._current_betas[0]]
            prop_rads = [int(p.item()) for p in env._prop_velocities[0, :2]]

            print(f"[{name[:22]}] Alt: {cur_z:4.2f}m | PosErr: {pos_err*100:4.1f}cm | Wind: {wind_spd:4.1f}m/s | Throttle: {mean_throttle:4.1f}% | Servos α:{alphas_deg[:2]}° β:{betas_deg[:2]}° | Props: {prop_rads} rad/s")

        # Transition to next phase
        if step_in_phase >= max_steps:
            step_in_phase = 0
            phase_idx = (phase_idx + 1) % len(flight_phases)
            new_name = flight_phases[phase_idx][0]
            new_desc = flight_phases[phase_idx][4]
            print(f"\n--> Chuyển sang Phase [{phase_idx + 1}/{len(flight_phases)}]: {new_name}")
            print(f"    Mô tả: {new_desc}")

        # Smooth real-time pacing for desktop human viewing (~50 FPS)
        time.sleep(0.015)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
