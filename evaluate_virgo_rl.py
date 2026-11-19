#!/usr/bin/env python3
"""
Virgo_RL: Evaluation & Multi-mode Benchmark Suite for 12-DOF MorphQuad
Tests 4 Omnidirectional Flight Modes:
1. Upright Hover (Roll 0 deg, Pitch 0 deg)
2. Tilted Hover (Pitch 45 deg)
3. Wall-Facing Inspection Hover (Pitch 90 deg)
4. Full Inverted Upside-Down Hover (Pitch 180 deg)

Generates 4-panel engineering plot of 6-DOF tracking, thrusts, and servo angles.
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless plotting
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from virgo_env import VirgoEnv, euler_to_quat, quat_to_rot_matrix
from train_virgo_rl import ActorCritic


def quat_to_euler(q):
    """
    Convert quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw] in radians
    """
    w, x, y, z = q
    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def run_benchmark_mode(env, agent, device, target_pos, target_rpy, steps=300):
    """
    Runs a single flight benchmark maneuver and records state telemetry
    """
    target_q = euler_to_quat(target_rpy[0], target_rpy[1], target_rpy[2])
    obs, _ = env.reset(options={'target_pos': target_pos, 'target_quat': target_q})

    history = {
        'time': [],
        'pos': [],
        'target_pos': [],
        'euler': [],
        'target_euler': [],
        'thrusts': [],
        'alphas': [],
        'betas': []
    }

    t = 0.0
    for step in range(steps):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            hidden = agent.actor_backbone(obs_t)
            action = agent.actor_mean(hidden).cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)

        obs, rew, term, trunc, info = env.step(action)
        t += env.dt

        roll, pitch, yaw = quat_to_euler(env.quat)
        history['time'].append(t)
        history['pos'].append(env.pos.copy())
        history['target_pos'].append(env.target_pos.copy())
        history['euler'].append([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])
        history['target_euler'].append([np.degrees(target_rpy[0]), np.degrees(target_rpy[1]), np.degrees(target_rpy[2])])
        history['thrusts'].append(env.thrusts.copy())
        history['alphas'].append(np.degrees(env.alphas.copy()))
        history['betas'].append(np.degrees(env.betas.copy()))

        if term:
            break

    for k in history:
        history[k] = np.array(history[k])

    return history


def evaluate_and_plot(ckpt_path, output_png="/home/duc/Raptor_UAV/Virgo_RL/evaluation_results.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Virgo_RL] Loading checkpoint from: {ckpt_path} on {device}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    action_mode = checkpoint['action_mode']

    agent = ActorCritic(obs_dim, action_dim).to(device)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()

    env = VirgoEnv(action_mode=action_mode, randomize_targets=False, randomize_dynamics=False)

    benchmarks = [
        ("Upright Hover (0 deg)", [0.0, 0.0, 1.5], [0.0, 0.0, 0.0]),
        ("Tilted Hover (45 deg Pitch)", [0.0, 0.0, 1.5], [0.0, np.radians(45.0), 0.0]),
        ("Wall Hover (90 deg Pitch)", [0.0, 0.0, 1.5], [0.0, np.radians(90.0), 0.0]),
        ("Inverted Hover (180 deg Upside-down)", [0.0, 0.0, 1.5], [0.0, np.radians(180.0), 0.0]),
    ]

    fig, axs = plt.subplots(4, 3, figsize=(18, 14), constrained_layout=True)
    fig.suptitle("Virgo_RL (12-DOF MorphQuad) RL Omnidirectional Benchmark", fontsize=16, fontweight='bold')

    for row_idx, (name, t_pos, t_rpy) in enumerate(benchmarks):
        print(f"\nEvaluating: {name} ...")
        hist = run_benchmark_mode(env, agent, device, t_pos, t_rpy, steps=300)
        t = hist['time']

        final_pos_err = np.linalg.norm(hist['pos'][-1] - hist['target_pos'][-1])
        final_pitch = hist['euler'][-1, 1]
        target_pitch = hist['target_euler'][-1, 1]
        print(f"  -> Final Position Error: {final_pos_err*100:.2f} cm")
        print(f"  -> Pitch: {final_pitch:.1f} deg (Target: {target_pitch:.1f} deg)")

        # Column 1: Position Tracking (X, Y, Z)
        ax = axs[row_idx, 0]
        ax.plot(t, hist['pos'][:, 0], 'r-', label='X')
        ax.plot(t, hist['pos'][:, 1], 'g-', label='Y')
        ax.plot(t, hist['pos'][:, 2], 'b-', label='Z')
        ax.plot(t, hist['target_pos'][:, 2], 'b--', alpha=0.5, label='Z target')
        ax.set_title(f"{name}: Position Tracking")
        ax.set_ylabel("Position (m)")
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.legend(loc='upper right')

        # Column 2: Attitude Tracking (Pitch & Roll)
        ax = axs[row_idx, 1]
        ax.plot(t, hist['euler'][:, 0], 'r-', label='Roll')
        ax.plot(t, hist['euler'][:, 1], 'g-', label='Pitch')
        ax.plot(t, hist['target_euler'][:, 1], 'k--', alpha=0.6, label='Target Pitch')
        ax.set_title(f"{name}: Attitude Tracking")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.legend(loc='upper right')

        # Column 3: Actuator Allocations (Thrusts & Betas)
        ax = axs[row_idx, 2]
        for i in range(4):
            ax.plot(t, hist['thrusts'][:, i], label=f'T{i+1}')
        ax.set_title(f"{name}: Rotor Thrusts (4 Arms)")
        ax.set_ylabel("Thrust (N)")
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.legend(loc='upper right')

    plt.savefig(output_png, dpi=200)
    print(f"\n[Virgo_RL] Benchmark plot successfully saved to: {output_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/duc/Raptor_UAV/Virgo_RL/checkpoints/virgo_ppo_best.pt")
    parser.add_argument("--output", type=str, default="/home/duc/Raptor_UAV/Virgo_RL/evaluation_results.png")
    args = parser.parse_args()

    evaluate_and_plot(args.ckpt, args.output)
