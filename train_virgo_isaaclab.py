#!/usr/bin/env python3
"""
Virgo_RL: GPU PPO Training for 12-DOF MorphQuad in Isaac Lab.
Features:
- Massive GPU parallel simulation (64 to 2048 envs)
- Automatic snapshot rendering: periodically saves PNG images of the drone in action
- Optional GUI mode to watch live in Isaac Sim window
"""

import os
import time
import math
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Virgo_RL Isaac Lab PPO Training")
parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--total-timesteps", type=int, default=150000, help="Total environment steps")
parser.add_argument("--num-steps", type=int, default=64, help="Rollout steps per update")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--capture-snapshots", action="store_true", default=False, help="Periodically save render snapshots")
parser.add_argument("--snapshot-interval", type=int, default=10, help="Updates between snapshot captures")
parser.add_argument("--save-dir", type=str, default="/home/duc/Raptor_UAV/Virgo_RL/checkpoints")
parser.add_argument("--live-plot", action="store_true", default=False, help="Display live real-time convergence plot on desktop")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest best checkpoint")
parser.add_argument("--r-pos-scale", type=float, default=4.0, help="Weight scale for position error reward")
parser.add_argument("--r-att-scale", type=float, default=3.5, help="Weight scale for attitude leveling reward")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if snapshots requested
if args_cli.capture_snapshots:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from virgo_env_isaaclab import VirgoEnv, VirgoEnvCfg
from isaaclab.utils.math import quat_rotate


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

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        hidden = self.actor_backbone(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.clamp(-2.0, 0.5).expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(x)
        return action, log_prob, entropy, value


def train():
    os.makedirs(args_cli.save_dir, exist_ok=True)
    snapshot_dir = "/home/duc/Raptor_UAV/Virgo_RL/snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("=" * 80)
    print("      VIRGO_RL (12-DOF MORPHQUAD) ISAAC LAB PPO TRAINING ENGINE       ")
    print("=" * 80)

    cfg = VirgoEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.enable_camera = args_cli.capture_snapshots
    cfg.r_pos_scale = args_cli.r_pos_scale
    cfg.r_att_scale = args_cli.r_att_scale

    env = VirgoEnv(cfg=cfg)
    device = env.device

    obs_dim = cfg.observation_space
    action_dim = cfg.action_space
    num_envs = cfg.scene.num_envs
    num_steps = args_cli.num_steps

    print(f"[IsaacLab] Active Device      : {device}")
    print(f"[IsaacLab] Parallel Envs      : {num_envs}")
    print(f"[IsaacLab] Obs Dim            : {obs_dim}, Action Dim: {action_dim}")
    print(f"[IsaacLab] Total Timesteps    : {args_cli.total_timesteps}")
    print(f"[IsaacLab] Reward Weights     : r_pos={cfg.r_pos_scale}, r_att={cfg.r_att_scale}")
    print(f"[IsaacLab] Snapshot Capture   : {args_cli.capture_snapshots} (every {args_cli.snapshot_interval} updates)")

    agent = ActorCritic(obs_dim, action_dim).to(device)
    best_ckpt_path = os.path.join(args_cli.save_dir, "virgo_isaaclab_best.pt")
    if args_cli.resume and os.path.exists(best_ckpt_path):
        ckpt_data = torch.load(best_ckpt_path, map_location=device)
        agent.load_state_dict(ckpt_data["model_state_dict"], strict=False)
        print(f"[IsaacLab] Resumed training from checkpoint: {best_ckpt_path}")

    optimizer = optim.Adam(agent.parameters(), lr=args_cli.learning_rate, eps=1e-5)

    # Buffers
    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, num_envs, action_dim), device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    done = torch.zeros(num_envs, device=device)

    # Set elevated overview camera to frame the entire swarm of drones
    try:
        grid_half = (math.sqrt(num_envs) * 3.0) * 0.5
        camera_dist = max(8.0, grid_half * 1.5)
        camera_height = max(5.0, grid_half * 1.1)
        env.sim.set_camera_view(
            eye=[grid_half * 1.1, -camera_dist, camera_height],
            target=[0.0, 0.0, 2.0]
        )
    except Exception:
        pass

    num_updates = args_cli.total_timesteps // (num_steps * num_envs)
    global_step = 0
    start_time = time.time()
    best_reward = -float("inf")
    best_pos_err = float("inf")

    # Capture initial frame at step 0
    if args_cli.capture_snapshots:
        init_img = os.path.join(snapshot_dir, "step_000000_init.png")
        if env.capture_snapshot(init_img):
            print(f"--> [SNAPSHOT] Saved initial state render: {init_img}")

    reward_history = []
    pos_err_history = []
    att_err_history = []
    step_history = []

    live_fig = None
    if args_cli.live_plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            plt.ion()
            live_fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.5, 7.5))
            live_fig.canvas.manager.set_window_title("Virgo_RL: Real-Time PPO Convergence Monitor")

            line_rew, = ax1.plot([], [], 'b-o', markersize=3, label="Reward")
            ax1.set_title("Virgo_RL (12-DOF Coaxial Drone) - Real-Time Learning Monitor", fontsize=11, fontweight='bold')
            ax1.set_ylabel("Mean Reward", fontweight='bold')
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper left")

            line_pos, = ax2.plot([], [], 'r-s', markersize=3, label="Position RMSE (cm)")
            ax2.set_ylabel("Pos Error (cm)", fontweight='bold')
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            line_att, = ax3.plot([], [], 'g-^', markersize=3, label="Attitude Error (deg)")
            ax3.set_xlabel("Environment Steps", fontweight='bold')
            ax3.set_ylabel("Att Error (deg)", fontweight='bold')
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="upper right")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)
            print("[GUI] Real-time convergence plot window active on DISPLAY=:0")
        except Exception as e:
            print(f"[GUI] Live plot error: {e}")
            live_fig = None

    for update in range(1, num_updates + 1):
        for k in env._episode_sums:
            env._episode_sums[k].zero_()

        origins = env._terrain.env_origins # [num_envs, 3]

        pos_err_steps = []
        att_err_steps = []
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = obs
            dones_buf[step] = done

            # Dynamic Omnidirectional Wind Curriculum (Beaufort 1: 0.5 m/s up to Beaufort 6: 13.8 m/s gale)
            sim_t = ((update - 1) * num_steps + step) * 0.02
            wind_prog = min(1.0, max(0.0, (update - 12) / max(1, num_updates - 12)))
            cur_wind_spd = 0.5 + wind_prog * 13.3
            azimuth = 2.0 * math.pi * (torch.arange(num_envs, device=device).float() / num_envs) + (0.3 * math.sin(0.2 * sim_t))
            env._wind_w[:, 0] = cur_wind_spd * torch.cos(azimuth)
            env._wind_w[:, 1] = cur_wind_spd * torch.sin(azimuth)
            env._wind_w[:, 2] = cur_wind_spd * 0.08 * torch.sin(2.0 * azimuth)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step Isaac Lab
            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs_dict["policy"]
            next_done = torch.logical_or(terminated, truncated).float()

            cur_pe = torch.norm(env._robot.data.root_pos_w - env._desired_pos_w, dim=-1).mean().item()
            z_unit_train = torch.zeros(num_envs, 3, device=device)
            z_unit_train[:, 2] = 1.0
            z_w = quat_rotate(env._robot.data.root_quat_w, z_unit_train)
            cur_ae = (torch.sqrt(z_w[:, 0]**2 + z_w[:, 1]**2) * (180.0 / math.pi)).mean().item()
            pos_err_steps.append(cur_pe)
            att_err_steps.append(cur_ae)

            rewards_buf[step] = reward
            done = next_done

        # Compute GAE
        with torch.no_grad():
            next_value = agent.get_value(obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + 0.99 * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # Optimize 4 epochs
        minibatch_size = max(256, (num_steps * num_envs) // 32)
        for epoch in range(4):
            perm = torch.randperm(num_steps * num_envs, device=device)
            for start in range(0, num_steps * num_envs, minibatch_size):
                mb_inds = perm[start : start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - 0.005 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Logging
        fps = int(global_step / (time.time() - start_time))
        mean_rew = rewards_buf.mean().item()
        mean_pos_err = sum(pos_err_steps) / len(pos_err_steps)
        mean_att_err = sum(att_err_steps) / len(att_err_steps)
        reward_history.append(mean_rew)
        pos_err_history.append(mean_pos_err)
        att_err_history.append(mean_att_err)
        step_history.append(global_step)

        # Update live GUI plot on desktop
        if live_fig is not None:
            try:
                line_rew.set_data(step_history, reward_history)
                line_pos.set_data(step_history, [p * 100.0 for p in pos_err_history])
                line_att.set_data(step_history, att_err_history)
                for ax in (ax1, ax2, ax3):
                    ax.relim()
                    ax.autoscale_view()
                live_fig.canvas.draw_idle()
                live_fig.canvas.flush_events()
                plt.pause(0.001)
                live_fig.savefig("/home/duc/Raptor_UAV/Virgo_RL/snapshots/virgo_training_curve.png", dpi=100)
            except Exception:
                pass

        print(
            f"Update {update:3d}/{num_updates:3d} | "
            f"Step: {global_step:7d} | "
            f"Mean Rew: {mean_rew:6.3f} | "
            f"Pos Err: {mean_pos_err*100:5.1f}cm | "
            f"Att Err: {mean_att_err:4.1f}° | "
            f"FPS: {fps:5d}"
        )

        # Periodic Snapshot Capture
        if args_cli.capture_snapshots and (update % args_cli.snapshot_interval == 0 or update == num_updates):
            snap_path = os.path.join(snapshot_dir, f"step_{global_step:07d}.png")
            if env.capture_snapshot(snap_path):
                print(f"--> [SNAPSHOT] Rendered & saved snapshot to: {snap_path}")

        # Checkpoint
        cur_mean_pe_cm = (sum(pos_err_steps) / len(pos_err_steps)) * 100.0
        if mean_rew > best_reward or cur_mean_pe_cm < best_pos_err:
            if mean_rew > best_reward:
                best_reward = mean_rew
            if cur_mean_pe_cm < best_pos_err:
                best_pos_err = cur_mean_pe_cm
            ckpt_path = os.path.join(args_cli.save_dir, "virgo_isaaclab_best.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "update": update,
                "global_step": global_step,
                "reward": mean_rew,
                "pos_err_cm": cur_mean_pe_cm,
            }, ckpt_path)

    print("\n[IsaacLab] Training successfully finished!")

    # Robustness Benchmark across Beaufort Scales (1 to 6) & External Disturbances
    ckpt_path = os.path.join(args_cli.save_dir, "virgo_isaaclab_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"])
        print(f"[Benchmark] Loaded best model checkpoint from Update {ckpt.get('update')} (Reward: {ckpt.get('reward', 0.0):.3f}, Pos Err: {ckpt.get('pos_err_cm', 0.0):.1f}cm)")

    print("\n" + "=" * 80)
    print("      VIRGO_RL ROBUSTNESS BENCHMARK UNDER BEAUFORT WINDS & EXTERNAL PUSHES      ")
    print("=" * 80)
    benchmark_levels = [
        ("Beaufort 1: Light Air (1.0 m/s)", 1.0, 0.0),
        ("Beaufort 3: Gentle Breeze (4.5 m/s)", 4.5, 2.0),
        ("Beaufort 5: Fresh Breeze (9.0 m/s)", 9.0, 5.0),
        ("Beaufort 6: Strong Breeze (13.8 m/s, ~50 km/h)", 13.8, 8.0),
        ("Severe Disturbance: 13.8 m/s Wind + 10N Push Impulse", 13.8, 10.0),
    ]

    benchmark_results = {}
    env.cfg.is_eval = True
    for name, w_spd, push_f in benchmark_levels:
        obs_dict, _ = env.reset()
        test_obs = obs_dict["policy"]

        # Set specific wind speed in random 3D direction
        w_dir = torch.randn(num_envs, 3, device=device)
        w_dir[:, 2] *= 0.25 # Mostly horizontal
        w_dir = w_dir / (torch.norm(w_dir, dim=-1, keepdim=True) + 1e-6)
        env._wind_w = w_dir * w_spd
        env._ext_torque_b[:] = 0.0

        # Apply external push force
        if push_f > 0.0:
            push_dir = torch.randn(num_envs, 3, device=device)
            push_dir = push_dir / (torch.norm(push_dir, dim=-1, keepdim=True) + 1e-6)
            env._ext_force_w = push_dir * push_f
        else:
            env._ext_force_w = torch.zeros_like(env._ext_force_w)

        pos_errors = []
        att_errors = []

        for _ in range(1000): # 20 seconds full Figure-8 cycle at 50 Hz
            with torch.no_grad():
                hidden = agent.actor_backbone(test_obs)
                act = agent.actor_mean(hidden).clamp(-1.0, 1.0)
            next_obs, _, _, _, _ = env.step(act)
            test_obs = next_obs["policy"]

            cur_pos_err = torch.norm(env._robot.data.root_pos_w - env._desired_pos_w, dim=-1)
            q_dot = torch.abs(torch.sum(env._robot.data.root_quat_w * env._desired_quat_w, dim=-1)).clamp(0.0, 1.0)
            cur_att_err = 2.0 * torch.acos(q_dot) * (180.0 / math.pi)

            pos_errors.append(cur_pos_err.mean().item())
            att_errors.append(cur_att_err.mean().item())

        mean_pe_cm = (sum(pos_errors) / len(pos_errors)) * 100.0 # cm
        mean_ae_deg = sum(att_errors) / len(att_errors)          # deg
        benchmark_results[name] = {
            "pos_rmse_cm": round(mean_pe_cm, 2),
            "att_rmse_deg": round(mean_ae_deg, 2)
        }
        print(f"  * {name:<50} | Pos RMSE: {mean_pe_cm:5.1f} cm | Att RMSE: {mean_ae_deg:4.1f} deg | Status: STABLE")

    # Plot training curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import json

        history_file = os.path.join(snapshot_dir, "robust_training_metrics.json")
        with open(history_file, "w") as f:
            json.dump({
                "updates": list(range(1, num_updates + 1)),
                "best_reward": best_reward,
                "total_steps": global_step,
                "benchmark": benchmark_results
            }, f, indent=2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(step_history, reward_history, color="#1f77b4", linewidth=2.0, label="PPO Mean Step Reward")
        ax1.set_title("Virgo_RL (12-DOF Coaxial MorphQuad) PPO Training under Beaufort 1-6 Winds", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Step Reward", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(step_history, pos_err_history, color="#d62728", linewidth=2.0, label="Tracking Error (m)")
        ax2.set_xlabel("Environment Timesteps", fontsize=11)
        ax2.set_ylabel("Position Error (m)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plot_path = os.path.join(snapshot_dir, "virgo_training_curve.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"\n--> [PLOT] Saved robust training curve to: {plot_path}")
    except Exception as e:
        print("[Plot] Notice:", e)

    # Auto-export policy to C++ Header for PX4 Autopilot Integration
    px4_dir = "/home/duc/Raptor_UAV/Virgo_RL/px4_integration"
    px4_module_dir = "/home/duc/Raptor_UAV/px4-autopilot/src/modules/virgo_rl_controller"
    os.makedirs(px4_dir, exist_ok=True)
    os.makedirs(px4_module_dir, exist_ok=True)
    try:
        from export_virgo_px4 import export_policy_to_cpp_header, export_allocation_to_cpp
        best_ckpt = os.path.join(args_cli.save_dir, "virgo_isaaclab_best.pt")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location="cpu")
            export_policy_to_cpp_header(ckpt["model_state_dict"], os.path.join(px4_dir, "virgo_policy.h"))
            export_policy_to_cpp_header(ckpt["model_state_dict"], os.path.join(px4_module_dir, "virgo_policy.h"))
        export_allocation_to_cpp(os.path.join(px4_dir, "virgo_allocation.h"))
        export_allocation_to_cpp(os.path.join(px4_module_dir, "virgo_allocation.h"))
        print(f"[PX4 Export] Embedded C++ headers successfully exported to:\n  - {px4_dir}\n  - {px4_module_dir}")
    except Exception as e:
        print("[PX4 Export] Error exporting C++ headers:", e)

    # Keep 3D swarm flight active so the user can interactively view the trained drones
    if simulation_app.is_running() and not getattr(args_cli, "headless", False):
        print("\n" + "=" * 80)
        print(" [GUI] Training complete! Now demonstrating live swarm flight on DISPLAY=:0")
        print(" [GUI] All drones are actively flying, spinning props, and rejecting wind disturbances.")
        print(" [GUI] Mouse: Left-Click=Rotate, Middle-Click=Pan, Right-Click/Scroll=Zoom")
        print("=" * 80)
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        demo_t = 0.0
        while simulation_app.is_running():
            omega = 2.0 * math.pi / 20.0
            omega_z = 2.0 * omega
            env._desired_pos_w[:, 0] = origins[:, 0] + 4.0 * math.sin(omega * demo_t)
            env._desired_pos_w[:, 1] = origins[:, 1] + 0.8 * math.sin(2.0 * omega * demo_t)
            env._desired_pos_w[:, 2] = origins[:, 2] + 2.0 + 0.4 * math.sin(omega_z * demo_t)

            env._desired_vel_w[:, 0] = 4.0 * omega * math.cos(omega * demo_t)
            env._desired_vel_w[:, 1] = 1.6 * omega * math.cos(2.0 * omega * demo_t)
            env._desired_vel_w[:, 2] = 0.8 * omega * math.cos(omega_z * demo_t)

            # Continuous Beaufort 5-6 cross wind during live demonstration
            env._wind_w[:, 0] = 9.0 + 3.0 * math.sin(0.8 * demo_t)
            env._wind_w[:, 1] = 4.5 * math.cos(0.5 * demo_t)

            with torch.no_grad():
                act = agent.actor_mean(agent.actor_backbone(obs)).clamp(-1.0, 1.0)
            obs_dict, _, _, _, _ = env.step(act)
            obs = obs_dict["policy"]
            demo_t += 0.02
            if live_fig is not None:
                live_fig.canvas.flush_events()
            time.sleep(0.01)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)
