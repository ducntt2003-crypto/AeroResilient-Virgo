#!/usr/bin/env python3
"""
================================================================================
VIRGO UAV: LIVE INTERACTIVE 3D FLIGHT IN ISAAC SIM WITH KEYBOARD & QGROUNDCONTROL
12-DOF Coaxial MorphQuad with 8 Dynamixel Servos & 8 Motors
================================================================================
Dual-Input Architecture:
1. Omniverse Window Keyboard: Click Isaac Sim 3D window and hold W/S/A/D/Q/E/C/V!
2. Terminal Keyboard: Type W/S/A/D/Space/C/H/R/1/2/3/4 in the terminal!
3. MAVLink Bridge: Open QGroundControl to view telemetry and control virtual sticks!
================================================================================
"""

import os
import sys
import time
import math
import argparse
import select
import termios
import tty
import threading
import numpy as np
import torch

# Ensure repository root and src are on sys.path
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Virgo UAV Interactive 3D Flight in Isaac Sim")
parser.add_argument("--speed", type=float, default=3.0, help="Translational fly speed in m/s (default: 3.0)")
parser.add_argument("--yaw-rate", type=float, default=1.2, help="Yaw rotation speed in rad/s (default: 1.2)")
parser.add_argument("--mavlink-port", type=int, default=14550, help="UDP port for QGroundControl connection (default: 14550)")
parser.add_argument("--checkpoint", type=str, default=os.path.join(REPO_DIR, "checkpoints", "virgo_isaaclab_best.pt"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Guarantee GUI display on DISPLAY=:0
os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from virgo_env_isaaclab import VirgoEnv, VirgoEnvCfg
import torch.nn as nn

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=35, action_dim=6):
        super().__init__()
        self.actor_backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, action_dim)

    def forward(self, obs):
        hidden = self.actor_backbone(obs)
        return self.actor_mean(hidden).clamp(-1.0, 1.0)


class TerminalKeyboardListener:
    """Non-blocking keyboard listener running in a background thread for the Terminal window."""
    def __init__(self):
        self.nudge_x = 0.0
        self.nudge_y = 0.0
        self.nudge_z = 0.0
        self.nudge_yaw = 0.0
        self.wind_mode = 1 # 1: Calm, 2: Beaufort 3, 3: Beaufort 5, 4: Beaufort 6
        self.reset_req = False
        self.hold_req = False
        self.running = True
        self.last_cmd = "Hover (Ready)"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self.running:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in ('\x1b', 'x', 'X'):
                        self.running = False
                        break
                    self._handle_key(ch)
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, ch):
        step_m = 0.75 # 0.75 meter nudge per tap in terminal
        yaw_step = 0.35 # ~20 degrees per tap

        if ch in ('w', 'W'):
            self.nudge_x += step_m
            self.last_cmd = f"Terminal: Forward (+{step_m}m X)"
        elif ch in ('s', 'S'):
            self.nudge_x -= step_m
            self.last_cmd = f"Terminal: Backward (-{step_m}m X)"
        elif ch in ('a', 'A'):
            self.nudge_y += step_m
            self.last_cmd = f"Terminal: Left (+{step_m}m Y)"
        elif ch in ('d', 'D'):
            self.nudge_y -= step_m
            self.last_cmd = f"Terminal: Right (-{step_m}m Y)"
        elif ch in (' ', 'e', 'E'):
            self.nudge_z += step_m * 0.5
            self.last_cmd = f"Terminal: Climb (+{step_m*0.5}m Z)"
        elif ch in ('c', 'C'):
            self.nudge_z -= step_m * 0.5
            self.last_cmd = f"Terminal: Descend (-{step_m*0.5}m Z)"
        elif ch in ('q', 'Q'):
            self.nudge_yaw += yaw_step
            self.last_cmd = f"Terminal: Yaw Left (+{int(math.degrees(yaw_step))}°)"
        elif ch in ('r', 'R'):
            self.nudge_yaw -= yaw_step
            self.last_cmd = f"Terminal: Yaw Right (-{int(math.degrees(yaw_step))}°)"
        elif ch in ('h', 'H'):
            self.hold_req = True
            self.last_cmd = "Terminal: Hold / Hover"
        elif ch in ('0', 'o', 'O'):
            self.reset_req = True
            self.last_cmd = "Terminal: Reset Origin"
        elif ch == '1':
            self.wind_mode = 1
            self.last_cmd = "Wind: Calm (0 m/s)"
        elif ch == '2':
            self.wind_mode = 2
            self.last_cmd = "Wind: Beaufort 3 (4.5 m/s)"
        elif ch == '3':
            self.wind_mode = 3
            self.last_cmd = "Wind: Beaufort 5 (9.0 m/s)"
        elif ch == '4':
            self.wind_mode = 4
            self.last_cmd = "Wind: Beaufort 6 Gale (14 m/s + 15N Push)"


class MavlinkQGCBridge:
    """Concurrent MAVLink telemetry and control bridge to QGroundControl."""
    def __init__(self, port=14550):
        self.port = port
        self.mav = None
        self.connected = False
        self.qgc_vx = 0.0
        self.qgc_vy = 0.0
        self.qgc_vz = 0.0
        self.qgc_yaw_rate = 0.0
        self.qgc_active = False
        self.boot_time = time.time()
        self.lat = 47.397742 * 1e7
        self.lon = 8.545594 * 1e7
        self.alt_base = 488.0

        if HAS_PYMAVLINK:
            try:
                self.mav = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=1, source_component=1)
                self.connected = True
                print(f"[MAVLink] Initialized UDP broadcast on 127.0.0.1:{port} for QGroundControl.")
            except Exception as e:
                print(f"[MAVLink] Notice: {e}")

    def update(self, pos_w, vel_w, quat_w, servo_a, servo_b):
        if not self.connected or self.mav is None:
            return

        t_ms = int((time.time() - self.boot_time) * 1000)

        # Receive incoming MAVLink packets from QGC
        while True:
            msg = self.mav.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "MANUAL_CONTROL":
                self.qgc_vx = (msg.x / 1000.0) * args_cli.speed
                self.qgc_vy = (msg.y / 1000.0) * args_cli.speed
                self.qgc_vz = ((msg.z - 500.0) / 500.0) * args_cli.speed * 0.7
                self.qgc_yaw_rate = (msg.r / 1000.0) * args_cli.yaw_rate
                self.qgc_active = True
            elif mtype == "SET_POSITION_TARGET_LOCAL_NED":
                self.qgc_vx = msg.vx if hasattr(msg, "vx") else 0.0
                self.qgc_vy = msg.vy if hasattr(msg, "vy") else 0.0
                self.qgc_vz = -msg.vz if hasattr(msg, "vz") else 0.0
                self.qgc_active = True

        # Send Periodic Telemetry to QGroundControl
        now = time.time()
        if not hasattr(self, "_last_heartbeat") or now - self._last_heartbeat >= 1.0:
            self._last_heartbeat = now
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_PX4,
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED | mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE
            )
            self.mav.mav.sys_status_send(
                0b11111111111111, 0b11111111111111, 0b11111111111111,
                150, 16800, 1200, 98, 0, 0, 0, 0, 0, 0
            )

        # Attitude (Euler angles from quat)
        qw, qx, qy, qz = quat_w[0], quat_w[1], quat_w[2], quat_w[3]
        roll = math.atan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw*qy - qz*qx))))
        yaw = math.atan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))

        self.mav.mav.attitude_send(t_ms, roll, pitch, yaw, 0.0, 0.0, 0.0)

        # Local Position NED (X=North, Y=East, Z=Down)
        self.mav.mav.local_position_ned_send(
            t_ms,
            float(pos_w[0]),
            float(pos_w[1]),
            float(-pos_w[2]),
            float(vel_w[0]),
            float(vel_w[1]),
            float(-vel_w[2])
        )

        # Global Position INT (GPS for QGC Map Display)
        d_lat = (pos_w[0] / 111139.0) * 1e7
        d_lon = (pos_w[1] / (111139.0 * math.cos(math.radians(47.397742)))) * 1e7
        self.mav.mav.global_position_int_send(
            t_ms,
            int(self.lat + d_lat),
            int(self.lon + d_lon),
            int((self.alt_base + pos_w[2]) * 1000),
            int(pos_w[2] * 1000),
            int(vel_w[0] * 100),
            int(vel_w[1] * 100),
            int(-vel_w[2] * 100),
            int((math.degrees(yaw) % 360) * 100)
        )


def main():
    print("=" * 85)
    print("   🎮 VIRGO UAV: LIVE DUAL-INPUT INTERACTIVE 3D FLIGHT IN ISAAC SIM    ")
    print("=" * 85)
    print(" 💡 YOU CAN CONTROL FROM EITHER WINDOW:")
    print("   1. [CLICK ISAAC SIM 3D WINDOW] & HOLD:")
    print("      * W / S : Fly Forward / Backward (+X / -X)")
    print("      * A / D : Fly Left / Right (+Y / -Y) [Watch 8 Servos Vector Dynamically!]")
    print("      * Q / E : Fly Up / Fly Down (+Z / -Z)")
    print("      * C / V : Yaw Turn Left / Yaw Turn Right")
    print("   2. [OR IN THIS TERMINAL WINDOW] TAP:")
    print("      * W / S : Nudge Forward / Backward (+0.75m)")
    print("      * A / D : Nudge Left / Right (+0.75m)")
    print("      * Space / C : Nudge Climb / Descend (+0.35m)")
    print("      * Q / R : Nudge Yaw Left / Right")
    print("      * H     : Hold Position / Hover")
    print("      * 0     : Reset to Origin (0, 0, 2m)")
    print("      * 1/2/3/4: Dynamic Wind (1:Calm, 2:Beaufort3, 3:Beaufort5, 4:Beaufort6)")
    print("=" * 85)
    print(f" [QGroundControl Bridge] MAVLink active on 127.0.0.1:{args_cli.mavlink_port}")
    print("=" * 85)

    cfg = VirgoEnvCfg()
    cfg.is_eval = True
    cfg.interactive_mode = True
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 6.0

    env = VirgoEnv(cfg=cfg)
    device = env.device

    # Initialize Omniverse Window Keyboard Listener (Hooks directly to Isaac Sim 3D Window)
    se3_kb_cfg = Se3KeyboardCfg(pos_sensitivity=args_cli.speed, rot_sensitivity=args_cli.yaw_rate)
    try:
        se3_kb = Se3Keyboard(se3_kb_cfg)
        has_se3_kb = True
        print("[Teleop] --> Successfully attached to Isaac Sim 3D Window Keyboard!")
    except Exception as e:
        se3_kb = None
        has_se3_kb = False
        print(f"[Teleop] Note: {e}")

    # Initialize Terminal Keyboard & QGC Bridge
    kb_term = TerminalKeyboardListener()
    mav_bridge = MavlinkQGCBridge(port=args_cli.mavlink_port)

    # Initialize policy and load trained weights
    agent = ActorCritic(cfg.observation_space, cfg.action_space).to(device)
    if os.path.exists(args_cli.checkpoint):
        ckpt = torch.load(args_cli.checkpoint, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[Policy] Successfully loaded checkpoint: {args_cli.checkpoint}")
    else:
        print(f"[Policy] Using nominal control with initialized residual policy.")
    agent.eval()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    # Prominent Visual Target Waypoint Marker (Bright Glowing Neon Crimson Sphere)
    target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetWaypoint",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.1, 0.05),
                    emissive_color=(1.0, 0.25, 0.05)
                ),
            )
        }
    )
    target_visualizer = VisualizationMarkers(target_marker_cfg)

    # Visual Wind Arrow Marker
    wind_arrow_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/WindDirectionArrow",
        markers={
            "arrow_node": sim_utils.SphereCfg(
                radius=0.065,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.85, 1.0),
                    emissive_color=(0.1, 0.65, 0.95)
                ),
            )
        }
    )
    wind_arrow_visualizer = VisualizationMarkers(wind_arrow_cfg)

    # Interactive Target Coordinates (Starts at Hover: 0.0, 0.0, 2.0 m)
    curr_target_pos = torch.tensor([0.0, 0.0, 2.0], device=device)
    curr_target_yaw = 0.0
    sim_dt = 0.02
    t_sim = 0.0
    last_hud_time = time.time()
    active_source = "Hover"

    # Initial camera view nicely framing the drone
    try:
        env.sim.set_camera_view(eye=[0.0, -5.0, 3.2], target=[0.0, 0.0, 1.8])
    except Exception:
        pass

    print("\n--> Drone is Airborne at (0, 0, 2m)! Click either window and press W/S/A/D to fly!\n")

    while simulation_app.is_running() and kb_term.running:
        # 1. Read input from Isaac Sim 3D Window (Omniverse Se3Keyboard)
        win_vx, win_vy, win_vz, win_yaw = 0.0, 0.0, 0.0, 0.0
        if has_se3_kb:
            delta_pose, _ = se3_kb.advance()
            # W/S controls x: +dx / -dx (Forward / Backward)
            # A/D controls y: +dy / -dy (Left / Right)
            # Q/E controls z: +dz / -dz (Climb / Descend)
            # C/V controls yaw: +dyaw / -dyaw (Yaw Left / Yaw Right)
            if abs(delta_pose[0]) > 0.01: win_vx = float(delta_pose[0])
            if abs(delta_pose[1]) > 0.01: win_vy = float(delta_pose[1])
            if abs(delta_pose[2]) > 0.01: win_vz = float(delta_pose[2])
            if abs(delta_pose[5]) > 0.01: win_yaw = float(delta_pose[5])

        # 2. Check Terminal Keyboard inputs
        if kb_term.reset_req:
            curr_target_pos = torch.tensor([0.0, 0.0, 2.0], device=device)
            curr_target_yaw = 0.0
            kb_term.reset_req = False
            active_source = "Reset to Origin"

        if kb_term.hold_req:
            kb_term.nudge_x = 0.0
            kb_term.nudge_y = 0.0
            kb_term.nudge_z = 0.0
            kb_term.nudge_yaw = 0.0
            kb_term.hold_req = False
            active_source = "Hold Hover"

        # Apply Terminal nudges directly to target position
        if abs(kb_term.nudge_x) > 0.001 or abs(kb_term.nudge_y) > 0.001 or abs(kb_term.nudge_z) > 0.001 or abs(kb_term.nudge_yaw) > 0.001:
            curr_target_pos[0] += kb_term.nudge_x
            curr_target_pos[1] += kb_term.nudge_y
            curr_target_pos[2] = torch.clamp(curr_target_pos[2] + kb_term.nudge_z, min=0.6, max=10.0)
            curr_target_yaw += kb_term.nudge_yaw
            active_source = kb_term.last_cmd
            # Clear nudges once consumed
            kb_term.nudge_x = 0.0
            kb_term.nudge_y = 0.0
            kb_term.nudge_z = 0.0
            kb_term.nudge_yaw = 0.0

        # Apply Window Continuous Movement
        if abs(win_vx) > 0.01 or abs(win_vy) > 0.01 or abs(win_vz) > 0.01 or abs(win_yaw) > 0.01:
            curr_target_pos[0] += win_vx * sim_dt
            curr_target_pos[1] += win_vy * sim_dt
            curr_target_pos[2] = torch.clamp(curr_target_pos[2] + win_vz * sim_dt, min=0.6, max=10.0)
            curr_target_yaw += win_yaw * sim_dt
            active_source = f"Window Key: Vx={win_vx:+.1f}, Vy={win_vy:+.1f}"

        # 3. Check QGroundControl input
        if mav_bridge.qgc_active:
            if abs(mav_bridge.qgc_vx) > 0.05 or abs(mav_bridge.qgc_vy) > 0.05 or abs(mav_bridge.qgc_vz) > 0.05:
                curr_target_pos[0] += mav_bridge.qgc_vx * sim_dt
                curr_target_pos[1] += mav_bridge.qgc_vy * sim_dt
                curr_target_pos[2] = torch.clamp(curr_target_pos[2] + mav_bridge.qgc_vz * sim_dt, min=0.6, max=10.0)
                curr_target_yaw += mav_bridge.qgc_yaw_rate * sim_dt
                active_source = "QGroundControl Stick"

        # Inject into environment desired targets
        env._desired_pos_w[0] = curr_target_pos
        env._desired_vel_w[0] = torch.tensor([win_vx, win_vy, win_vz], device=device)
        env._desired_acc_w[0] = torch.tensor([0.0, 0.0, 0.0], device=device)

        # Desired Quaternion (Roll=0.0°, Pitch=0.0°, Yaw=curr_target_yaw)
        sp = math.sin(curr_target_yaw * 0.5)
        cp = math.cos(curr_target_yaw * 0.5)
        env._desired_quat_w[0, 0] = cp
        env._desired_quat_w[0, 1] = 0.0
        env._desired_quat_w[0, 2] = 0.0
        env._desired_quat_w[0, 3] = sp
        env._psi_des[0] = curr_target_yaw
        env._desired_yaw_rate[0] = win_yaw

        # Set Environmental Wind according to user selection (1/2/3/4)
        wind_mode = kb_term.wind_mode
        if wind_mode == 1:
            w_spd = 0.0
            w_dir = torch.tensor([0.0, 0.0, 0.0], device=device)
            impulse_f = 0.0
            stage_str = "Calm (0 m/s)"
        elif wind_mode == 2:
            w_spd = 4.5
            w_dir = torch.tensor([1.0, 0.0, 0.0], device=device)
            impulse_f = 0.0
            stage_str = "Beaufort 3 (4.5 m/s)"
        elif wind_mode == 3:
            w_spd = 9.0 + 1.5 * math.sin(3.0 * t_sim)
            w_dir = torch.tensor([0.707, 0.707, 0.0], device=device)
            impulse_f = 0.0
            stage_str = "Beaufort 5 (9.0 m/s)"
        else:
            w_spd = 14.0 + 3.0 * math.sin(4.0 * t_sim)
            w_dir = torch.tensor([0.866, -0.5, 0.1], device=device)
            in_pulse = (int(t_sim * 2.0) % 4 == 0) and (t_sim % 0.5 < 0.15)
            impulse_f = 15.0 if in_pulse else 0.0
            stage_str = "Beaufort 6 (14 m/s + 15N Push!)"

        env._wind_w[0] = w_dir * w_spd
        env._ext_force_w[0] = torch.tensor([0.0, 1.0, 0.0], device=device) * impulse_f

        # Update 3D Target Marker in viewport
        target_visualizer.visualize(translations=curr_target_pos.unsqueeze(0))

        # Update 3D Wind Indicator Arrow
        w_arrow_center = curr_target_pos.clone()
        w_arrow_center[2] += 1.5
        w_arrow_len = min(1.8, 0.1 * max(0.5, w_spd))
        w_arrow_steps = torch.linspace(0.0, w_arrow_len, 5, device=device).unsqueeze(-1)
        w_eff_dir = w_dir if w_spd > 0.1 else torch.tensor([0.0, 0.0, 0.0], device=device)
        w_pts = w_arrow_center + w_arrow_steps * w_eff_dir
        wind_arrow_visualizer.visualize(translations=w_pts)

        # RL Policy Forward Pass
        with torch.no_grad():
            actions = agent(obs)

        # Environment Physics Step (PhysX 50 Hz)
        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"]

        # Telemetry extraction
        cur_pos = env._robot.data.root_pos_w[0].cpu().numpy()
        cur_vel = env._robot.data.root_lin_vel_w[0].cpu().numpy()
        cur_quat = env._robot.data.root_quat_w[0].cpu().numpy()
        servo_a = env._current_alphas[0].cpu().numpy() * (180.0 / math.pi)
        servo_b = env._current_betas[0].cpu().numpy() * (180.0 / math.pi)

        # Smooth Camera Tracking: Follow drone smoothly in 3D
        try:
            cam_eye = [float(cur_pos[0]), float(cur_pos[1]) - 4.5, float(cur_pos[2]) + 2.2]
            cam_target = [float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2])]
            env.sim.set_camera_view(eye=cam_eye, target=cam_target)
        except Exception:
            pass

        # Update MAVLink Telemetry to QGroundControl
        mav_bridge.update(cur_pos, cur_vel, cur_quat, servo_a, servo_b)

        # On-Screen HUD in Terminal (Every 0.20s)
        if time.time() - last_hud_time >= 0.20:
            last_hud_time = time.time()
            qw, qx, qy, qz = cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3]
            roll_deg = math.degrees(math.atan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy)))
            pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (qw*qy - qz*qx)))))
            pos_err = np.linalg.norm(cur_pos - curr_target_pos.cpu().numpy()) * 100.0

            qgc_badge = "[QGC LINKED]" if mav_bridge.connected else "[QGC READY]"
            pulse_badge = " [💥 PUSH!]" if impulse_f > 0.0 else ""

            print(
                f"\r[{t_sim:5.1f}s | {active_source:<22}] "
                f"Tgt:({curr_target_pos[0]:+4.1f},{curr_target_pos[1]:+4.1f},{curr_target_pos[2]:3.1f}m) | "
                f"Drone:({cur_pos[0]:+4.1f},{cur_pos[1]:+4.1f},{cur_pos[2]:3.1f}m) | "
                f"Tilt:(R:{roll_deg:+3.1f}°,P:{pitch_deg:+3.1f}°) | "
                f"Gimbal β:{np.mean(np.abs(servo_b)):4.1f}° | "
                f"{stage_str}{pulse_badge}",
                end="", flush=True
            )

        t_sim += sim_dt

    print("\n\n[Interactive Flight] Flight session ended cleanly.")
    kb_term.running = False
    simulation_app.close()


if __name__ == "__main__":
    main()
