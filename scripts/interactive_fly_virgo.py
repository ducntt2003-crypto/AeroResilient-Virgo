#!/usr/bin/env python3
"""
================================================================================
VIRGO UAV: LIVE INTERACTIVE 3D FLIGHT IN ISAAC SIM WITH KEYBOARD & QGROUNDCONTROL
12-DOF Coaxial MorphQuad with 8 Dynamixel Servos & 8 Motors
================================================================================
Features:
1. Real-time 3D Viewport in Isaac Sim (DISPLAY=:0) with auto-tracking camera.
2. Direct Keyboard Control (W/A/S/D/Space/C/Q/E/H/R/1/2/3/4).
3. Concurrent MAVLink Bridge to QGroundControl on UDP 14550.
4. Active Vectoring: Dead-level body attitude (0.0° roll, 0.0° pitch) while translating laterally.
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
parser.add_argument("--speed", type=float, default=2.5, help="Translational fly speed in m/s (default: 2.5)")
parser.add_argument("--yaw-rate", type=float, default=1.0, help="Yaw rotation speed in rad/s (default: 1.0)")
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


class InteractiveKeyboardListener:
    """Non-blocking keyboard listener running in a background thread."""
    def __init__(self):
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.cmd_yaw_rate = 0.0
        self.wind_mode = 1 # 1: Calm, 2: Beaufort 3, 3: Beaufort 5, 4: Beaufort 6
        self.reset_req = False
        self.hold_req = False
        self.running = True
        self.last_key = "Hover"
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
                    if ch in ('\x1b', 'x', 'X'): # ESC or X to quit
                        self.running = False
                        break
                    self._handle_key(ch)
                else:
                    # Decay continuous translation when key is released
                    self.cmd_vx *= 0.88
                    self.cmd_vy *= 0.88
                    self.cmd_vz *= 0.88
                    self.cmd_yaw_rate *= 0.85
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, ch):
        speed = args_cli.speed
        yaw_rate = args_cli.yaw_rate

        if ch in ('w', 'W'):
            self.cmd_vx = speed
            self.last_key = "Forward (+X)"
        elif ch in ('s', 'S'):
            self.cmd_vx = -speed
            self.last_key = "Backward (-X)"
        elif ch in ('a', 'A'):
            self.cmd_vy = -speed
            self.last_key = "Left (-Y)"
        elif ch in ('d', 'D'):
            self.cmd_vy = speed
            self.last_key = "Right (+Y)"
        elif ch in (' ', 'e', 'E'): # Space or E for Climb
            self.cmd_vz = speed * 0.7
            self.last_key = "Climb (+Z)"
        elif ch in ('c', 'C'): # C for Descend
            self.cmd_vz = -speed * 0.7
            self.last_key = "Descend (-Z)"
        elif ch in ('q', 'Q'):
            self.cmd_yaw_rate = yaw_rate
            self.last_key = "Yaw Left"
        elif ch in ('r', 'R'):
            self.cmd_yaw_rate = -yaw_rate
            self.last_key = "Yaw Right"
        elif ch in ('h', 'H'):
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_vz = 0.0
            self.cmd_yaw_rate = 0.0
            self.hold_req = True
            self.last_key = "Hold Position"
        elif ch in ('0', 'o', 'O'):
            self.reset_req = True
            self.last_key = "Reset Origin"
        elif ch == '1':
            self.wind_mode = 1
            self.last_key = "Wind: Calm (0 m/s)"
        elif ch == '2':
            self.wind_mode = 2
            self.last_key = "Wind: Beaufort 3 (4.5 m/s)"
        elif ch == '3':
            self.wind_mode = 3
            self.last_key = "Wind: Beaufort 5 (9.0 m/s)"
        elif ch == '4':
            self.wind_mode = 4
            self.last_key = "Wind: Beaufort 6 Gale (14 m/s + 15N Push)"


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
        self.lat = 47.397742 * 1e7 # Default Zurich airport reference (standard PX4 origin)
        self.lon = 8.545594 * 1e7
        self.alt_base = 488.0

        if HAS_PYMAVLINK:
            try:
                # Send out to QGC on 14550
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
                # msg.x: pitch (-1000..1000), msg.y: roll (-1000..1000), msg.z: throttle (0..1000), msg.r: yaw (-1000..1000)
                self.qgc_vx = (msg.x / 1000.0) * args_cli.speed
                self.qgc_vy = (msg.y / 1000.0) * args_cli.speed
                self.qgc_vz = ((msg.z - 500.0) / 500.0) * args_cli.speed * 0.7
                self.qgc_yaw_rate = (msg.r / 1000.0) * args_cli.yaw_rate
                self.qgc_active = True
            elif mtype == "SET_POSITION_TARGET_LOCAL_NED":
                # Convert NED to World
                self.qgc_vx = msg.vx if hasattr(msg, "vx") else 0.0
                self.qgc_vy = msg.vy if hasattr(msg, "vy") else 0.0
                self.qgc_vz = -msg.vz if hasattr(msg, "vz") else 0.0
                self.qgc_active = True

        # Send Periodic Telemetry to QGroundControl
        now = time.time()
        if not hasattr(self, "_last_heartbeat") or now - self._last_heartbeat >= 1.0:
            self._last_heartbeat = now
            # Heartbeat
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_PX4,
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED | mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE
            )
            # System status (Battery 100%, 16.8V, 4S)
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
            float(-pos_w[2]), # -Z for Down
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
    print("    VIRGO UAV: LIVE INTERACTIVE 3D FLIGHT IN ISAAC SIM & QGROUNDCONTROL    ")
    print("=" * 85)
    print(" [Controls Guide - Keyboard Teleoperation]")
    print("   * W / S : Fly Forward / Backward (+X / -X)")
    print("   * A / D : Fly Left / Right (-Y / +Y) [Decoupled Zero-Tilt Vectoring!]")
    print("   * Space / C : Fly Up / Fly Down (+Z / -Z)")
    print("   * Q / R : Yaw Turn Left / Yaw Turn Right")
    print("   * H     : Hold Position / Hover in Place")
    print("   * 0     : Reset to Center Hover (X=0, Y=0, Z=2.0m)")
    print("   * 1/2/3/4: Dynamic Wind Injection (1:Calm, 2:Beaufort3, 3:Beaufort5, 4:Beaufort6)")
    print("   * ESC/X : Exit Flight Session")
    print("=" * 85)
    print(f" [QGroundControl Bridge] Broadcasting MAVLink telemetry to 127.0.0.1:{args_cli.mavlink_port}")
    print("   Launch QGC in another terminal: /home/duc/QGroundControl-v4.3.0.AppImage")
    print("=" * 85)

    cfg = VirgoEnvCfg()
    cfg.is_eval = True
    cfg.interactive_mode = True
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 6.0

    env = VirgoEnv(cfg=cfg)
    device = env.device

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

    # Visual Target Waypoint Marker (Glowing Golden/Red Sphere)
    target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetWaypoint",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.10,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.35, 0.1),
                    emissive_color=(0.95, 0.45, 0.05)
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
                radius=0.06,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.8, 1.0),
                    emissive_color=(0.1, 0.6, 0.9)
                ),
            )
        }
    )
    wind_arrow_visualizer = VisualizationMarkers(wind_arrow_cfg)

    # Initialize teleop listeners
    kb = InteractiveKeyboardListener()
    mav_bridge = MavlinkQGCBridge(port=args_cli.mavlink_port)

    # Interactive Target Coordinates (Starts at Hover: 0.0, 0.0, 2.0 m)
    curr_target_pos = torch.tensor([0.0, 0.0, 2.0], device=device)
    curr_target_yaw = 0.0
    sim_dt = 0.02 # 50 Hz control step
    t_sim = 0.0
    last_hud_time = time.time()

    # Initial camera view nicely framing the drone
    try:
        env.sim.set_camera_view(eye=[0.0, -5.0, 3.2], target=[0.0, 0.0, 1.8])
    except Exception:
        pass

    print("\n--> Drone is Airborne & Ready for Live Command! Start pressing W/A/S/D or use QGC!\n")

    while simulation_app.is_running() and kb.running:
        # Check Reset command
        if kb.reset_req:
            curr_target_pos = torch.tensor([0.0, 0.0, 2.0], device=device)
            curr_target_yaw = 0.0
            kb.reset_req = False

        # Fuse Keyboard and QGroundControl inputs
        vx = kb.cmd_vx
        vy = kb.cmd_vy
        vz = kb.cmd_vz
        w_yaw = kb.cmd_yaw_rate

        if mav_bridge.qgc_active:
            if abs(mav_bridge.qgc_vx) > 0.05 or abs(mav_bridge.qgc_vy) > 0.05 or abs(mav_bridge.qgc_vz) > 0.05:
                vx = mav_bridge.qgc_vx
                vy = mav_bridge.qgc_vy
                vz = mav_bridge.qgc_vz
                w_yaw = mav_bridge.qgc_yaw_rate

        # Integrate velocity commands into target position & yaw
        curr_target_pos[0] += vx * sim_dt
        curr_target_pos[1] += vy * sim_dt
        curr_target_pos[2] = torch.clamp(curr_target_pos[2] + vz * sim_dt, min=0.6, max=10.0)
        curr_target_yaw += w_yaw * sim_dt

        # Inject into environment desired targets
        env._desired_pos_w[0] = curr_target_pos
        env._desired_vel_w[0] = torch.tensor([vx, vy, vz], device=device)
        env._desired_acc_w[0] = torch.tensor([0.0, 0.0, 0.0], device=device)

        # Desired Quaternion (Roll=0.0°, Pitch=0.0°, Yaw=curr_target_yaw)
        sp = math.sin(curr_target_yaw * 0.5)
        cp = math.cos(curr_target_yaw * 0.5)
        env._desired_quat_w[0, 0] = cp
        env._desired_quat_w[0, 1] = 0.0
        env._desired_quat_w[0, 2] = 0.0
        env._desired_quat_w[0, 3] = sp
        env._psi_des[0] = curr_target_yaw
        env._desired_yaw_rate[0] = w_yaw

        # Set Environmental Wind according to user selection (1/2/3/4)
        if kb.wind_mode == 1: # Calm
            w_spd = 0.0
            w_dir = torch.tensor([0.0, 0.0, 0.0], device=device)
            impulse_f = 0.0
            stage_str = "Calm (0 m/s)"
        elif kb.wind_mode == 2: # Beaufort 3
            w_spd = 4.5
            w_dir = torch.tensor([1.0, 0.0, 0.0], device=device)
            impulse_f = 0.0
            stage_str = "Beaufort 3 (4.5 m/s)"
        elif kb.wind_mode == 3: # Beaufort 5
            w_spd = 9.0 + 1.5 * math.sin(3.0 * t_sim)
            w_dir = torch.tensor([0.707, 0.707, 0.0], device=device)
            impulse_f = 0.0
            stage_str = "Beaufort 5 (9.0 m/s)"
        else: # Beaufort 6 Gale + Push
            w_spd = 14.0 + 3.0 * math.sin(4.0 * t_sim)
            w_dir = torch.tensor([0.866, -0.5, 0.1], device=device)
            in_pulse = (int(t_sim * 2.0) % 4 == 0) and (t_sim % 0.5 < 0.15)
            impulse_f = 15.0 if in_pulse else 0.0
            stage_str = f"Beaufort 6 (14 m/s + 15N Push!)"

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

        # On-Screen HUD in Terminal (Every 0.25s)
        if time.time() - last_hud_time >= 0.25:
            last_hud_time = time.time()
            qw, qx, qy, qz = cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3]
            roll_deg = math.degrees(math.atan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy)))
            pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (qw*qy - qz*qx)))))
            yaw_deg = math.degrees(math.atan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz)))
            pos_err = np.linalg.norm(cur_pos - curr_target_pos.cpu().numpy()) * 100.0

            qgc_badge = "[QGC LINKED]" if mav_bridge.connected else "[QGC READY]"
            pulse_badge = " [💥 15N PUSH!]" if impulse_f > 0.0 else ""

            print(
                f"\r[{t_sim:5.1f}s | {kb.last_key:<16}] "
                f"Pos: ({cur_pos[0]:+4.1f}, {cur_pos[1]:+4.1f}, {cur_pos[2]:4.1f}m) | "
                f"Tilt: (R:{roll_deg:+4.1f}°, P:{pitch_deg:+4.1f}°) | "
                f"Gimbal β: {np.mean(np.abs(servo_b)):4.1f}° | "
                f"Err: {pos_err:4.1f}cm | "
                f"{stage_str}{pulse_badge} {qgc_badge}",
                end="", flush=True
            )

        t_sim += sim_dt

    print("\n\n[Interactive Flight] Flight session ended cleanly.")
    kb.running = False
    simulation_app.close()


if __name__ == "__main__":
    main()
