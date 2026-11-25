#!/usr/bin/env python3
"""
================================================================================
VIRGO UAV 12-DOF MORPHQUAD — PROFESSIONAL FLIGHT CONTROLLER (GCS HUD)
High-Speed UDP IPC Teleoperation Client for Isaac Sim Simulation
================================================================================
"""

import sys
import os
import time
import math
import json
import socket
import select
import termios
import tty

UDP_HOST = "127.0.0.1"
UDP_SEND_PORT = 9876
UDP_RECV_PORT = 9877


def get_key(fd, is_tty=True):
    """Reads a single keypress or escape sequence non-blockingly."""
    if not is_tty:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            line = sys.stdin.readline()
            if not line:
                return "ESC"
            return line.strip()[:1]
        return None

    rlist, _, _ = select.select([fd], [], [], 0.04)
    if not rlist:
        return None
    try:
        ch1 = os.read(fd, 1).decode("latin1", errors="ignore")
    except Exception:
        return None
    if ch1 == "\x1b":
        # Potential arrow key sequence
        rlist, _, _ = select.select([fd], [], [], 0.02)
        if rlist:
            try:
                ch2 = os.read(fd, 1).decode("latin1", errors="ignore")
                if ch2 == "[":
                    ch3 = os.read(fd, 1).decode("latin1", errors="ignore")
                    if ch3 == "A": return "UP"
                    elif ch3 == "B": return "DOWN"
                    elif ch3 == "C": return "RIGHT"
                    elif ch3 == "D": return "LEFT"
            except Exception:
                pass
        return "ESC"
    return ch1


def make_bar(val, max_val=15.0, width=22):
    """Draws a smooth graphical unicode progress bar."""
    ratio = max(0.0, min(1.0, val / max_val))
    filled = int(round(ratio * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def render_dashboard(step_m, step_yaw, last_cmd, drone_state, wind_name, wind_mode, is_connected, fps):
    """Draws a clean, publication-grade, flicker-free ASCII flight control HUD."""
    out = "\033[H\033[J" # Move cursor home and clear screen

    # Header Box
    out += "\033[1;36m"
    out += "╔══════════════════════════════════════════════════════════════════════════════════════╗\n"
    out += "║             🚀 VIRGO UAV 12-DOF MORPHQUAD GROUND CONTROL TERMINAL                    ║\n"
    out += "║         Decoupled Omni-Directional Vectoring & Disturbance Rejection                 ║\n"
    out += "╠══════════════════════════════════════════════════════════════════════════════════════╣\n"
    out += "\033[0m"

    # Status Bar
    if is_connected:
        conn_str = f"\033[1;32m● ONLINE ({fps:4.1f} Hz)\033[0m"
        t_sim = drone_state.get("time", 0.0) if drone_state else 0.0
        out += f"║  STATUS: {conn_str}  │  SIM TIME: \033[1;37m{t_sim:5.1f} s\033[0m  │  PORT: \033[1;33m127.0.0.1:9876/9877\033[0m         ║\n"
    else:
        conn_str = "\033[1;33m○ WAITING FOR SIMULATION (Run Terminal 1: ./run_isaac_sim.sh)\033[0m"
        out += f"║  STATUS: {conn_str:<82}║\n"

    out += "\033[1;36m╠══════════════════════════════════════════════════════════════════════════════════════╣\033[0m\n"

    # Navigation & Coordinates Section
    out += "║  \033[1;37m[NAVIGATION & POSITION TRACKING]\033[0m                                                    ║\n"
    if drone_state and is_connected:
        pos = drone_state.get("pos", [0, 0, 0])
        tgt = drone_state.get("target", [0, 0, 0])
        vel = drone_state.get("vel", [0, 0, 0])
        spd = drone_state.get("speed", 0.0)
        yaw = drone_state.get("yaw", 0.0)
        tgt_yaw = drone_state.get("target_yaw", 0.0)
        yaw_err = drone_state.get("yaw_err", 0.0)
        pos_err = drone_state.get("pos_err", 0.0)

        err_badge = "\033[1;32m[OPTIMAL]\033[0m" if pos_err < 15.0 else ("\033[1;33m[TRACKING]\033[0m" if pos_err < 40.0 else "\033[1;31m[ADJUSTING]\033[0m")

        out += f"║    Target Coords :  X = \033[1;33m{tgt[0]:+5.2f} m\033[0m  │  Y = \033[1;33m{tgt[1]:+5.2f} m\033[0m  │  Z = \033[1;33m{tgt[2]:4.2f} m\033[0m  │  Ψ = \033[1;33m{tgt_yaw:+5.1f}°\033[0m  ║\n"
        out += f"║    Drone Coords  :  X = \033[1;32m{pos[0]:+5.2f} m\033[0m  │  Y = \033[1;32m{pos[1]:+5.2f} m\033[0m  │  Z = \033[1;32m{pos[2]:4.2f} m\033[0m  │  Ψ = \033[1;32m{yaw:+5.1f}°\033[0m  ║\n"
        out += f"║    Precision     :  Error: \033[1;37m{pos_err:4.1f} cm\033[0m {err_badge}  │  Airspeed: \033[1;37m{spd:4.2f} m/s\033[0m  │  Yaw Err: \033[1;37{yaw_err:4.1f}°\033[0m  ║\n"
    else:
        out += "║    Waiting for active telemetry feed from Isaac Sim...                               ║\n"
        out += "║    (Drone starts at Hover: X=0.00m, Y=0.00m, Z=2.00m)                                ║\n"
        out += "║                                                                                      ║\n"

    out += "\033[1;36m╠══════════════════════════════════════════════════════════════════════════════════════╣\033[0m\n"

    # Decoupled Attitude & Actuator Status
    out += "║  \033[1;37m[DECOUPLED ATTITUDE & DYNAMIXEL VECTORING]\033[0m                                          ║\n"
    if drone_state and is_connected:
        tilt = drone_state.get("tilt", [0, 0])
        beta = drone_state.get("gimbal_beta", drone_state.get("gimbal", 0.0))
        alpha = drone_state.get("gimbal_alpha", 0.0)

        out += f"║    Body Attitude :  Roll = \033[1;32m{tilt[0]:+4.1f}°\033[0m  │  Pitch = \033[1;32m{tilt[1]:+4.1f}°\033[0m  \033[1;32m(0.0° Dead-Level Body!)\033[0m        ║\n"
        out += f"║    Dynamixel β   :  \033[1;33m{beta:4.1f}° / 90.0°\033[0m   [Active Lateral Force & Wind Resistance]        ║\n"
        out += f"║    Dynamixel α   :  \033[1;33m{alpha:4.1f}° / 180.0°\033[0m  [Continuous Azimuth Vectoring & Heading Yaw]    ║\n"
    else:
        out += "║    Body Attitude :  Roll = 0.0°  │  Pitch = 0.0° (Dead-Level Translation)            ║\n"
        out += "║    Actuators     :  8 Motors + 8 Dynamixel Servos ready                              ║\n"
        out += "║                                                                                      ║\n"

    out += "\033[1;36m╠══════════════════════════════════════════════════════════════════════════════════════╣\033[0m\n"

    # Environmental Disturbance & Wind Gauge
    out += "║  \033[1;37m[BEAUFORT WIND REGIME & DISTURBANCES]\033[0m                                               ║\n"
    w_spd = drone_state.get("wind_spd", 0.0) if (drone_state and is_connected) else (0.0 if wind_mode == 1 else (4.5 if wind_mode == 2 else (9.0 if wind_mode == 3 else 14.0)))
    w_dir = drone_state.get("wind_dir", [0, 0, 0]) if (drone_state and is_connected) else [0, 0, 0]
    impulse_f = drone_state.get("impulse", 0.0) if (drone_state and is_connected) else 0.0

    bar_str = make_bar(w_spd, max_val=15.0, width=22)
    if impulse_f > 1.0:
        impulse_str = "\033[1;31;5m💥 15.0N GUST SHOCKWAVE ACTIVE!\033[0m"
    else:
        impulse_str = "\033[1;32mLaminar Streamline Flow\033[0m"

    out += f"║    Active Preset :  \033[1;35m[{wind_mode}] {wind_name:<50}\033[0m║\n"
    out += f"║    Wind Velocity :  [\033[1;36m{bar_str}\033[0m] \033[1;37m{w_spd:4.1f} m/s\033[0m (Vx:{w_dir[0]:+.1f}, Vy:{w_dir[1]:+.1f})           ║\n"
    out += f"║    Impulse Push  :  {impulse_str:<70}║\n"

    out += "\033[1;36m╠══════════════════════════════════════════════════════════════════════════════════════╣\033[0m\n"

    # Flight Command Matrix
    out += "║  \033[1;37m[FLIGHT CONTROLS & SHORTCUTS]\033[0m                                                       ║\n"
    out += "║    \033[1;32m[W / ↑]\033[0m Fly Forward (+X heading)      \033[1;33m[Space / E]\033[0m Climb Up (+Z)                     ║\n"
    out += "║    \033[1;32m[S / ↓]\033[0m Fly Backward (-X heading)     \033[1;33m[C]        \033[0m Descend Down (-Z)                 ║\n"
    out += "║    \033[1;32m[A / ←]\033[0m Strafe Left (0.0° Body Tilt!) \033[1;33m[Q]        \033[0m Yaw Rotate Left                   ║\n"
    out += "║    \033[1;32m[D / →]\033[0m Strafe Right (0.0° Body Tilt!)\033[1;33m[R]        \033[0m Yaw Rotate Right                  ║\n"
    out += "║    ──────────────────────────────────────────────────────────────────────────────    ║\n"
    out += "║    \033[1;35m[1]\033[0m Calm (0 m/s)              \033[1;35m[2]\033[0m Beaufort 3 (4.5 m/s)                            ║\n"
    out += "║    \033[1;35m[3]\033[0m Beaufort 5 (9.0 m/s)      \033[1;35m[4]\033[0m Beaufort 6 (14 m/s + 15N Gust Push!)            ║\n"
    out += f"║    \033[1;34m[H]\033[0m Hold Hover  \033[1;34m[0]\033[0m Reset Origin  \033[1;34m[+/-]\033[0m Step: \033[1;33m{step_m:.2f}m\033[0m   \033[1;34m[[/]]\033[0m Yaw: \033[1;33m{math.degrees(step_yaw):.1f}°\033[0m        ║\n"
    out += "║    \033[1;31m[ESC / X]\033[0m Exit Flight Controller                                                  ║\n"

    out += "\033[1;36m╠══════════════════════════════════════════════════════════════════════════════════════╣\033[0m\n"

    # Action Status
    out += f"║  LAST ACTION: \033[1;32m{last_cmd:<70}\033[0m║\n"
    out += "\033[1;36m╚══════════════════════════════════════════════════════════════════════════════════════╝\033[0m\n"
    out += "  \033[2mTip: Keep this terminal focused while commanding flight in Isaac Sim!\033[0m\n"

    sys.stdout.write(out)
    sys.stdout.flush()


def main():
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock_recv.bind((UDP_HOST, UDP_RECV_PORT))
        sock_recv.setblocking(False)
    except Exception as e:
        print(f"Notice: Could not bind UDP telemetry receiver ({e})")

    step_m = 0.8       # 0.8 meter horizontal step per keypress
    step_z = 0.4       # 0.4 meter altitude step per keypress
    step_yaw = 0.2618  # 15.0 degrees per keypress

    wind_names = {
        1: "Calm (0 m/s)",
        2: "Beaufort 3 - Gentle Breeze (4.5 m/s)",
        3: "Beaufort 5 - Fresh Breeze (9.0 m/s)",
        4: "Beaufort 6 - Gale (14 m/s + 15N Gust Push!)"
    }
    wind_mode = 1
    last_cmd = "System Ready - Press W/A/S/D to Fly"
    drone_state = None
    last_rx_time = 0.0
    packet_count = 0
    fps = 0.0
    last_fps_calc = time.time()

    is_tty = sys.stdin.isatty()
    old_settings = None
    fd = None
    if is_tty:
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            is_tty = False

    # Initial dashboard render
    render_dashboard(step_m, step_yaw, last_cmd, drone_state, wind_names[wind_mode], wind_mode, False, 0.0)
    last_render_time = time.time()

    try:
        while True:
            # Drain telemetry queue to obtain the freshest packet
            while True:
                try:
                    data, _ = sock_recv.recvfrom(4096)
                    drone_state = json.loads(data.decode())
                    last_rx_time = time.time()
                    packet_count += 1
                    if "wind_mode" in drone_state:
                        wind_mode = int(drone_state["wind_mode"])
                except (BlockingIOError, socket.error):
                    break
                except Exception:
                    break

            now = time.time()
            if now - last_fps_calc >= 1.0:
                fps = packet_count / (now - last_fps_calc)
                packet_count = 0
                last_fps_calc = now

            is_connected = (now - last_rx_time) < 1.5

            # Read user keypress non-blockingly
            key = get_key(fd, is_tty)
            packet = None

            if key in ('x', 'X', 'ESC', '\x03'):
                break
            elif key in ('w', 'W', 'UP'):
                packet = {"nudge_forward": step_m}
                last_cmd = f"FLY FORWARD (+{step_m:.2f}m in heading direction)"
            elif key in ('s', 'S', 'DOWN'):
                packet = {"nudge_forward": -step_m}
                last_cmd = f"FLY BACKWARD (-{step_m:.2f}m in heading direction)"
            elif key in ('a', 'A', 'LEFT'):
                packet = {"nudge_strafe": step_m}
                last_cmd = f"STRAFE LEFT (+{step_m:.2f}m - 0.0° Dead-Level Tilt!)"
            elif key in ('d', 'D', 'RIGHT'):
                packet = {"nudge_strafe": -step_m}
                last_cmd = f"STRAFE RIGHT (-{step_m:.2f}m - 0.0° Dead-Level Tilt!)"
            elif key in (' ', 'e', 'E'):
                packet = {"nudge_z": step_z}
                last_cmd = f"CLIMB UP (+{step_z:.2f}m Altitude)"
            elif key in ('c', 'C'):
                packet = {"nudge_z": -step_z}
                last_cmd = f"DESCEND DOWN (-{step_z:.2f}m Altitude)"
            elif key in ('q', 'Q'):
                packet = {"nudge_yaw": step_yaw}
                last_cmd = f"YAW TURN LEFT (+{int(round(math.degrees(step_yaw)))}° Heading)"
            elif key in ('r', 'R'):
                packet = {"nudge_yaw": -step_yaw}
                last_cmd = f"YAW TURN RIGHT (-{int(round(math.degrees(step_yaw)))}° Heading)"
            elif key in ('h', 'H'):
                packet = {"hold": True}
                last_cmd = "HOVER HOLD (Maintaining Current Position)"
            elif key == '0':
                packet = {"reset": True}
                last_cmd = "RESET TO ORIGIN (Hover at 0, 0, 2m)"
            elif key in ('1', '2', '3', '4'):
                wind_mode = int(key)
                packet = {"wind_mode": wind_mode}
                last_cmd = f"SET WIND DISTURBANCE: {wind_names[wind_mode]}"
            elif key in ('+', '='):
                step_m = round(min(3.0, step_m + 0.2), 2)
                step_z = round(min(2.0, step_z + 0.1), 2)
                last_cmd = f"INCREASED STEP SIZE: {step_m:.2f}m"
            elif key in ('-', '_'):
                step_m = round(max(0.2, step_m - 0.2), 2)
                step_z = round(max(0.1, step_z - 0.1), 2)
                last_cmd = f"DECREASED STEP SIZE: {step_m:.2f}m"
            elif key in (']', '}'):
                step_yaw = round(min(math.radians(45.0), step_yaw + math.radians(5.0)), 4)
                last_cmd = f"INCREASED YAW STEP: {int(round(math.degrees(step_yaw)))}°"
            elif key in ('[', '{'):
                step_yaw = round(max(math.radians(5.0), step_yaw - math.radians(5.0)), 4)
                last_cmd = f"DECREASED YAW STEP: {int(round(math.degrees(step_yaw)))}°"

            # Transmit UDP packet to Isaac Sim
            if packet is not None:
                msg_bytes = json.dumps(packet).encode()
                sock_send.sendto(msg_bytes, (UDP_HOST, UDP_SEND_PORT))
                render_dashboard(step_m, step_yaw, last_cmd, drone_state, wind_names[wind_mode], wind_mode, is_connected, fps)
                last_render_time = time.time()

            # Refresh dashboard at 12 Hz for fluid HUD numbers
            now = time.time()
            if now - last_render_time >= 0.08:
                render_dashboard(step_m, step_yaw, last_cmd, drone_state, wind_names[wind_mode], wind_mode, is_connected, fps)
                last_render_time = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        if is_tty and old_settings is not None and fd is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
        sock_send.close()
        sock_recv.close()
        print("\n\n[Flight Controller] Exited cleanly. Terminal restored.\n")


if __name__ == "__main__":
    main()
