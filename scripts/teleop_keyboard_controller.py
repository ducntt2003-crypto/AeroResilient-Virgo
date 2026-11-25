#!/usr/bin/env python3
"""
================================================================================
VIRGO UAV: DEDICATED TERMINAL FLIGHT CONTROLLER (CLI TELEOP)
Runs in Terminal 3 (or Terminal 2) to command Virgo UAV in Isaac Sim via local UDP.
================================================================================
Controls:
  * W / Up Arrow    : Fly Forward (+X in Heading Direction)
  * S / Down Arrow  : Fly Backward (-X in Heading Direction)
  * A / Left Arrow  : Fly Left (Strafe Left, Zero Body Tilt!)
  * D / Right Arrow : Fly Right (Strafe Right, Zero Body Tilt!)
  * Space / E       : Climb (+Z Altitude)
  * C               : Descend (-Z Altitude)
  * Q               : Yaw Turn Left
  * R               : Yaw Turn Right
  * H               : Hover / Hold Position
  * 0               : Reset Origin (0, 0, 2m)
  * 1 / 2 / 3 / 4   : Select Wind (1:Calm, 2:Beaufort3, 3:Beaufort5, 4:Beaufort6)
  * + / -           : Increase / Decrease Step Size
  * ESC / X         : Quit Controller
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


def render_dashboard(step_m, last_cmd, drone_state, wind_name, is_connected):
    """Draws a clean, flicker-free ASCII flight control dashboard."""
    out = "\033[H\033[J" # Move cursor home and clear screen
    out += "\033[1;36m"
    out += "================================================================================\n"
    out += "              🎮 VIRGO UAV 12-DOF MORPHQUAD TERMINAL CONTROLLER                 \n"
    out += "================================================================================\n"
    out += "\033[0m"
    out += "  [Directional Controls]               [Altitude & Heading]    [Disturbances]\n"
    out += "    \033[1;32mW / ↑\033[0m : Forward (Heading dir)        \033[1;33mSpace / E\033[0m : Climb (+Z)      \033[1;35m1\033[0m: Calm (0 m/s)\n"
    out += "    \033[1;32mS / ↓\033[0m : Backward (Heading dir)       \033[1;33mC\033[0m         : Descend (-Z)    \033[1;35m2\033[0m: Beaufort 3 (4.5m/s)\n"
    out += "    \033[1;32mA / ←\033[0m : Strafe Left (0° Tilt!)       \033[1;33mQ\033[0m         : Yaw Left        \033[1;35m3\033[0m: Beaufort 5 (9.0m/s)\n"
    out += "    \033[1;32mD / →\033[0m : Strafe Right (0° Tilt!)      \033[1;33mR\033[0m         : Yaw Right       \033[1;35m4\033[0m: Beaufort 6 (14m/s+Push)\n"
    out += "  ------------------------------------------------------------------------------\n"
    out += "    \033[1;34mH\033[0m     : Hover / Hold Position         \033[1;34m0\033[0m         : Reset Origin    \033[1;34m+ / -\033[0m: Step Size\n"
    out += "    \033[1;31mESC/X\033[0m : Quit Controller\n"
    out += "================================================================================\n"
    conn_status = "\033[1;32m● CONNECTED (Isaac Sim Active)\033[0m" if is_connected else "\033[1;33m○ WAITING FOR SIMULATION (Run Terminal 1)\033[0m"
    out += f"  Status: {conn_status}\n"
    out += f"  \033[1;37mCommand Step:\033[0m \033[1;33m{step_m:.2f} m\033[0m  |  \033[1;37mActive Wind:\033[0m \033[1;35m{wind_name}\033[0m\n"
    out += f"  \033[1;37mLast Action:\033[0m  \033[1;32m{last_cmd:<40}\033[0m\n"
    out += "--------------------------------------------------------------------------------\n"

    if drone_state and is_connected:
        pos = drone_state.get("pos", [0, 0, 0])
        tgt = drone_state.get("target", [0, 0, 0])
        tilt = drone_state.get("tilt", [0, 0])
        servo = drone_state.get("gimbal", 0.0)
        yaw = drone_state.get("yaw", 0.0)
        t_sim = drone_state.get("time", 0.0)

        out += f"  \033[1;36m[FLIGHT TELEMETRY]\033[0m Sim Time: \033[1;37m{t_sim:5.1f}s\033[0m | Heading Yaw: \033[1;37m{yaw:+5.1f}°\033[0m\n"
        out += f"    Target Position : (\033[1;33m{tgt[0]:+5.2f}\033[0m, \033[1;33m{tgt[1]:+5.2f}\033[0m, \033[1;33m{tgt[2]:4.2f}\033[0mm)\n"
        out += f"    Drone Position  : (\033[1;32m{pos[0]:+5.2f}\033[0m, \033[1;32m{pos[1]:+5.2f}\033[0m, \033[1;32m{pos[2]:4.2f}\033[0mm)\n"
        out += f"    Body Tilt       : Roll \033[1;32m{tilt[0]:+4.1f}°\033[0m, Pitch \033[1;32m{tilt[1]:+4.1f}°\033[0m (Near 0.0° Dead-Level!)\n"
        out += f"    Gimbal Servos β : \033[1;33m{servo:4.1f}°\033[0m (Dynamixel Servos Actively Vectoring Thrust!)\n"
    else:
        out += "  \033[1;33m[TELEMETRY OFFLINE]\033[0m Launching Isaac Sim in Terminal 1 will stream live data here.\n"
        out += "                      Commands sent via UDP to 127.0.0.1:9876.\n"

    out += "================================================================================\n"
    out += "  \033[2mTip: Keep this terminal window focused to send keyboard commands!\033[0m\n"
    sys.stdout.write(out)
    sys.stdout.flush()


def main():
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock_recv.bind((UDP_HOST, UDP_RECV_PORT))
        sock_recv.setblocking(False)
    except Exception as e:
        print(f"Warning: Could not bind UDP telemetry receiver ({e})")

    step_m = 0.8  # 0.8 meter horizontal step per keypress
    step_z = 0.4  # 0.4 meter altitude step per keypress
    step_yaw = 0.35 # ~20 degrees per keypress

    wind_names = {
        1: "Calm (0 m/s)",
        2: "Beaufort 3 (4.5 m/s)",
        3: "Beaufort 5 (9.0 m/s)",
        4: "Beaufort 6 (14 m/s + 15N Push!)"
    }
    wind_mode = 1
    last_cmd = "System Ready - Press W/A/S/D to Fly"
    drone_state = None
    last_rx_time = 0.0

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
    render_dashboard(step_m, last_cmd, drone_state, wind_names[wind_mode], False)
    last_render_time = time.time()

    try:
        while True:
            # Check telemetry from Isaac Sim (port 9877)
            while True:
                try:
                    data, _ = sock_recv.recvfrom(2048)
                    drone_state = json.loads(data.decode())
                    last_rx_time = time.time()
                except (BlockingIOError, socket.error):
                    break
                except Exception:
                    break

            is_connected = (time.time() - last_rx_time) < 1.5

            # Read user keypress non-blockingly
            key = get_key(fd, is_tty)
            packet = None

            if key in ('x', 'X', 'ESC', '\x03'):
                break
            elif key in ('w', 'W', 'UP'):
                packet = {"nudge_forward": step_m}
                last_cmd = f"FLY FORWARD (+{step_m:.2f}m Ahead)"
            elif key in ('s', 'S', 'DOWN'):
                packet = {"nudge_forward": -step_m}
                last_cmd = f"FLY BACKWARD (-{step_m:.2f}m Back)"
            elif key in ('a', 'A', 'LEFT'):
                packet = {"nudge_strafe": step_m}
                last_cmd = f"STRAFE LEFT (+{step_m:.2f}m Left - Zero Tilt!)"
            elif key in ('d', 'D', 'RIGHT'):
                packet = {"nudge_strafe": -step_m}
                last_cmd = f"STRAFE RIGHT (-{step_m:.2f}m Right - Zero Tilt!)"
            elif key in (' ', 'e', 'E'):
                packet = {"nudge_z": step_z}
                last_cmd = f"CLIMB (+{step_z:.2f}m Up)"
            elif key in ('c', 'C'):
                packet = {"nudge_z": -step_z}
                last_cmd = f"DESCEND (-{step_z:.2f}m Down)"
            elif key in ('q', 'Q'):
                packet = {"nudge_yaw": step_yaw}
                last_cmd = f"YAW LEFT (+{int(math.degrees(step_yaw))}°)"
            elif key in ('r', 'R'):
                packet = {"nudge_yaw": -step_yaw}
                last_cmd = f"YAW RIGHT (-{int(math.degrees(step_yaw))}°)"
            elif key in ('h', 'H'):
                packet = {"hold": True}
                last_cmd = "HOLD POSITION / HOVER"
            elif key == '0':
                packet = {"reset": True}
                last_cmd = "RESET TO ORIGIN (0, 0, 2m)"
            elif key in ('1', '2', '3', '4'):
                wind_mode = int(key)
                packet = {"wind_mode": wind_mode}
                last_cmd = f"SET WIND: {wind_names[wind_mode]}"
            elif key in ('+', '='):
                step_m = round(min(3.0, step_m + 0.2), 2)
                step_z = round(min(2.0, step_z + 0.1), 2)
                last_cmd = f"INCREASED STEP: {step_m:.2f}m"
            elif key in ('-', '_'):
                step_m = round(max(0.2, step_m - 0.2), 2)
                step_z = round(max(0.1, step_z - 0.1), 2)
                last_cmd = f"DECREASED STEP: {step_m:.2f}m"

            # Transmit UDP packet to Isaac Sim
            if packet is not None:
                msg_bytes = json.dumps(packet).encode()
                sock_send.sendto(msg_bytes, (UDP_HOST, UDP_SEND_PORT))
                render_dashboard(step_m, last_cmd, drone_state, wind_names[wind_mode], is_connected)
                last_render_time = time.time()

            # Refresh dashboard at 10 Hz
            now = time.time()
            if now - last_render_time >= 0.10:
                render_dashboard(step_m, last_cmd, drone_state, wind_names[wind_mode], is_connected)
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
        print("\n\n[Terminal Controller] Exited cleanly.\n")


if __name__ == "__main__":
    main()
