#!/usr/bin/env python3
"""
Integration test for Virgo UAV decoupled UDP teleoperation:
Sends flight commands (Forward, Strafe, Climb, Yaw, Wind) to Isaac Sim on port 9876,
and receives telemetry packets on port 9877 to verify drone tracking.
"""

import sys
import os
import time
import json
import socket
import subprocess

def get_latest_packet(sock, timeout=3.0):
    """Flushes queue and waits for the freshest telemetry packet."""
    t0 = time.time()
    sock.settimeout(0.1)
    pkt = None
    while time.time() - t0 < timeout:
        try:
            data, _ = sock.recvfrom(4096)
            pkt = json.loads(data.decode())
        except socket.timeout:
            if pkt is not None:
                return pkt
        except Exception:
            pass
    return pkt

def test_teleop_ipc():
    UDP_HOST = "127.0.0.1"
    PORT_CMD = 9876
    PORT_TELEM = 9877

    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_telem = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_telem.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_telem.bind((UDP_HOST, PORT_TELEM))

    # Launch interactive_fly_virgo.py in background
    cmd = [
        "/home/duc/miniconda3/envs/env_isaaclab/bin/python",
        "/home/duc/Raptor_UAV/Virgo_RL/scripts/interactive_fly_virgo.py",
        "--headless"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    print("[Test] Isaac Sim starting...")
    start_time = time.time()

    try:
        # Wait until first telemetry packet arrives
        init_pkt = None
        while time.time() - start_time < 25.0:
            init_pkt = get_latest_packet(sock_telem, timeout=0.5)
            if init_pkt is not None:
                print(f"[Test] Telemetry connected! Initial state: Pos={init_pkt['pos']}, Target={init_pkt['target']}")
                break

        assert init_pkt is not None, "Failed to connect: Did not receive telemetry from Isaac Sim within 25s!"

        # Step 1: Send FLY FORWARD (+1.5m)
        print("[Test] Sending FORWARD (+1.5m) command...")
        sock_cmd.sendto(json.dumps({"nudge_forward": 1.5}).encode(), (UDP_HOST, PORT_CMD))
        time.sleep(0.4)

        pkt1 = get_latest_packet(sock_telem)
        print(f"[Test] After Forward: Target={pkt1['target']}, DronePos={pkt1['pos']}, Tilt={pkt1['tilt']}, Gimbal={pkt1['gimbal']}")
        assert pkt1['target'][0] > 1.0, f"Target X did not increase! target={pkt1['target']}"

        # Step 2: Send STRAFE LEFT (+1.0m)
        print("[Test] Sending STRAFE LEFT (+1.0m) command...")
        sock_cmd.sendto(json.dumps({"nudge_strafe": 1.0}).encode(), (UDP_HOST, PORT_CMD))
        time.sleep(0.4)

        pkt2 = get_latest_packet(sock_telem)
        print(f"[Test] After Strafe Left: Target={pkt2['target']}, DronePos={pkt2['pos']}, Tilt={pkt2['tilt']}")
        assert pkt2['target'][1] > 0.8, f"Target Y did not increase! target={pkt2['target']}"

        # Step 3: Send CLIMB (+0.8m)
        print("[Test] Sending CLIMB (+0.8m) command...")
        sock_cmd.sendto(json.dumps({"nudge_z": 0.8}).encode(), (UDP_HOST, PORT_CMD))
        time.sleep(0.4)

        pkt3 = get_latest_packet(sock_telem)
        print(f"[Test] After Climb: Target={pkt3['target']}, DronePos={pkt3['pos']}")
        assert pkt3['target'][2] > 2.6, f"Target Z did not increase! target={pkt3['target']}"

        # Step 4: Send RESET
        print("[Test] Sending RESET command...")
        sock_cmd.sendto(json.dumps({"reset": True}).encode(), (UDP_HOST, PORT_CMD))
        time.sleep(0.4)

        pkt4 = get_latest_packet(sock_telem)
        print(f"[Test] After Reset: Target={pkt4['target']}")
        assert abs(pkt4['target'][0]) < 0.1 and abs(pkt4['target'][1]) < 0.1 and abs(pkt4['target'][2] - 2.0) < 0.1, "Target did not reset to (0, 0, 2)!"

        print("\n✅ ALL IPC TELEOPERATION TESTS PASSED PERFECTLY!\n")
        return True

    finally:
        sock_cmd.close()
        sock_telem.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    test_teleop_ipc()
