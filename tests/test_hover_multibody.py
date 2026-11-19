import os
import sys
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import torch
import math
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from virgo_robot_cfg import VIRGO_CFG
from isaaclab.assets import Articulation
import isaaclab.sim as sim_utils

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
robot = Articulation(VIRGO_CFG.replace(prim_path="/World/Robot"))
sim.reset()

shaft_ids, _ = robot.find_bodies(["SHAFT_MOTOR_1", "SHAFT_MOTOR_2", "SHAFT_MOTOR_3", "SHAFT_MOTOR_4"])
prop_joint_ids, _ = robot.find_joints(["PROP_.*"])
servo_joint_ids, _ = robot.find_joints(["ARM_MOTOR_.*", "SHAFT_MOTOR_.*"])

# Total mass = 4.56 kg -> mg = 44.7336 N -> 11.1834 N per arm
# Let's test hover: apply 11.1834 N on each SHAFT_MOTOR_i in world Z
forces = torch.zeros(1, 4, 3, device=robot.device)
forces[0, :, 2] = 11.1834
torques = torch.zeros(1, 4, 3, device=robot.device)

prop_vels = torch.tensor([[500.0, -500.0, 500.0, -500.0, 500.0, -500.0, 500.0, -500.0]], device=robot.device)
servo_targets = torch.zeros(1, 8, device=robot.device)

body_ids = torch.tensor(shaft_ids, dtype=torch.int, device=robot.device)

for step in range(100):
    robot.permanent_wrench_composer.set_forces_and_torques(
        body_ids=body_ids,
        forces=forces,
        torques=torques,
        is_global=True
    )
    robot.set_joint_velocity_target(prop_vels, joint_ids=prop_joint_ids)
    robot.set_joint_position_target(servo_targets, joint_ids=servo_joint_ids)
    robot.write_data_to_sim()
    sim.step()
    robot.update(0.01)

pos = robot.data.root_pos_w[0].cpu().numpy()
vel = robot.data.root_lin_vel_w[0].cpu().numpy()
quat = robot.data.root_quat_w[0].cpu().numpy()
print(f"Hover test after 1s (100 steps):")
print(f"Position: {pos}")
print(f"Velocity: {vel}")
print(f"Orientation (quat): {quat}")

app.close()
