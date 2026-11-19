import sys
sys.path.append("/home/duc/Raptor_UAV/Virgo_RL")
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

prop_vels = torch.tensor([[400.0, -400.0, 400.0, -400.0, 400.0, -400.0, 400.0, -400.0]], device=robot.device)
servo_targets = torch.tensor([[0.5, -0.5, 0.5, -0.5, 0.3, -0.3, 0.3, -0.3]], device=robot.device)

for step in range(50):
    robot.set_joint_velocity_target(prop_vels, joint_ids=prop_joint_ids)
    robot.set_joint_position_target(servo_targets, joint_ids=servo_joint_ids)
    robot.write_data_to_sim()
    sim.step()
    robot.update(0.01)

prop_vel = robot.data.joint_vel[0, prop_joint_ids]
servo_pos = robot.data.joint_pos[0, servo_joint_ids]

print("Final Prop velocities:", prop_vel.cpu().numpy())
print("Final Servo positions:", servo_pos.cpu().numpy())
print("Expected Servo positions:", servo_targets.cpu().numpy())

app.close()
