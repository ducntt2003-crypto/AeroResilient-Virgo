# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Virgo_RL: DirectRLEnv for 12-DOF MorphQuad Omnidirectional UAV in Isaac Lab.
Paper: "MorphQuad: Morphable Quadrotor for Superhuman Maneuverability, Manipulation, and Resiliency"
ArXiv: 2607.02764 (Tzoumas Lab, University of Michigan)
"""

from __future__ import annotations

import math
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    subtract_frame_transforms,
)

from isaaclab.sensors import TiledCamera, TiledCameraCfg

from virgo_robot_cfg import VIRGO_CFG


@configclass
class VirgoEnvCfg(DirectRLEnvCfg):
    # Environment timing & spaces
    episode_length_s = 20.0 # Full 20s Figure-8 cycle for ample response time
    decimation = 2          # sim at 100 Hz (dt=0.01), RL control at 50 Hz
    action_space = 6        # 6-DOF commanded Body Wrench [fx, fy, fz, tau_x, tau_y, tau_z]
    observation_space = 35   # Decoupled 6-DOF tracking state observation
    state_space = 0
    debug_vis = False

    # Optional camera for rendering snapshots (disabled during headless training)
    enable_camera: bool = False
    camera: TiledCameraCfg | None = None

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Interactive Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # Robot Asset
    robot = VIRGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Robot specifications (MorphQuad)
    nominal_mass = 4.56 # kg
    arm_l = 0.22        # m
    arm_b = 0.22        # m
    max_thrust_per_arm = 28.0 # N
    max_torque = 12.0   # Nm

    # Flight Mode
    upright_only: bool = True # True: Always fly upright (Roll=0, Pitch=0), vectoring thrust via servos only
    is_eval: bool = False     # True: Standalone evaluation mode (no divergence reset, continuous loop)

    # Reward weights
    r_pos_scale = 5.0
    r_att_scale = 3.0
    r_vel_scale = 0.5
    r_ang_vel_scale = 0.5
    r_smooth_scale = -0.35      # Action rate penalty
    r_servo_vel_scale = -1.5    # Servo angle displacement penalty (forces smooth, steady holding)
    r_servo_jerk_scale = -0.05  # Servo rate displacement penalty (prevents sudden direction flips)
    r_quiescence_scale = 0.5    # Servo steady-state holding bonus
    r_gimbal_scale = -1.0


class BatchedVirgoKinematics:
    """
    Batched GPU PyTorch Null-Space Thrust Allocation for MorphQuad
    Eqs. (12), (13), (15), (16) in arXiv:2607.02764
    """
    def __init__(self, arm_l=0.22, arm_b=0.22, max_thrust_per_arm=28.0, device="cuda:0"):
        self.device = device
        self.l = arm_l
        self.b = arm_b
        self.r = math.sqrt(self.l**2 + self.b**2)
        self.max_thrust = max_thrust_per_arm

        # Rotor shaft positions in Body Frame: [4, 3] exactly matching SolidWorks CAD / URDF
        # Arm 1: SHAFT_MOTOR_1 (Front-Right: +x, -y)
        # Arm 2: SHAFT_MOTOR_2 (Rear-Left:   -x, +y)
        # Arm 3: SHAFT_MOTOR_3 (Front-Left:  +x, +y)
        # Arm 4: SHAFT_MOTOR_4 (Rear-Right:  -x, -y)
        positions = torch.tensor([
            [ 0.19899, -0.19780, 0.01664],
            [-0.19899,  0.19680, 0.01665],
            [ 0.19899,  0.19742, 0.01665],
            [-0.19899, -0.19742, 0.01664]
        ], dtype=torch.float32, device=device)

        # M Matrix: [6, 12] such that W_B = M * t
        M = torch.zeros(6, 12, dtype=torch.float32, device=device)
        for i in range(4):
            M[0:3, i*3:(i+1)*3] = torch.eye(3, device=device)
            px, py, pz = positions[i]
            skew = torch.tensor([
                [ 0.0, -pz,   py],
                [ pz,   0.0, -px],
                [-py,   px,   0.0]
            ], dtype=torch.float32, device=device)
            M[3:6, i*3:(i+1)*3] = skew

        self.M = M
        self.M_pinv = torch.pinverse(M) # [12, 6]

        # SVD Null-space basis: [12, 6]
        _, _, Vh = torch.linalg.svd(M, full_matrices=True)
        self.N_M = Vh[6:].T.contiguous() # [12, 6]

    def allocate(self, f_cmd: torch.Tensor, tau_cmd: torch.Tensor):
        """
        Batched Allocation:
        f_cmd: [B, 3]
        tau_cmd: [B, 3]
        Returns:
          thrusts: [B, 4]
          alphas:  [B, 4]
          betas:   [B, 4]
        """
        B = f_cmd.shape[0]
        w_des = torch.cat([f_cmd, tau_cmd], dim=-1) # [B, 6]

        # Step 1: Minimum-norm pseudo-inverse solution [B, 12]
        # Clean, energy-optimal, and perfectly symmetric across all 4 arms
        t_stacked = torch.matmul(w_des, self.M_pinv.T)

        # Step 3: Recover thrusts, alphas, betas for each arm
        thrusts = torch.zeros(B, 4, dtype=torch.float32, device=self.device)
        alphas = torch.zeros(B, 4, dtype=torch.float32, device=self.device)
        betas = torch.zeros(B, 4, dtype=torch.float32, device=self.device)

        for i in range(4):
            ti = t_stacked[:, i*3:(i+1)*3].clone()
            ti[:, 2] = ti[:, 2].clamp(min=1.0) # Enforce positive vertical thrust so rotors never flip upside down
            t_mag = torch.norm(ti, dim=-1).clamp(min=1e-4) # [B]
            thrusts[:, i] = t_mag.clamp(max=self.max_thrust)

            t_hat = ti / t_mag.unsqueeze(-1)
            # beta_i = arcsin(-t_y)
            betas[:, i] = torch.asin((-t_hat[:, 1]).clamp(-0.65, 0.65)) # max 37 deg
            # alpha_i = atan2(t_x, t_z)
            alphas[:, i] = torch.atan2(t_hat[:, 0], t_hat[:, 2]).clamp(-0.65, 0.65)

        return thrusts, alphas, betas


class VirgoEnv(DirectRLEnv):
    """
    Direct RL Environment for Virgo_RL (12-DOF MorphQuad) in Isaac Lab
    """
    cfg: VirgoEnvCfg

    def __init__(self, cfg: VirgoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.kinematics = BatchedVirgoKinematics(
            arm_l=self.cfg.arm_l,
            arm_b=self.cfg.arm_b,
            max_thrust_per_arm=self.cfg.max_thrust_per_arm,
            device=self.device
        )

        # Body & joint handles
        body_id, _ = self._robot.find_bodies("BODY")
        self._body_id = body_id[0]
        self._shaft_body_ids, _ = self._robot.find_bodies(["SHAFT_MOTOR_1", "SHAFT_MOTOR_2", "SHAFT_MOTOR_3", "SHAFT_MOTOR_4"])
        self._all_thrust_body_ids = torch.tensor(body_id + self._shaft_body_ids, dtype=torch.int, device=self.device)
        
        # 8 Servo joint indices (Dynamixel XC330 outer gimbal, XL330 inner gimbal)
        servo_names = [f"ARM_MOTOR_{i}" for i in range(1, 5)] + [f"SHAFT_MOTOR_{i}" for i in range(1, 5)]
        self._servo_joint_ids, _ = self._robot.find_joints(servo_names)

        # 8 Propeller joint indices (coaxial counter-rotating blades)
        prop_names = [f"PROP_{i}_{j}" for i in range(1, 5) for j in (1, 2)]
        self._prop_joint_ids, _ = self._robot.find_joints(prop_names)

        # Robot physical properties (Exact MorphQuad takeoff mass from Tzoumas Lab arXiv:2607.02764)
        self._robot_mass = 4.56 # kg
        self._gravity_mag = 9.81
        self._robot_weight = self._robot_mass * self._gravity_mag # 44.7336 N

        # 8 Coaxial Motors & 8 Servos tracking
        # Each arm has 2 counter-rotating coaxial motors:
        # Arm 1: [PROP_1_1, PROP_1_2], Arm 2: [PROP_2_1, PROP_2_2], Arm 3: [PROP_3_1, PROP_3_2], Arm 4: [PROP_4_1, PROP_4_2]
        self._current_motor_thrusts = torch.zeros(self.num_envs, 8, device=self.device)
        self._current_thrusts = torch.zeros(self.num_envs, 4, device=self.device) # Sum per arm
        self._current_alphas = torch.zeros(self.num_envs, 4, device=self.device)
        self._current_betas = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_alphas = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_betas = torch.zeros(self.num_envs, 4, device=self.device)
        self._servo_rate_alpha = torch.zeros(self.num_envs, 4, device=self.device)
        self._servo_rate_beta = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_servo_rate_alpha = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_servo_rate_beta = torch.zeros(self.num_envs, 4, device=self.device)
        self._servo_jerk_alpha = torch.zeros(self.num_envs, 4, device=self.device)
        self._servo_jerk_beta = torch.zeros(self.num_envs, 4, device=self.device)
        self._prop_velocities = torch.zeros(self.num_envs, 8, device=self.device)

        # Environmental Disturbances (Beaufort Scale 1-6 Wind + Force/Torque Impulses)
        self._wind_w = torch.zeros(self.num_envs, 3, device=self.device)      # 3D wind velocity vector in m/s
        self._ext_force_w = torch.zeros(self.num_envs, 3, device=self.device) # External disturbance force in N
        self._ext_torque_b = torch.zeros(self.num_envs, 3, device=self.device)# External disturbance torque in Nm
        self._pos_err_integral = torch.zeros(self.num_envs, 3, device=self.device) # Integral buffer for wind/disturbance compensation

        # Multi-body distributed force/torque buffers: [num_envs, 5 bodies, 3] (0: BODY disturbance, 1..4: SHAFT_MOTOR_1..4 thrust)
        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._multibody_forces = torch.zeros(self.num_envs, 5, 3, device=self.device)
        self._multibody_torques = torch.zeros(self.num_envs, 5, 3, device=self.device)

        # 6-DOF Target buffers
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_acc_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._current_acc_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_quat_w[:, 0] = 1.0 # default upright
        self._psi_des = torch.zeros(self.num_envs, device=self.device)
        self._desired_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self._lookahead_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._traj_time = torch.zeros(self.num_envs, device=self.device)
        self._d_servo_pos = torch.zeros(self.num_envs, device=self.device)
        self._d_servo_vel = torch.zeros(self.num_envs, device=self.device)

        # Logging metrics
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["r_pos", "r_att", "pos_error", "att_error_deg"]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Optional TiledCamera sensor for visual snapshots
        if self.cfg.enable_camera and self.cfg.camera is not None:
            self._camera = TiledCamera(self.cfg.camera)
            self.scene.sensors["camera"] = self._camera
        else:
            self._camera = None

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Dome lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def capture_snapshot(self, output_path: str):
        """Captures an RGB image from the camera and saves to disk."""
        if self._camera is not None:
            self._camera.update(self.step_dt)
            rgb_data = self._camera.data.output["rgb"][0] # [H, W, 3] or [H, W, 4]
            rgb_tensor = rgb_data[..., :3].permute(2, 0, 1).cpu()
            if rgb_tensor.dtype != torch.uint8:
                rgb_tensor = rgb_tensor.to(torch.uint8)
            import torchvision.io
            torchvision.io.write_png(rgb_tensor, output_path)
            return True
        return False

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Update Large-Scale 3D Figure-8 (Lemniscate) Target smoothly based on trajectory time
        self._traj_time += self.step_dt
        tau = self._traj_time
        T_period = 20.0 # 20 seconds period provides ample runway & gentle, natural cornering
        omega = 2.0 * math.pi / T_period
        omega_z = 2.0 * omega

        origins = self._terrain.env_origins # [num_envs, 3]
        A_x = 4.0  # Width span = 8.0 m
        A_y = 1.6  # Length span = 3.2 m
        A_z = 0.4  # Altitude amplitude
        z_0 = 2.0  # Base hover altitude = 2.0 m

        self._desired_pos_w[:, 0] = origins[:, 0] + A_x * torch.sin(omega * tau)
        self._desired_pos_w[:, 1] = origins[:, 1] + (A_y * 0.5) * torch.sin(2.0 * omega * tau)
        self._desired_pos_w[:, 2] = origins[:, 2] + z_0 + A_z * torch.sin(omega_z * tau)

        self._desired_vel_w[:, 0] = A_x * omega * torch.cos(omega * tau)
        self._desired_vel_w[:, 1] = A_y * omega * torch.cos(2.0 * omega * tau)
        self._desired_vel_w[:, 2] = A_z * omega_z * torch.cos(omega_z * tau)

        # Desired Heading: Align nose with flight trajectory velocity tangent
        self._psi_des = torch.atan2(self._desired_vel_w[:, 1], self._desired_vel_w[:, 0]) # [num_envs]
        sp = torch.sin(self._psi_des * 0.5)
        cp = torch.cos(self._psi_des * 0.5)
        self._desired_quat_w[:, 0] = cp
        self._desired_quat_w[:, 1] = 0.0
        self._desired_quat_w[:, 2] = 0.0
        self._desired_quat_w[:, 3] = sp

        # Analytical Desired Acceleration for Centripetal Cornering Feedforward
        self._desired_acc_w[:, 0] = -A_x * (omega**2) * torch.sin(omega * tau)
        self._desired_acc_w[:, 1] = -2.0 * A_y * (omega**2) * torch.sin(2.0 * omega * tau)
        self._desired_acc_w[:, 2] = -A_z * (omega_z**2) * torch.sin(omega_z * tau)

        # Analytical Feedforward Yaw Angular Velocity for Figure-8 Cornering
        vx = self._desired_vel_w[:, 0]
        vy = self._desired_vel_w[:, 1]
        ax = self._desired_acc_w[:, 0]
        ay = self._desired_acc_w[:, 1]
        v_xy_sq = torch.clamp(vx**2 + vy**2, min=1e-4)
        self._desired_yaw_rate = torch.clamp((vx * ay - vy * ax) / v_xy_sq, -1.5, 1.5)

        # Future Lookahead Waypoint (tau + 0.6s) to allow anticipation of turns
        tau_look = tau + 0.6
        self._lookahead_pos_w[:, 0] = origins[:, 0] + A_x * torch.sin(omega * tau_look)
        self._lookahead_pos_w[:, 1] = origins[:, 1] + (A_y * 0.5) * torch.sin(2.0 * omega * tau_look)
        self._lookahead_pos_w[:, 2] = origins[:, 2] + z_0 + A_z * torch.sin(omega_z * tau_look)

        # Estimate actual linear acceleration: a_t = (v_t - v_{t-1}) / dt
        cur_vel_w = self._robot.data.root_lin_vel_w
        self._current_acc_w = (cur_vel_w - self._prev_lin_vel_w) / self.step_dt
        self._prev_lin_vel_w = cur_vel_w.clone()

        q_w = self._robot.data.root_quat_w # [B, 4]
        pos_w = self._robot.data.root_pos_w # [B, 3]

        # 1. Unified World-Frame Force Synthesis
        pos_err_w = self._desired_pos_w - pos_w
        vel_err_w = self._desired_vel_w - cur_vel_w

        self._pos_err_integral += pos_err_w * self.step_dt
        self._pos_err_integral = torch.clamp(self._pos_err_integral, -2.5, 2.5)

        f_cmd_w = torch.zeros(self.num_envs, 3, device=self.device)
        f_cmd_w[:, 0:2] = torch.clamp(
            self.cfg.nominal_mass * self._desired_acc_w[:, 0:2]
            + 10.0 * pos_err_w[:, 0:2]
            + 4.5 * vel_err_w[:, 0:2]
            + 3.5 * self._pos_err_integral[:, 0:2]
            + self._actions[:, 0:2] * 5.0,
            -32.0, 32.0
        )
        z_err = self._desired_pos_w[:, 2] - pos_w[:, 2]
        vz_err = self._desired_vel_w[:, 2] - cur_vel_w[:, 2]
        f_cmd_w[:, 2] = (
            self._robot_weight
            + 18.0 * z_err
            + 6.0 * vz_err
            + self._actions[:, 2] * (self._robot_weight * 0.25)
        )

        # Rotate total commanded force cleanly into Body Frame
        f_cmd_b = quat_rotate_inverse(q_w, f_cmd_w)

        # 2. Desired Torque in Body Frame
        omega_b = self._robot.data.root_ang_vel_b

        # Global Singularity-Free Lyapunov Leveling
        z_w = torch.zeros(self.num_envs, 3, device=self.device)
        z_w[:, 2] = 1.0
        z_b = quat_rotate_inverse(q_w, z_w)
        tau_level_x = -8.0 * z_b[:, 1] - 1.5 * omega_b[:, 0]
        tau_level_y = 8.0 * z_b[:, 0] - 1.5 * omega_b[:, 1]

        # Tangent Yaw Tracking
        x_unit = torch.zeros(self.num_envs, 3, device=self.device)
        x_unit[:, 0] = 1.0
        x_body_w = quat_rotate(q_w, x_unit)
        psi_cur = torch.atan2(x_body_w[:, 1], x_body_w[:, 0])
        yaw_err = torch.atan2(torch.sin(self._psi_des - psi_cur), torch.cos(self._psi_des - psi_cur))
        yaw_rate_err = self._desired_yaw_rate - omega_b[:, 2]
        tau_level_z = 4.0 * yaw_err + 1.0 * yaw_rate_err

        tau_cmd_b = torch.zeros(self.num_envs, 3, device=self.device)
        tau_cmd_b[:, 0] = torch.clamp(tau_level_x + self._actions[:, 3] * 1.5, -self.cfg.max_torque, self.cfg.max_torque)
        tau_cmd_b[:, 1] = torch.clamp(tau_level_y + self._actions[:, 4] * 1.5, -self.cfg.max_torque, self.cfg.max_torque)
        tau_cmd_b[:, 2] = torch.clamp(tau_level_z + self._actions[:, 5] * 2.0, -self.cfg.max_torque * 0.5, self.cfg.max_torque * 0.5)

        # 3. Batched Null-Space Thrust Allocation for 4 Arms
        thrusts, alphas_raw, betas_raw = self.kinematics.allocate(f_cmd_b, tau_cmd_b)

        # 1st-Order Low-Pass Filter (EMA) on Servo Angles for High "Độ Đầm" and Smoothness
        # Cutoff frequency ~ 15 Hz matches physical Dynamixel XC330 / XL330 mechanical bandwidth
        gamma_filter = 0.65
        alphas = (1.0 - gamma_filter) * self._prev_alphas + gamma_filter * alphas_raw
        betas = (1.0 - gamma_filter) * self._prev_betas + gamma_filter * betas_raw

        # Dynamic Vertical Decoupling:
        cos_beta = torch.cos(betas)
        sin_beta = torch.sin(betas)
        cos_alpha = torch.cos(alphas)
        sin_alpha = torch.sin(alphas)
        z_delivered = torch.sum(thrusts * cos_alpha * cos_beta, dim=-1, keepdim=True).clamp(min=1.0)
        thrust_scale = torch.clamp(f_cmd_b[:, 2:3] / z_delivered, min=0.3, max=1.8)
        thrusts = torch.clamp(thrusts * thrust_scale, max=self.cfg.max_thrust_per_arm)

        # Measure Step-to-Step Servo Displacement & Acceleration for "Độ Đầm" Penalties
        self._d_servo_pos = torch.sum((alphas - self._prev_alphas)**2 + (betas - self._prev_betas)**2, dim=-1)
        cur_rate_alpha = (alphas - self._prev_alphas) / self.step_dt
        cur_rate_beta = (betas - self._prev_betas) / self.step_dt
        self._d_servo_vel = torch.sum((cur_rate_alpha - self._prev_servo_rate_alpha)**2 + (cur_rate_beta - self._prev_servo_rate_beta)**2, dim=-1)

        self._prev_servo_rate_alpha = cur_rate_alpha.clone()
        self._prev_servo_rate_beta = cur_rate_beta.clone()
        self._servo_rate_alpha = cur_rate_alpha
        self._servo_rate_beta = cur_rate_beta
        self._prev_alphas = alphas.clone()
        self._prev_betas = betas.clone()

        self._current_thrusts = thrusts
        self._current_alphas = alphas
        self._current_betas = betas

        # 4. Map Arm Thrusts to 8 Coaxial Counter-Rotating Motors (PROP_1_1..4_2)
        for i in range(4):
            t_half = thrusts[:, i] * 0.5
            self._current_motor_thrusts[:, 2 * i] = t_half
            self._current_motor_thrusts[:, 2 * i + 1] = t_half

            omega_prop = 180.0 + 45.0 * torch.sqrt(torch.clamp(t_half, min=0.0))
            self._prop_velocities[:, 2 * i] = omega_prop
            self._prop_velocities[:, 2 * i + 1] = -omega_prop

        # 5. Actuate Joints in PhysX:
        servo_targets = torch.cat([alphas, betas], dim=-1) # [B, 8]
        self._robot.set_joint_position_target(servo_targets, joint_ids=self._servo_joint_ids)
        self._robot.set_joint_velocity_target(self._prop_velocities, joint_ids=self._prop_joint_ids)

        # 6. Delivered Aerodynamic Wrench in Body Frame:
        t_bx = thrusts * sin_alpha * cos_beta
        t_by = -thrusts * sin_beta
        t_bz = thrusts * cos_alpha * cos_beta

        f_aero_b = torch.stack([t_bx.sum(dim=-1), t_by.sum(dim=-1), t_bz.sum(dim=-1)], dim=-1)
        tau_b = torch.zeros(self.num_envs, 3, device=self.device)
        positions = torch.tensor([
            [ 0.19899, -0.19780, 0.01664],
            [-0.19899,  0.19680, 0.01665],
            [ 0.19899,  0.19742, 0.01665],
            [-0.19899, -0.19742, 0.01664]
        ], device=self.device)
        for i in range(4):
            t_i = torch.stack([t_bx[:, i], t_by[:, i], t_bz[:, i]], dim=-1)
            tau_b += torch.cross(positions[i].unsqueeze(0).expand(self.num_envs, -1), t_i, dim=-1)

        # 7. Environmental Disturbances (Beaufort Wind + External Push in Body Frame):
        v_quad_w = self._robot.data.root_lin_vel_w
        v_rel_w = v_quad_w - self._wind_w
        v_rel_speed = torch.norm(v_rel_w, dim=-1, keepdim=True)
        f_wind_w = -0.5 * 1.225 * 0.09 * v_rel_speed * v_rel_w - 0.35 * v_rel_w
        f_dist_w = f_wind_w + self._ext_force_w
        f_dist_b = quat_rotate_inverse(q_w, f_dist_w)
        tau_dist_b = self._ext_torque_b

        # Apply Total Wrench directly to Body Frame (index 0)
        self._multibody_forces[:, 0, :] = f_aero_b + f_dist_b
        self._multibody_torques[:, 0, :] = tau_b + tau_dist_b
        self._multibody_forces[:, 1:5, :] = 0.0
        self._multibody_torques[:, 1:5, :] = 0.0

    def _apply_action(self):
        # Apply total multi-body wrenches directly in Link (Body) Frame: is_global=False
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._all_thrust_body_ids,
            forces=self._multibody_forces,
            torques=self._multibody_torques,
            is_global=False
        )

    def _get_observations(self) -> dict:
        q_w = self._robot.data.root_quat_w
        pos_w = self._robot.data.root_pos_w
        vel_w = self._robot.data.root_lin_vel_w
        omega_b = self._robot.data.root_ang_vel_b

        # 1. Target vector from robot to desired position in body frame
        pos_err_w = self._desired_pos_w - pos_w
        pos_err_b = quat_rotate_inverse(q_w, pos_err_w)

        # 2. Velocity in body frame
        vel_b = quat_rotate_inverse(q_w, vel_w)

        # 3. Target velocity in body frame
        target_vel_b = quat_rotate_inverse(q_w, self._desired_vel_w)

        # 4. Quaternion error: q_err = q_target^-1 * q
        q_target_inv = quat_conjugate(self._desired_quat_w)
        q_err = quat_mul(q_target_inv, q_w)
        # Canonical qw >= 0
        neg_mask = (q_err[:, 0] < 0).unsqueeze(-1)
        q_err = torch.where(neg_mask, -q_err, q_err)

        # 5. Normalized actuator states
        norm_t = self._current_thrusts / self.cfg.max_thrust_per_arm
        norm_a = self._current_alphas / math.pi
        norm_b = self._current_betas / (0.5 * math.pi)

        # 6. Desired Centripetal Acceleration in body frame
        target_acc_b = quat_rotate_inverse(q_w, self._desired_acc_w)

        # 7. Future Lookahead Vector from robot to lookahead waypoint in body frame
        lookahead_err_w = self._lookahead_pos_w - pos_w
        lookahead_err_b = quat_rotate_inverse(q_w, lookahead_err_w)

        # 8. Trajectory Periodic Phase Progress [-1, 1]
        phase_feat = torch.sin((2.0 * math.pi / 20.0) * self._traj_time).unsqueeze(-1)

        # 35-Dim Observation Vector (100% Invariant to World Environment Origins):
        obs = torch.cat([
            pos_err_b,                  # [3]
            vel_b,                      # [3]
            omega_b,                    # [3]
            target_vel_b,               # [3]
            q_err,                      # [4]
            norm_t,                     # [4]
            norm_a,                     # [4]
            norm_b,                     # [4]
            target_acc_b,               # [3]
            lookahead_err_b,            # [3]
            phase_feat                  # [1]
        ], dim=-1) # Total = 35

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_err = torch.norm(self._robot.data.root_pos_w - self._desired_pos_w, dim=-1)
        pos_err_xy = torch.norm(self._robot.data.root_pos_w[:, 0:2] - self._desired_pos_w[:, 0:2], dim=-1)
        pos_err_z = torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2])
        # Dual-Plane Restoring: Linear gradient (-4.0 XY, -6.0 Z) forcefully pulls UAV back to course,
        # with strong exponential rewards (+6.0 XY, +4.0 Z) within 20cm tracking accuracy
        r_pos = (-4.0 * pos_err_xy + 6.0 * torch.exp(-pos_err_xy**2 / 0.18)) + (-6.0 * pos_err_z + 4.0 * torch.exp(-pos_err_z**2 / 0.08))

        # Decoupled Attitude Control:
        # 1. Levelness (Keep Roll=0, Pitch=0 rock solid):
        q_w = self._robot.data.root_quat_w
        z_unit = torch.zeros(self.num_envs, 3, device=self.device)
        z_unit[:, 2] = 1.0
        z_body_w = quat_rotate(q_w, z_unit) # [B, 3]
        level_err = z_body_w[:, 0]**2 + z_body_w[:, 1]**2 # 0 when perfectly level
        r_level = -8.0 * level_err + 3.0 * torch.exp(-20.0 * level_err)

        # 2. Tangent Yaw Alignment: Rotate nose to face trajectory velocity tangent
        x_unit = torch.zeros(self.num_envs, 3, device=self.device)
        x_unit[:, 0] = 1.0
        x_body_w = quat_rotate(q_w, x_unit) # [B, 3]
        cos_yaw_err = x_body_w[:, 0] * torch.cos(self._psi_des) + x_body_w[:, 1] * torch.sin(self._psi_des)
        r_yaw = 3.0 * cos_yaw_err + 2.0 * torch.exp(-6.0 * torch.clamp(1.0 - cos_yaw_err, min=0.0))

        # 3. Tangent Yaw Angular Rate Feedforward: Policy actively matches turn rate at corner lobes
        omega_b = self._robot.data.root_ang_vel_b
        yaw_rate_err = torch.abs(omega_b[:, 2] - self._desired_yaw_rate)
        r_yaw_rate = torch.exp(-3.0 * yaw_rate_err)

        # 4. Energy Optimization in Normal Flight:
        # Penalize non-zero servo tilt when calm, saving battery and keeping actuators resting upright
        servo_deflection = torch.sum(self._current_alphas**2 + self._current_betas**2, dim=-1)
        r_energy = -0.05 * servo_deflection

        # 5. Feedforward Velocity Tracking (forces policy to match figure-8 speed vector)
        vel_err = torch.norm(self._robot.data.root_lin_vel_w - self._desired_vel_w, dim=-1)
        r_vel = -1.5 * vel_err + 3.0 * torch.exp(-vel_err**2 / 0.5)

        # 6. Centripetal Acceleration Tracking (eliminates centrifugal drift at curve lobes)
        acc_err = torch.norm(self._current_acc_w - self._desired_acc_w, dim=-1)
        r_acc = -0.3 * acc_err + 2.0 * torch.exp(-acc_err**2 / 0.8)

        omega_mag = torch.norm(self._robot.data.root_ang_vel_b, dim=-1)
        r_omega = -0.1 * omega_mag + 1.0 * torch.exp(-1.0 * omega_mag)

        action_diff = torch.norm(self._actions - self._prev_actions, dim=-1)
        r_smooth = action_diff

        # 7. 8 Dynamixel Servos "Độ Đầm" Penalties: Direct Step-to-Step Displacement & Rate Penalties
        d_servo_pos = self._d_servo_pos
        d_servo_vel = self._d_servo_vel

        # Quiescence / Steady-state holding bonus when tracking error is small
        steady_mask = (pos_err < 0.25).float()
        r_quiescence = steady_mask * torch.exp(-3.0 * torch.sqrt(d_servo_pos + 1e-6))

        # 8. Gimbal lock avoidance: penalize beta exceeding 75 deg
        beta_limit_rad = 75.0 * math.pi / 180.0
        gimbal_violation = torch.sum(torch.clamp(torch.abs(self._current_betas) - beta_limit_rad, min=0.0)**2, dim=-1)

        reward = (
            self.cfg.r_pos_scale * r_pos +
            self.cfg.r_att_scale * r_level +
            2.5 * r_yaw +
            1.5 * r_yaw_rate +
            1.0 * r_energy +
            1.5 * r_vel +
            0.8 * r_acc +
            0.3 * r_omega +
            self.cfg.r_smooth_scale * r_smooth +
            self.cfg.r_servo_vel_scale * d_servo_pos +
            self.cfg.r_servo_jerk_scale * d_servo_vel +
            self.cfg.r_quiescence_scale * r_quiescence +
            self.cfg.r_gimbal_scale * gimbal_violation
        ) * self.step_dt

        # Logging
        self._episode_sums["r_pos"] += r_pos
        self._episode_sums["r_att"] += r_level
        self._episode_sums["pos_error"] += pos_err
        self._episode_sums["att_error_deg"] += (level_err * 180.0 / math.pi)

        self._prev_actions = self._actions.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_z = self._robot.data.root_pos_w[:, 2]
        dist_to_target = torch.norm(self._robot.data.root_pos_w - self._desired_pos_w, dim=-1)

        crashed = pos_z < 0.2
        # In evaluation mode: never prematurely abort/reset, let the drone complete the entire flight!
        if getattr(self.cfg, "is_eval", False):
            diverged = torch.zeros_like(crashed)
            time_out = torch.zeros_like(time_out)
        else:
            too_high = pos_z > 4.5
            diverged = torch.logical_or(dist_to_target > 4.5, too_high)
        died = torch.logical_or(crashed, diverged)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # Uniform Trajectory Phase Distribution on Large 3D Figure-8
        T_period = 20.0
        omega = 2.0 * math.pi / T_period
        omega_z = 2.0 * omega
        A_x, A_y, A_z, z_0 = 4.0, 1.6, 0.4, 2.0

        if getattr(self.cfg, "is_eval", False):
            tau_0 = torch.zeros(num_resets, device=self.device)
        else:
            # Random phase tau_0 across [0, T_period] ensures all curve lobes, turns & crosses are trained concurrently
            tau_0 = torch.empty(num_resets, device=self.device).uniform_(0.0, T_period)

        self._traj_time[env_ids] = tau_0

        origins = self._terrain.env_origins[env_ids]
        px = origins[:, 0] + A_x * torch.sin(omega * tau_0)
        py = origins[:, 1] + (A_y * 0.5) * torch.sin(2.0 * omega * tau_0)
        pz = origins[:, 2] + z_0 + A_z * torch.sin(omega_z * tau_0)
        self._desired_pos_w[env_ids, 0] = px
        self._desired_pos_w[env_ids, 1] = py
        self._desired_pos_w[env_ids, 2] = pz

        vx = A_x * omega * torch.cos(omega * tau_0)
        vy = A_y * omega * torch.cos(2.0 * omega * tau_0)
        vz = A_z * omega_z * torch.cos(omega_z * tau_0)
        self._desired_vel_w[env_ids, 0] = vx
        self._desired_vel_w[env_ids, 1] = vy
        self._desired_vel_w[env_ids, 2] = vz

        psi_des = torch.atan2(vy, vx)
        self._psi_des[env_ids] = psi_des
        sp = torch.sin(psi_des * 0.5)
        cp = torch.cos(psi_des * 0.5)
        self._desired_quat_w[env_ids, 0] = cp
        self._desired_quat_w[env_ids, 1] = 0.0
        self._desired_quat_w[env_ids, 2] = 0.0
        self._desired_quat_w[env_ids, 3] = sp

        # Analytical Desired Acceleration for Centripetal Feedforward
        self._desired_acc_w[env_ids, 0] = -A_x * (omega**2) * torch.sin(omega * tau_0)
        self._desired_acc_w[env_ids, 1] = -2.0 * A_y * (omega**2) * torch.sin(2.0 * omega * tau_0)
        self._desired_acc_w[env_ids, 2] = -A_z * (omega_z**2) * torch.sin(omega_z * tau_0)

        # Analytical Feedforward Yaw Angular Velocity
        ax = self._desired_acc_w[env_ids, 0]
        ay = self._desired_acc_w[env_ids, 1]
        v_xy_sq = torch.clamp(vx**2 + vy**2, min=1e-4)
        self._desired_yaw_rate[env_ids] = (vx * ay - vy * ax) / v_xy_sq

        # Future Lookahead Waypoint (tau_0 + 0.6s)
        tau_look = tau_0 + 0.6
        self._lookahead_pos_w[env_ids, 0] = origins[:, 0] + A_x * torch.sin(omega * tau_look)
        self._lookahead_pos_w[env_ids, 1] = origins[:, 1] + (A_y * 0.5) * torch.sin(2.0 * omega * tau_look)
        self._lookahead_pos_w[env_ids, 2] = origins[:, 2] + z_0 + A_z * torch.sin(omega_z * tau_look)

        self._pos_err_integral[env_ids] = 0.0

        if getattr(self.cfg, "is_eval", False):
            self._wind_w[env_ids] = 0.0
            self._ext_force_w[env_ids] = 0.0
            self._ext_torque_b[env_ids] = 0.0
            pos_noise = torch.zeros(num_resets, 3, device=self.device)
            vel_noise = torch.zeros(num_resets, 3, device=self.device)
        else:
            # Sample Wind Conditions (Beaufort Scale 1 to 6: 0.5 m/s up to 13.8 m/s, ~50 km/h)
            wind_speed = torch.empty(num_resets, device=self.device).uniform_(0.5, 13.8)
            wind_azimuth = torch.empty(num_resets, device=self.device).uniform_(0.0, 2.0 * math.pi)
            wind_elevation = torch.empty(num_resets, device=self.device).uniform_(-math.radians(15.0), math.radians(15.0))
            self._wind_w[env_ids, 0] = wind_speed * torch.cos(wind_elevation) * torch.cos(wind_azimuth)
            self._wind_w[env_ids, 1] = wind_speed * torch.cos(wind_elevation) * torch.sin(wind_azimuth)
            self._wind_w[env_ids, 2] = wind_speed * torch.sin(wind_elevation)

            # External Disturbance Force & Torque Impulses
            self._ext_force_w[env_ids] = torch.empty(num_resets, 3, device=self.device).uniform_(-5.0, 5.0)
            self._ext_torque_b[env_ids] = torch.empty(num_resets, 3, device=self.device).uniform_(-0.05, 0.05)
            # Wide domain randomization at reset enables learning strong restoring gradient across all lobes
            pos_noise = torch.empty(num_resets, 3, device=self.device).uniform_(-0.5, 0.5)
            vel_noise = torch.empty(num_resets, 3, device=self.device).uniform_(-0.3, 0.3)

        # Spawn Robot Root Pose near target with matching forward flight velocity
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] = self._desired_pos_w[env_ids] + pos_noise
        default_root_state[:, 3:7] = self._desired_quat_w[env_ids]
        default_root_state[:, 7:10] = self._desired_vel_w[env_ids] + vel_noise # Smooth airborne launch matching trajectory velocity
        default_root_state[:, 10:13] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        # Pre-spin coaxial counter-rotating propellers upon spawn
        for i in range(4):
            joint_vel[:, self._prop_joint_ids[2 * i]] = 250.0
            joint_vel[:, self._prop_joint_ids[2 * i + 1]] = -250.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_alphas[env_ids] = 0.0
        self._prev_betas[env_ids] = 0.0
        self._servo_rate_alpha[env_ids] = 0.0
        self._servo_rate_beta[env_ids] = 0.0
        self._prev_servo_rate_alpha[env_ids] = 0.0
        self._prev_servo_rate_beta[env_ids] = 0.0
        self._servo_jerk_alpha[env_ids] = 0.0
        self._servo_jerk_beta[env_ids] = 0.0
        self._d_servo_pos[env_ids] = 0.0
        self._d_servo_vel[env_ids] = 0.0
        self._prev_lin_vel_w[env_ids] = 0.0
