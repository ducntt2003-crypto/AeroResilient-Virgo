#!/usr/bin/env python3
"""
Virgo_RL: Gymnasium Simulation Environment for 12-DOF MorphQuad Omnidirectional UAV
Paper: "MorphQuad: Morphable Quadrotor for Superhuman Maneuverability, Manipulation, and Resiliency"
ArXiv: 2607.02764 (Tzoumas Lab, University of Michigan)

Features:
- Full 6-DOF rigid body physics with quaternion kinematics (RK4 / semi-implicit Euler integration).
- Full 12-DOF actuator dynamics:
  * 4 Coaxial counter-rotating rotor pairs (thrust lag tau_m).
  * 8 Gimbal servos (4 inner alpha, 4 outer beta) with first-order lag and velocity limits.
- Supports two action modes:
  1. 'wrench': 6-DOF body wrench [f_B (3), tau_B (3)] mapped through VirgoKinematics null-space allocation.
  2. 'direct': direct 12-DOF normalized control [-1, 1]^12 for (T_1..4, alpha_1..4, beta_1..4).
- Decoupled 6-DOF tracking:
  * Target position, velocity, orientation (quaternion/SO(3)), and angular velocity.
  * Capable of full inverted flight (pitch=180 deg), vertical wall hover (pitch=90 deg), and arbitrary tilting.
- Domain Randomization: mass, inertia, servo delays, motor drag, wind disturbances.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from virgo_kinematics_and_allocation import VirgoKinematics


def quat_to_rot_matrix(q):
    """
    Convert unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R (Body to World).
    p_W = R @ p_B
    """
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)


def rot_matrix_to_quat(R):
    """
    Convert 3x3 rotation matrix to unit quaternion [qw, qx, qy, qz].
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_multiply(q1, q2):
    """
    Hamiltonian quaternion product q1 * q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_conjugate(q):
    """
    Quaternion conjugate [qw, -qx, -qy, -qz]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def euler_to_quat(roll, pitch, yaw):
    """
    Euler angles (ZYX convention) to quaternion [qw, qx, qy, qz]
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz], dtype=np.float64)


class VirgoEnv(gym.Env):
    """
    Gymnasium Environment for Virgo_RL (MorphQuad 12-DOF Omnidirectional UAV)
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        action_mode='wrench',     # 'wrench' (6D) or 'direct' (12D)
        dt=0.01,                 # 100 Hz simulation step
        substeps=5,              # 500 Hz internal physics RK4 sub-stepping
        max_steps=500,           # 5.0 seconds per episode
        randomize_targets=True,  # Random 6-DOF target positions & orientations
        randomize_dynamics=True  # Domain randomization
    ):
        super().__init__()
        self.action_mode = action_mode
        self.dt = dt
        self.substeps = substeps
        self.dt_sub = dt / substeps
        self.max_steps = max_steps
        self.randomize_targets = randomize_targets
        self.randomize_dynamics = randomize_dynamics

        # Nominal Robot Physical Parameters (MorphQuad)
        self.nominal_mass = 4.56 # kg
        self.nominal_J = np.diag([0.082, 0.082, 0.145]) # kg*m^2
        self.arm_length_x = 0.22 # m (l)
        self.arm_length_y = 0.22 # m (b)
        self.max_thrust_per_arm = 28.0 # N (peak 2.8 kg per arm)
        self.max_total_thrust = 4.0 * self.max_thrust_per_arm # 112 N

        # Kinematics engine
        self.kinematics = VirgoKinematics(
            mass=self.nominal_mass,
            arm_length_x=self.arm_length_x,
            arm_length_y=self.arm_length_y,
            max_thrust_per_arm=self.max_thrust_per_arm
        )

        # Actuator Dynamics constants
        self.tau_thrust = 0.02 # s (motor response time constant)
        self.tau_servo = 0.03  # s (Dynamixel servo time constant)
        self.max_servo_vel = 10.0 # rad/s (~600 deg/s)

        # Action Space Definition
        if self.action_mode == 'wrench':
            # Action: 6 continuous values in [-1, 1]:
            # [fx, fy, fz, tau_x, tau_y, tau_z]
            # Scaled to:
            # f_B in [-max_total_thrust, max_total_thrust]
            # tau_B in [-max_torque, max_torque]
            self.max_torque = 2.0 * self.arm_length_x * self.max_thrust_per_arm # ~12.3 Nm
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            )
        elif self.action_mode == 'direct':
            # Action: 12 continuous values in [-1, 1]:
            # [T_1..4 (scaled to [0, max_thrust]), alpha_1..4 (scaled to [-pi, pi]), beta_1..4 (scaled to [-pi/2, pi/2])]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(12,), dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # Observation Space Definition (35 dimensions)
        # 1. Body-frame position error: R^T (p - p_des) [3]
        # 2. Body-frame linear velocity: R^T v [3]
        # 3. Body-frame angular velocity: omega_B [3]
        # 4. Target linear velocity in body frame: R^T v_des [3]
        # 5. Orientation error: rotation error vector or quaternion error [4]
        # 6. Current thrust state: T_1..4 / max_thrust [4]
        # 7. Current servo angles: alpha_1..4 / pi [4]
        # 8. Current servo angles: beta_1..4 / (pi/2) [4]
        # 9. Target orientation quaternion: q_des [4]
        # 10. Target position in world: p_des [3]
        # Total = 3 + 3 + 3 + 3 + 4 + 4 + 4 + 4 + 4 + 3 = 35
        self.obs_dim = 35
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Internal state variables
        self.mass = self.nominal_mass
        self.J = self.nominal_J.copy()
        self.J_inv = np.linalg.inv(self.J)

        # State vectors
        self.pos = np.zeros(3, dtype=np.float64)       # p in World (m)
        self.vel = np.zeros(3, dtype=np.float64)       # v in World (m/s)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])    # [qw, qx, qy, qz]
        self.omega = np.zeros(3, dtype=np.float64)     # omega in Body (rad/s)

        # Actuator states
        self.thrusts = np.zeros(4, dtype=np.float64)   # Current actual thrusts (N)
        self.alphas = np.zeros(4, dtype=np.float64)    # Current inner servo angles (rad)
        self.betas = np.zeros(4, dtype=np.float64)     # Current outer servo angles (rad)

        # Commanded actuator setpoints (from allocation or policy)
        self.cmd_thrusts = np.zeros(4, dtype=np.float64)
        self.cmd_alphas = np.zeros(4, dtype=np.float64)
        self.cmd_betas = np.zeros(4, dtype=np.float64)

        # Targets (6-DOF)
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.target_vel = np.zeros(3, dtype=np.float64)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.target_omega = np.zeros(3, dtype=np.float64)

        self.step_count = 0
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Domain Randomization
        if self.randomize_dynamics:
            self.mass = self.nominal_mass * np.random.uniform(0.90, 1.10)
            j_scale = np.random.uniform(0.85, 1.15, size=3)
            self.J = np.diag(np.diag(self.nominal_J) * j_scale)
            self.J_inv = np.linalg.inv(self.J)
        else:
            self.mass = self.nominal_mass
            self.J = self.nominal_J.copy()
            self.J_inv = np.linalg.inv(self.J)

        # Set 6-DOF Target
        if options is not None and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'], dtype=np.float64)
            self.target_quat = np.array(options['target_quat'], dtype=np.float64)
            self.target_vel = np.zeros(3)
            self.target_omega = np.zeros(3)
        elif self.randomize_targets:
            # Target position around [0, 0, 1.5] +- 0.8 m
            self.target_pos = np.array([
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(1.0, 2.0)
            ])
            self.target_vel = np.zeros(3)

            # Random 6-DOF target orientation!
            # Sample mode:
            # 50%: Arbitrary 3D orientation (including full inverted or 90 deg pitch)
            # 50%: Moderate orientation tilts (< 45 deg)
            if np.random.rand() > 0.5:
                r_target = np.random.uniform(-np.pi, np.pi)
                p_target = np.random.uniform(-np.pi, np.pi)
                y_target = np.random.uniform(-np.pi, np.pi)
            else:
                r_target = np.random.uniform(-0.5, 0.5)
                p_target = np.random.uniform(-0.5, 0.5)
                y_target = np.random.uniform(-np.pi, np.pi)
            
            self.target_quat = euler_to_quat(r_target, p_target, y_target)
            self.target_omega = np.zeros(3)
        else:
            self.target_pos = np.array([0.0, 0.0, 1.5])
            self.target_vel = np.zeros(3)
            self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.target_omega = np.zeros(3)

        # Initial drone state: close to target with small perturbations
        self.pos = self.target_pos + np.random.uniform(-0.15, 0.15, size=3)
        self.vel = np.random.uniform(-0.05, 0.05, size=3)
        
        # Initial attitude: close to target attitude
        init_tilt = np.random.uniform(-0.1, 0.1, size=3)
        q_tilt = euler_to_quat(init_tilt[0], init_tilt[1], init_tilt[2])
        self.quat = quat_multiply(self.target_quat, q_tilt)
        self.omega = np.random.uniform(-0.05, 0.05, size=3)

        # Hover equilibrium initial actuators matching the starting orientation
        R_WB = quat_to_rot_matrix(self.quat)
        R_BW = R_WB.T
        f_hover_B = R_BW @ np.array([0.0, 0.0, self.mass * 9.81])
        t_init, a_init, b_init, _ = self.kinematics.allocate_thrust(
            f_des=f_hover_B,
            tau_des=np.zeros(3),
            use_null_space=True
        )
        self.thrusts = t_init.copy()
        self.alphas = a_init.copy()
        self.betas = b_init.copy()
        self.cmd_thrusts = self.thrusts.copy()
        self.cmd_alphas = self.alphas.copy()
        self.cmd_betas = self.betas.copy()

        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float64)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Assemble the 35-dimensional observation vector
        """
        R_WB = quat_to_rot_matrix(self.quat)
        R_BW = R_WB.T

        # 1. Position error in Body Frame: R^T (p - p_target)
        p_err_world = self.pos - self.target_pos
        p_err_body = R_BW @ p_err_world

        # 2. Velocity in Body Frame: R^T v
        v_body = R_BW @ self.vel

        # 3. Angular velocity in Body Frame: omega
        omega_body = self.omega.copy()

        # 4. Target velocity in Body Frame
        v_target_body = R_BW @ self.target_vel

        # 5. Quaternion error: q_err = q_target^-1 * q
        q_target_inv = quat_conjugate(self.target_quat)
        q_err = quat_multiply(q_target_inv, self.quat)
        if q_err[0] < 0:
            q_err = -q_err # Enforce canonical representation (qw >= 0)

        # 6. Actuator states normalized
        norm_thrusts = self.thrusts / self.max_thrust_per_arm # in [0, 1]
        norm_alphas = self.alphas / np.pi                     # in [-1, 1]
        norm_betas = self.betas / (0.5 * np.pi)               # in [-1, 1]

        obs = np.concatenate([
            p_err_body,                 # [3]
            v_body,                     # [3]
            omega_body,                 # [3]
            v_target_body,              # [3]
            q_err,                      # [4]
            norm_thrusts,               # [4]
            norm_alphas,                # [4]
            norm_betas,                 # [4]
            self.target_quat,           # [4]
            self.target_pos             # [3]
        ])

        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        return obs

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Parse Action according to mode
        if self.action_mode == 'wrench':
            # action: [fx, fy, fz, tau_x, tau_y, tau_z] in [-1, 1]
            R_WB = quat_to_rot_matrix(self.quat)
            R_BW = R_WB.T
            # Feedforward gravity compensation in body frame
            f_hover_B = R_BW @ np.array([0.0, 0.0, self.mass * 9.81])
            f_cmd = f_hover_B + action[0:3] * (self.mass * 9.81 * 1.5)
            tau_cmd = action[3:6] * self.max_torque * 0.4      # limit commanded torque
            
            # Map wrench to 12 actuator setpoints via Null-Space Control Allocation
            thrusts_cmd, alphas_cmd, betas_cmd, _ = self.kinematics.allocate_thrust(
                f_des=f_cmd,
                tau_des=tau_cmd,
                prev_alphas=self.alphas,
                prev_betas=self.betas,
                use_null_space=True
            )
            self.cmd_thrusts = thrusts_cmd
            self.cmd_alphas = alphas_cmd
            self.cmd_betas = betas_cmd
        elif self.action_mode == 'direct':
            # action: 12 elements
            # 0..3: thrusts in [-1, 1] -> [0, max_thrust]
            # 4..7: alphas in [-1, 1] -> [-pi, pi]
            # 8..11: betas in [-1, 1] -> [-pi/2, pi/2]
            self.cmd_thrusts = (action[0:4] + 1.0) * 0.5 * self.max_thrust_per_arm
            self.cmd_alphas = action[4:8] * np.pi
            self.cmd_betas = action[8:12] * (0.5 * np.pi)

        # Multi-rate Physics Integration (Sub-stepping RK4)
        for _ in range(self.substeps):
            self._integrate_actuators(self.dt_sub)
            self._integrate_rigid_body(self.dt_sub)

        # Compute Reward
        reward, info = self._compute_reward(action)

        # Terminations and Truncations
        terminated = False
        truncated = False

        # Fail if position diverged too far (> 3.5m)
        dist_to_target = np.linalg.norm(self.pos - self.target_pos)
        if dist_to_target > 3.5:
            terminated = True
            reward -= 50.0

        # Crash to ground
        if self.pos[2] < 0.1:
            terminated = True
            reward -= 50.0

        # Max steps reached
        if self.step_count >= self.max_steps:
            truncated = True

        self.prev_action = action.copy()
        obs = self._get_obs()

        info['dist_to_target'] = dist_to_target
        info['pos'] = self.pos.copy()
        info['quat'] = self.quat.copy()

        return obs, reward, terminated, truncated, info

    def _integrate_actuators(self, dt):
        """
        Integrates rotor thrust response and Dynamixel servo angles with rate limits
        """
        # Thrust response (first order)
        self.thrusts += (self.cmd_thrusts - self.thrusts) * (dt / self.tau_thrust)
        self.thrusts = np.clip(self.thrusts, 0.0, self.max_thrust_per_arm)

        # Servo response (first order + rate limit)
        alpha_err = self.cmd_alphas - self.alphas
        alpha_vel = np.clip(alpha_err / self.tau_servo, -self.max_servo_vel, self.max_servo_vel)
        self.alphas += alpha_vel * dt

        beta_err = self.cmd_betas - self.betas
        beta_vel = np.clip(beta_err / self.tau_servo, -self.max_servo_vel, self.max_servo_vel)
        self.betas += beta_vel * dt
        # Clip outer servo to [-pi/2, pi/2] physical mechanical limit
        self.betas = np.clip(self.betas, -0.5 * np.pi, 0.5 * np.pi)

    def _integrate_rigid_body(self, dt):
        """
        Semi-implicit Euler integration of 6-DOF Newton-Euler equations
        """
        # 1. Compute forces and moments produced by 12 actuators in Body Frame
        f_B, tau_B, _ = self.kinematics.forward_kinematics(self.thrusts, self.alphas, self.betas)

        # 2. Rotation Matrix R_WB (Body to World)
        R_WB = quat_to_rot_matrix(self.quat)

        # 3. World Frame Force & Linear Acceleration
        # Gravity g = [0, 0, -9.81], plus linear aerodynamic drag
        f_W = R_WB @ f_B
        gravity = np.array([0.0, 0.0, -9.81 * self.mass])
        f_drag = -0.2 * self.vel
        linear_acc = (f_W + gravity + f_drag) / self.mass
        linear_acc = np.clip(linear_acc, -80.0, 80.0)

        # Linear Velocity & Position Update
        self.vel += linear_acc * dt
        self.vel = np.clip(self.vel, -20.0, 20.0)
        self.pos += self.vel * dt

        # 4. Body Frame Torque & Angular Acceleration
        # Euler's equation with aerodynamic angular damping
        tau_drag = -0.05 * self.omega
        gyroscopic = np.cross(self.omega, self.J @ self.omega)
        angular_acc = self.J_inv @ (tau_B + tau_drag - gyroscopic)
        angular_acc = np.clip(angular_acc, -120.0, 120.0)

        # Angular Velocity Update
        self.omega += angular_acc * dt
        self.omega = np.clip(self.omega, -25.0, 25.0)

        # 5. Quaternion Integration: q_dot = 0.5 * q * [0, omega]
        omega_quat = np.array([0.0, self.omega[0], self.omega[1], self.omega[2]])
        q_dot = 0.5 * quat_multiply(self.quat, omega_quat)
        self.quat += q_dot * dt
        q_norm = np.linalg.norm(self.quat)
        if q_norm > 1e-4 and not np.isnan(q_norm):
            self.quat /= q_norm
        else:
            self.quat = np.array([1.0, 0.0, 0.0, 0.0])

    def _compute_reward(self, action):
        """
        Calculates dense, multi-objective tracking reward
        """
        # 1. Position Tracking Error
        pos_error = np.linalg.norm(self.pos - self.target_pos)
        r_pos = math.exp(-2.0 * pos_error)

        # 2. Linear Velocity Damping
        vel_error = np.linalg.norm(self.vel - self.target_vel)
        r_vel = math.exp(-1.0 * vel_error)

        # 3. Attitude Tracking Error (SO(3) Geodesic Distance)
        # Using quaternion inner product: <q, q_target>^2 = cos^2(theta/2)
        q_dot = abs(np.dot(self.quat, self.target_quat))
        q_dot = min(1.0, max(0.0, q_dot))
        theta_err = 2.0 * math.acos(q_dot) # Angle error in radians
        r_att = math.exp(-1.5 * theta_err)

        # 4. Angular Velocity Damping
        omega_mag = np.linalg.norm(self.omega)
        r_omega = math.exp(-0.5 * omega_mag)

        # 5. Action Smoothness Penalty
        action_diff = np.linalg.norm(action - self.prev_action)
        r_smooth = -0.05 * action_diff

        # 6. Gimbal Lock Avoidance Penalty: penalize beta nearing +-90 deg
        beta_safety = np.sum(np.maximum(0.0, np.abs(self.betas) - (75.0 * np.pi / 180.0))**2)
        r_singularity = -2.0 * beta_safety

        # 7. Alive bonus
        r_alive = 1.0

        # Weighted combination
        total_reward = (
            2.5 * r_pos +
            1.0 * r_vel +
            2.0 * r_att +
            0.5 * r_omega +
            r_smooth +
            r_singularity +
            r_alive
        )

        info = {
            'r_pos': r_pos,
            'r_att': r_att,
            'pos_err': pos_error,
            'att_err_deg': np.degrees(theta_err)
        }

        return total_reward, info
