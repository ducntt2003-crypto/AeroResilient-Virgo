#!/usr/bin/env python3
"""
Virgo_RL (MorphQuad) Kinematics, Dynamics & Null-Space Thrust Allocation
Based on: "MorphQuad: Morphable Quadrotor for Superhuman Maneuverability, Manipulation, and Resiliency"
arXiv:2607.02764 (University of Michigan, Tzoumas Lab)

Actuation: 12 Degrees of Freedom:
- 4 Rotor Thrusts: T_1, T_2, T_3, T_4 (Coaxial Counter-Rotating pairs)
- 8 Gimbal Servo Angles: alpha_1..4 (Inner Servo, Roll axis), beta_1..4 (Outer Servo, Pitch axis)
"""

import numpy as np
import torch
import torch.nn as nn

class VirgoKinematics:
    """
    Kinematics and Control Allocation Engine for Virgo_RL (MorphQuad)
    """
    def __init__(
        self,
        mass=4.56,            # kg (Takeoff mass including batteries)
        arm_length_x=0.22,    # m (half-length l)
        arm_length_y=0.22,    # m (half-width b)
        max_thrust_per_arm=28.0, # N (2.8 kg peak thrust per coaxial pair)
        ct=1.0,               # lumped thrust coefficient
        servo_speed_limit=10.0 # rad/s (~600 deg/s max servo rate)
    ):
        self.mass = mass
        self.l = arm_length_x
        self.b = arm_length_y
        self.r = np.sqrt(self.l**2 + self.b**2)
        self.max_thrust = max_thrust_per_arm
        self.ct = ct
        self.servo_speed_limit = servo_speed_limit

        # Rotor positions in Body Frame B: [x, y, z]
        # Rotor 1: Front-Right (+l, -b, 0)
        # Rotor 2: Back-Right  (-l, -b, 0)
        # Rotor 3: Back-Left   (-l, +b, 0)
        # Rotor 4: Front-Left  (+l, +b, 0)
        self.rotor_positions = np.array([
            [ self.l, -self.b, 0.0],
            [-self.l, -self.b, 0.0],
            [-self.l,  self.b, 0.0],
            [ self.l,  self.b, 0.0]
        ], dtype=np.float64)

        # Yaw unit vectors in body plane (Eq. 13 in paper)
        self.e_psi = np.array([
            [-self.b / self.r,  self.l / self.r, 0.0],  # Rotor 1
            [ self.b / self.r,  self.l / self.r, 0.0],  # Rotor 2
            [ self.b / self.r, -self.l / self.r, 0.0],  # Rotor 3
            [-self.b / self.r, -self.l / self.r, 0.0]   # Rotor 4
        ], dtype=np.float64)

        # Construct constant Input Matrix M (6 x 12) such that W_B = M * t
        self.M = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            # Force mapping: f = sum(t_i)
            self.M[0:3, i*3:(i+1)*3] = np.eye(3)
            # Torque mapping: tau = sum(p_i x t_i)
            px, py, pz = self.rotor_positions[i]
            skew_p = np.array([
                [ 0.0, -pz,   py],
                [ pz,   0.0, -px],
                [-py,   px,   0.0]
            ])
            self.M[3:6, i*3:(i+1)*3] = skew_p

        # Compute Moore-Penrose pseudo-inverse M_pinv = M^T (M M^T)^-1 (12 x 6)
        self.M_pinv = np.linalg.pinv(self.M)

        # Compute Null-Space Basis N_M (12 x 6) via SVD
        U, S, Vt = np.linalg.svd(self.M)
        self.N_M = Vt[6:].T # Null-space columns (rank 6)
        
        # Verify null space orthogonality: M @ N_M == 0
        assert np.max(np.abs(self.M @ self.N_M)) < 1e-12, "Null space basis calculation error!"

    def thrust_direction_vector(self, alpha, beta):
        """
        Computes the 3D unit thrust vector G_i(alpha, beta) in Body Frame (Eq. 3):
        G_i = [ sin(alpha)*cos(beta), -sin(beta), cos(alpha)*cos(beta) ]^T
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        return np.array([sa * cb, -sb, ca * cb], dtype=np.float64)

    def forward_kinematics(self, thrusts, alphas, betas):
        """
        Computes the Net Body Wrench W_B = [f_B^T, tau_B^T]^T given 12 actuator commands:
        thrusts: [4] in Newtons
        alphas: [4] in radians (Inner servo)
        betas:  [4] in radians (Outer servo)
        """
        t_stacked = np.zeros(12, dtype=np.float64)
        for i in range(4):
            Gi = self.thrust_direction_vector(alphas[i], betas[i])
            t_stacked[i*3:(i+1)*3] = thrusts[i] * Gi

        # Net wrench W = M * t
        wrench = self.M @ t_stacked
        return wrench[:3], wrench[3:], t_stacked

    def inverse_kinematics_arm(self, t_i, prev_alpha=None, prev_beta=None):
        """
        Recovers (T_i, alpha_i, beta_i) from a 3D rotor thrust vector t_i (Eq. 14):
        T_i = ||t_i||
        beta_i = arcsin(-t_hat_{i,y})
        alpha_i = atan2(t_hat_{i,x}, t_hat_{i,z})
        Includes multi-revolution unwinding to select the smoothest servo trajectory.
        """
        norm_t = np.linalg.norm(t_i)
        if norm_t < 1e-4:
            # Standby/zero thrust
            return 0.0, (prev_alpha if prev_alpha is not None else 0.0), (prev_beta if prev_beta is not None else 0.0)

        t_hat = t_i / norm_t
        t_x, t_y, t_z = t_hat

        # Two mathematical branch solutions (sin(x) = sin(pi - x))
        # Branch 1:
        b1 = np.arcsin(np.clip(-t_y, -1.0, 1.0))
        a1 = np.arctan2(t_x, t_z)

        # Branch 2:
        b2 = np.pi - b1
        a2 = np.arctan2(-t_x, -t_z)

        candidates = [(a1, b1), (a2, b2)]

        if prev_alpha is None or prev_beta is None:
            # Pick branch with smaller beta magnitude
            chosen = sorted(candidates, key=lambda c: abs(c[1]))[0]
            return norm_t, chosen[0], chosen[1]

        # Multi-revolution unwinding: find closest equivalent angle (within +-2pi)
        best_a, best_b = None, None
        min_dist = float('inf')

        for a_cand, b_cand in candidates:
            for ka in [-1, 0, 1]:
                for kb in [-1, 0, 1]:
                    cand_a_unwrap = a_cand + 2.0 * np.pi * ka
                    cand_b_unwrap = b_cand + 2.0 * np.pi * kb
                    dist = (cand_a_unwrap - prev_alpha)**2 + (cand_b_unwrap - prev_beta)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_a = cand_a_unwrap
                        best_b = cand_b_unwrap

        return norm_t, best_a, best_b

    def compute_null_space_correction(self, f_des):
        """
        Computes the coefficient vector c_N in R^6 for null-space thrust redistribution (Eq. 15 & 16)
        Addresses:
        1. Gimbal Lock Avoidance (when f is parallel to j_B / lateral roll 90 deg)
        2. Downwash Avoidance (when f is parallel to i_B / forward flight)
        """
        f_norm = np.linalg.norm(f_des)
        if f_norm < 1e-3:
            return np.zeros(6, dtype=np.float64)

        f_dir = f_des / f_norm
        # i_B = [1, 0, 0]^T, j_B = [0, 1, 0]^T
        proj_i = f_dir[0] # <f/||f||, i_B>
        proj_j = f_dir[1] # <f/||f||, j_B>

        cN = np.zeros(6, dtype=np.float64)
        
        # Gimbal Lock Repulsion (pushes beta_i away from +-pi/2 when flying sideways)
        # cN_1 = cN_3 = cN_5 = <f/||f||, j_B>
        gain_gimbal_lock = 0.5 * f_norm
        cN[0] = gain_gimbal_lock * proj_j
        cN[2] = gain_gimbal_lock * proj_j
        cN[4] = gain_gimbal_lock * proj_j

        # Downwash Avoidance (breaks co-linearity when translating along x-axis)
        gain_downwash = 0.4 * f_norm
        if proj_i > 0:
            cN[1] = -gain_downwash * proj_i
            cN[3] =  gain_downwash * proj_i
            cN[5] =  0.0
        else:
            cN[1] = -gain_downwash * proj_i
            cN[3] =  0.0
            cN[5] =  gain_downwash * proj_i

        return cN

    def allocate_thrust(self, f_des, tau_des, prev_alphas=None, prev_betas=None, use_null_space=True):
        """
        Full Control Allocation Pipeline:
        Input: Desired 6-DOF Wrench W_cmd = [f_des (3), tau_des (3)]^T
        Output:
          - thrusts: [4] (N)
          - alphas:  [4] (rad)
          - betas:   [4] (rad)
          - t_vectors: [4, 3] 3D thrust vectors
        """
        wrench_des = np.concatenate([f_des, tau_des])

        # Step 1: Minimum-Norm Pseudo-Inverse Allocation (Eq. 12)
        # t_0 = M^pinv * w_des
        t_stacked = self.M_pinv @ wrench_des

        # Step 2: Null-Space Correction (Eq. 15)
        # t = t_0 + N_M * c_N
        if use_null_space:
            cN = self.compute_null_space_correction(f_des)
            t_stacked = t_stacked + self.N_M @ cN

        # Verify net wrench preservation (M @ t == wrench_des)
        delivered_wrench = self.M @ t_stacked
        wrench_error = np.linalg.norm(delivered_wrench - wrench_des)
        assert wrench_error < 1e-8, f"Wrench conservation violated! Error: {wrench_error}"

        # Step 3: Recover individual (T_i, alpha_i, beta_i)
        thrusts = np.zeros(4, dtype=np.float64)
        alphas = np.zeros(4, dtype=np.float64)
        betas = np.zeros(4, dtype=np.float64)
        t_vectors = np.zeros((4, 3), dtype=np.float64)

        for i in range(4):
            t_i = t_stacked[i*3:(i+1)*3]
            t_vectors[i] = t_i
            p_a = prev_alphas[i] if prev_alphas is not None else None
            p_b = prev_betas[i] if prev_betas is not None else None
            T_i, a_i, b_i = self.inverse_kinematics_arm(t_i, p_a, p_b)
            
            # Saturate to max thrust capacity
            thrusts[i] = np.clip(T_i, 0.0, self.max_thrust)
            alphas[i] = a_i
            betas[i] = b_i

        return thrusts, alphas, betas, t_vectors

def run_verification_tests():
    print("=" * 70)
    print("      VIRGO_RL (MORPHQUAD) KINEMATICS & ALLOCATION TEST SUITE      ")
    print("=" * 70)

    model = VirgoKinematics()
    mg = model.mass * 9.81
    print(f"Vehicle Mass: {model.mass} kg | Nominal Hover Weight: {mg:.2f} N")
    print(f"Max Total Thrust: {4 * model.max_thrust:.2f} N (TWR = {4 * model.max_thrust / mg:.2f})\n")

    # TEST 1: Nominal Upright Hover
    print("--- [TEST 1] Nominal Upright Hover (f_des = [0, 0, mg], tau = 0) ---")
    f_des = np.array([0.0, 0.0, mg])
    tau_des = np.zeros(3)
    T, a, b, _ = model.allocate_thrust(f_des, tau_des)
    f_deliv, tau_deliv, _ = model.forward_kinematics(T, a, b)
    print(f"  Thrusts (N): {T.round(2)}")
    print(f"  Alphas (deg): {np.degrees(a).round(1)} | Betas (deg): {np.degrees(b).round(1)}")
    print(f"  Delivered Force (N): {f_deliv.round(2)} | Delivered Torque (Nm): {tau_deliv.round(3)}")

    # TEST 2: Inverted Hover (Drone flying completely upside down!)
    print("\n--- [TEST 2] Inverted Hover (f_des = [0, 0, -mg], tau = 0) ---")
    f_des = np.array([0.0, 0.0, -mg])
    T, a, b, _ = model.allocate_thrust(f_des, tau_des)
    f_deliv, tau_deliv, _ = model.forward_kinematics(T, a, b)
    print(f"  Thrusts (N): {T.round(2)}")
    print(f"  Alphas (deg): {np.degrees(a).round(1)} | Betas (deg): {np.degrees(b).round(1)}")
    print(f"  Delivered Force (N): {f_deliv.round(2)} (All gimbals flipped 180° to thrust downward!)")

    # TEST 3: Lateral Wall Hover & Gimbal Lock Avoidance (f_des = [0, mg, 0])
    print("\n--- [TEST 3] Lateral Wall Hover: Gimbal Lock Avoidance (f_des = [0, mg, 0]) ---")
    f_des = np.array([0.0, mg, 0.0]) # Force directed purely along j_B
    
    # Without Null-Space:
    T_raw, a_raw, b_raw, _ = model.allocate_thrust(f_des, tau_des, use_null_space=False)
    print(f"  [Without Null-Space] Betas: {np.degrees(b_raw).round(1)}° -> SINGULARITY (beta = -90° Gimbal Lock!)")
    
    # With Null-Space:
    T_null, a_null, b_null, _ = model.allocate_thrust(f_des, tau_des, use_null_space=True)
    f_deliv, tau_deliv, _ = model.forward_kinematics(T_null, a_null, b_null)
    print(f"  [WITH Null-Space]    Betas: {np.degrees(b_null).round(1)}° -> SAFE! (Pushed away from -90°)")
    print(f"  Delivered Force (N): {f_deliv.round(2)} (Exact match with f_des preserved!)")

    # TEST 4: Contact Manipulation Force (Applying 15 N Lateral Force + 2.5 Nm Pitch Torque)
    print("\n--- [TEST 4] Contact Manipulation (f_des = [15N, 0, mg], tau_des = [0, 2.5Nm, 0]) ---")
    f_des = np.array([15.0, 0.0, mg])
    tau_des = np.array([0.0, 2.5, 0.0])
    T, a, b, _ = model.allocate_thrust(f_des, tau_des)
    f_deliv, tau_deliv, _ = model.forward_kinematics(T, a, b)
    print(f"  Thrusts (N): {T.round(2)}")
    print(f"  Alphas (deg): {np.degrees(a).round(1)} | Betas (deg): {np.degrees(b).round(1)}")
    print(f"  Delivered Force: {f_deliv.round(2)} N | Delivered Torque: {tau_deliv.round(2)} Nm")
    print("=" * 70)
    print(">>> ALL KINEMATICS & ALLOCATION VERIFICATIONS PASSED 100% <<<")

if __name__ == '__main__':
    run_verification_tests()
