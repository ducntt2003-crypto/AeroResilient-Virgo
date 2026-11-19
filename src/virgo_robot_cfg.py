# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the Virgo_RL (MorphQuad 12-DOF Omnidirectional UAV) robot asset in Isaac Lab.
Paper: "MorphQuad: Morphable Quadrotor for Superhuman Maneuverability, Manipulation, and Resiliency"
ArXiv: 2607.02764
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Path to the generated USD asset (supports both relative repo path and absolute system path)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOCAL_USD = os.path.join(REPO_ROOT, "assets", "Virgo_USDF_V2", "AUV_USDF_V2.usd")
FALLBACK_USD = "/home/duc/Raptor_UAV/Virgo_USDF_V2/AUV_USDF_V2.usd"
VIRGO_USD_PATH = LOCAL_USD if os.path.exists(LOCAL_USD) else FALLBACK_USD

VIRGO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=VIRGO_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=25.0,
            max_angular_velocity=35.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "ARM_MOTOR_.*": 0.0,
            "SHAFT_MOTOR_.*": 0.0,
            "PROP_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "gimbal_servos": ImplicitActuatorCfg(
            joint_names_expr=["ARM_MOTOR_.*", "SHAFT_MOTOR_.*"],
            effort_limit_sim=10.0,       # Dynamixel XC330 / XL330 peak torque
            velocity_limit_sim=25.0,    # Max servo angular rate
            stiffness=250.0,            # High stiffness PD drive for precision gimbal tracking
            damping=12.0,               # Enhanced physical damping for steady, jitter-free servo motion
        ),
        "props": ImplicitActuatorCfg(
            joint_names_expr=["PROP_.*"],
            effort_limit_sim=5.0,       # Brushless motor peak torque
            velocity_limit_sim=3000.0,  # High RPM propeller rotation
            stiffness=0.0,              # Pure velocity drive
            damping=0.2,                # Velocity gain
        ),
    },
)
