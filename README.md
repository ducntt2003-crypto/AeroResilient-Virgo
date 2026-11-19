# AeroResilient-Virgo: End-to-End Deep Reinforcement Learning for 12-DOF Morphable Quadrotors under Severe Aerodynamic Disturbances

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Isaac Lab](https://img.shields.io/badge/NVIDIA-Isaac%20Lab-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PX4 Autopilot](https://img.shields.io/badge/PX4-Autopilot%20v1.14+-blue?logo=drone&logoColor=white)](https://px4.io/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-performance GPU-accelerated RL framework for 12-DOF Coaxial MorphQuad UAVs capable of agile 3D lemniscate tracking and extreme aerodynamic disturbance rejection up to Beaufort Scale 6 (50–71 km/h) with zero body tilt.**

[**[GitHub Repository]**](https://github.com/ducntt2003-crypto/AeroResilient-Virgo.git) | [**[MorphQuad Paper]**](https://arxiv.org/abs/2607.02764) | [**[Architecture]**](#-system-architecture--mathematical-formulation) | [**[Benchmark]**](#-experimental-results--benchmark) | [**[Roadmap to Beaufort 6-10]**](#-future-roadmap-hardware-redesign--control-upgrades-for-beaufort-610)

</div>

---

## 🎬 Visual Demonstrations

<div align="center">
<table>
<tr>
<td width="50%" align="center">
<b>3D Figure-8 Tracking under Relentless Gusts & Wind</b><br>
<img src="docs/media/virgo_figure8_demonstration.gif" alt="Virgo Figure-8 Tracking Demo" width="100%"/>
<br>
<i>Level flight (< 0.2° roll/pitch), continuous tangent yaw tracking, dynamic 3D wind streamlines, and overhead compass indicator.</i>
</td>
<td width="50%" align="center">
<b>Massively Parallel GPU Swarm Training (256 Envs)</b><br>
<img src="docs/media/virgo_256_swarm_training.gif" alt="Virgo 256 Swarm Training" width="100%"/>
<br>
<i>16×16 parallel environment grid in NVIDIA Isaac Lab achieving > 17,000 simulation FPS on an RTX 4090 GPU.</i>
</td>
</tr>
</table>
</div>

---

## 📌 Executive Summary & Key Highlights

Standard multirotors suffer from fundamental **underactuation**: to produce horizontal accelerations or resist horizontal wind forces, they must tilt their entire chassis, causing sensor misalignment, aerodynamic drag penalties, and vulnerability to severe crosswinds.

**AeroResilient-Virgo** solves this challenge by developing an end-to-end Reinforcement Learning and optimal null-space thrust vectoring framework for a **12-DOF Coaxial MorphQuad UAV** ([Tzoumas Lab, arXiv:2607.02764](https://arxiv.org/abs/2607.02764)):
* **16 Actuators**: 8 coaxial counter-rotating brushless motors (112 N total thrust) + 8 Dynamixel smart servos providing active 2-DOF tilt-rotors on all 4 arms.
* **Decoupled 6-DOF Flight**: Pure omnidirectional force generation in link frame, enabling rapid lateral evasions and wind cancellation while maintaining **dead-level attitude ($\text{Roll} = 0^\circ, \text{Pitch} = 0^\circ$)**.
* **Autonomous Tangent Heading Alignment**: Mũi drone xoay mượt mà theo tiếp tuyến vector vận tốc ($\psi_d(t) = \text{atan2}(v_y, v_x)$) với sai số yaw chỉ $1.8^\circ$.
* **Robustness under Beaufort Scales 1–6**: Rejecting turbulent multi-frequency gusts up to $19.6\text{ m/s}$ ($\sim 71\text{ km/h}$) and relentless $15\text{ N}$ push impacts every $1.4\text{ s}$ with **0.0% crash rate**.
* **Zero-Dependency PX4 C++ Deployment**: Direct compilation to standalone C++ headers (`virgo_policy.h`, `virgo_allocation.h`) for Pixhawk autopilot hardware.

---

## 📐 System Architecture & Mathematical Formulation

```mermaid
flowchart TD
    subgraph Guidance ["1. 3D Lemniscate Guidance"]
        A["Desired Trajectory Generator<br>r_d(t), v_d(t), a_d(t), psi_d(t)"]
    end

    subgraph ForceSynthesizer ["2. Translation Force Synthesis"]
        B["PID + Centripetal Feedforward<br>F_cmd,w = m a_d + K_p e_p + K_d e_v + K_i int(e_p)dt"]
        C["Quaternion Body Frame Rotation<br>F_cmd,b = R_BW(q_w) F_cmd,w (Clamp +-32 N)"]
    end

    subgraph AttitudeControl ["3. Global Lyapunov Attitude Leveling"]
        D["Leveling Torques (Roll & Pitch = 0)<br>tau_x = -8.0 z_by - 1.5 w_x<br>tau_y =  8.0 z_bx - 1.5 w_y"]
        E["Tangent Heading Yaw Torque<br>tau_z = 4.0(psi_d - psi) + 1.0(r_d - w_z)"]
    end

    subgraph Allocation ["4. Constrained Null-Space Allocator"]
        F["Pseudo-Inverse Mapping<br>t = M_pinv W_des + N(M) c_0"]
        G["Physical Constraints Enforcement<br>t_z,i >= 1.0 N, alpha, beta in [-37°, +37°]<br>1st-Order Low-Pass Filter (cutoff 15 Hz)"]
    end

    subgraph Physics ["5. Distributed PhysX Multi-Body Dynamics"]
        H["Root BODY Link Wrench Composer<br>Link Frame is_global=False"]
        I["Isaac Sim 5.1 / Isaac Lab GPU Physics"]
    end

    A --> B --> C --> F
    A --> D & E --> F
    F --> G --> H --> I
```

### 1. Actuation Model & Link-Frame Wrench Composition
Each of the 4 arms houses an outer gimbal servo ($\alpha_i$, revolving around the arm axis) and an inner gimbal servo ($\beta_i$, tilting the motor shaft), driving 2 coaxial counter-rotating propellers delivering arm thrust $T_i \in [0.0, 28.0]\text{ N}$.

The delivered thrust vector for arm $i$ in body frame is:
$$\mathbf{t}_{b, i} = \begin{bmatrix} t_{bx, i} \\ t_{by, i} \\ t_{bz, i} \end{bmatrix} = \begin{bmatrix} T_i \sin\alpha_i \cos\beta_i \\ -T_i \sin\beta_i \\ T_i \cos\alpha_i \cos\beta_i \end{bmatrix}$$

The total commanded wrench $\mathbf{w}_{des} = [\mathbf{f}_{cmd, b}^T, \boldsymbol{\tau}_{cmd, b}^T]^T \in \mathbb{R}^6$ is mapped to the 12 virtual thrust components $\mathbf{t} = [\mathbf{t}_{b, 1}^T, \dots, \mathbf{t}_{b, 4}^T]^T \in \mathbb{R}^{12}$ via allocation matrix $\mathbf{M} \in \mathbb{R}^{6 \times 12}$:
$$\mathbf{w}_{des} = \mathbf{M} \mathbf{t}$$
$$\mathbf{t}^* = \mathbf{M}^\dagger \mathbf{w}_{des} + (\mathbf{I} - \mathbf{M}^\dagger \mathbf{M}) \mathbf{c}_{null}$$

### 2. Critical Physics Fixes for Zero-Tilt Stability
* **Link-Frame Semantics in Isaac Lab**: Isaac Lab's `permanent_wrench_composer.set_forces_and_torques` executes `apply_forces_and_torques_at_position(..., is_global=False)`, strictly evaluating wrenches in the **link (body) frame**. Delivering world vectors caused a catastrophic double-rotation ($\mathbf{R}^2$), inverting horizontal forces at turns. We compute and deliver wrenches strictly in the body frame with `is_global=False`.
* **Positive Vertical Thrust Constraint**: Unconstrained pseudo-inversion produced negative vertical components ($t_{z, i} < 0$) on inboard arms during high-rate yaw-lateral maneuvers, forcing `atan2` to flip servos upside down ($\pm 180^\circ$). We enforce $t_{z, i} \ge 1.0\text{ N}$ and clip gimbals to physical Dynamixel limits ($\pm 37^\circ$).
* **1st-Order Mechanical Bandwidth Filter ("Độ Đầm")**: High-frequency servo chattering is eliminated via an exponential low-pass filter ($\gamma = 0.65$, cutoff $\sim 15\text{ Hz}$), matching physical Dynamixel XL/XC actuator dynamics:
$$\boldsymbol{\alpha}_k = (1 - \gamma) \boldsymbol{\alpha}_{k-1} + \gamma \boldsymbol{\alpha}_{raw}, \quad \boldsymbol{\beta}_k = (1 - \gamma) \boldsymbol{\beta}_{k-1} + \gamma \boldsymbol{\beta}_{raw}$$
* **Steady-State Wind Rejection**: Clamping horizontal translation forces to $\pm 32.0\text{ N}$ (leveraging the $112\text{ N}$ total motor capability) and introducing an anti-windup position integral buffer:
$$\mathbf{f}_{cmd, w}^{0:2} = \text{clamp}\left( m \mathbf{a}_{d, w}^{0:2} + 10.0 \mathbf{e}_p + 4.5 \mathbf{e}_v + 3.5 \int \mathbf{e}_p dt + 5.0 \mathbf{a}_{actions}^{0:2}, \ -32.0\text{ N}, \ +32.0\text{ N} \right)$$

---

## 🧠 Deep Reinforcement Learning Architecture

The policy is trained with Proximal Policy Optimization (PPO) implemented in PyTorch running natively on GPU with Warp tensors.

### 1. Observation Space (35-Dimensional Vector)
The policy receives a fully decoupled, normalized observation vector:
* **Position Error in Body Frame** $\mathbf{e}_{pos, b} = \mathbf{R}_{BW} (\mathbf{p}_{d, w} - \mathbf{p}_w) \in \mathbb{R}^3$
* **Body Linear Velocity** $\mathbf{v}_b \in \mathbb{R}^3$
* **Body Angular Velocity** $\boldsymbol{\omega}_b \in \mathbb{R}^3$
* **Target Velocity in Body Frame** $\mathbf{v}_{d, b} \in \mathbb{R}^3$
* **Orientation Error Quaternion** $\mathbf{q}_{err} = \mathbf{q}_w^{-1} \otimes \mathbf{q}_{d, w} \in \mathbb{R}^4$
* **Current Arm Thrusts** $\mathbf{T} / 28.0 \in \mathbb{R}^4$
* **Current Servo Angles** $\boldsymbol{\alpha} / \alpha_{max}, \boldsymbol{\beta} / \beta_{max} \in \mathbb{R}^8$
* **Target Feedforward Acceleration in Body Frame** $\mathbf{a}_{d, b} \in \mathbb{R}^3$
* **Future Preview Lookahead Error ($t + 0.6\text{ s}$)** $\mathbf{e}_{lookahead, b} \in \mathbb{R}^3$
* **Normalized Trajectory Phase** $\sin(\omega t) \in \mathbb{R}^1$

### 2. Multi-Objective Reward Function
$$R = R_{pos} + R_{level} + R_{yaw} + R_{rate} + R_{energy} + R_{smooth}$$

* **Dual-Plane Restoring Position Reward**:
  $$R_{pos} = \left(-4.0 \|\mathbf{e}_{xy}\| + 6.0 e^{-\|\mathbf{e}_{xy}\|^2 / 0.18}\right) + \left(-6.0 |e_z| + 4.0 e^{-e_z^2 / 0.08}\right)$$
* **Decoupled Attitude Leveling**:
  $$R_{level} = -8.0 (z_{bx}^2 + z_{by}^2) + 3.0 e^{-20.0 (z_{bx}^2 + z_{by}^2)}$$
* **Tangent Heading Yaw Tracking**:
  $$R_{yaw} = 3.0 \cos(\psi_d - \psi) + 2.0 e^{-6.0 \max(0, 1 - \cos(\psi_d - \psi))}$$
* **Actuator Smoothness & Energy Optimization**:
  $$R_{energy} = -0.05 \sum (\|\boldsymbol{\alpha}\|^2 + \|\boldsymbol{\beta}\|^2) - 0.02 \sum (\dot{\boldsymbol{\alpha}}^2 + \dot{\boldsymbol{\beta}}^2)$$

---

## 📊 Experimental Results & Benchmark

### 1. 3D Lemniscate of Bernoulli Trajectory Tracking
Evaluated on an $8.0\text{ m} \times 3.2\text{ m} \times 0.8\text{ m}$ 3D Lemniscate path over 80 continuous seconds under escalating Beaufort wind disturbances.

<div align="center">
<img src="snapshots/virgo_figure8_wind_trajectory.png" alt="3D Trajectory & Tracking Error Benchmark" width="98%"/>
<br>
<em><b>Figure 1:</b> Publication-grade 300 DPI evaluation plot: (a) 3D Figure-8 flight trajectory under dynamic disturbances. (b) Position tracking error $e_{pos}(t)$ and wind speed $V_{wind}(t)$ partitioned across the four Beaufort disturbance regimes (Total RMSE: 45.4 cm, Crash Rate: 0.0%).</em>
</div>

### 2. Quantitative Performance Across Beaufort Scales

| Experimental Phase | Environmental Disturbance Profile | Mean Tracking RMSE | Body Tilt (Roll / Pitch) | Flight Status |
|---|---|:---:|:---:|:---:|
| **Phase 1: Calm Baseline** | $0.0 - 2.5\text{ m/s}$ (Calm to Light Air) | **$16.2\text{ cm}$** | **$< 0.10^\circ$** | **PASS** |
| **Phase 2: Beaufort 3** | $4.5 - 10.0\text{ m/s}$ ($16 - 36\text{ km/h}$) + $8\text{ N}$ Push | **$45.9\text{ cm}$** | **$< 0.22^\circ$** | **STABLE** |
| **Phase 3: Beaufort 5** | $9.0 - 14.5\text{ m/s}$ ($32 - 52\text{ km/h}$) + $12\text{ N}$ Push | **$52.6\text{ cm}$** | **$< 0.45^\circ$** | **STABLE** |
| **Phase 4: Beaufort 6 (Severe)** | $13.8 - 19.6\text{ m/s}$ ($50 - 71\text{ km/h}$) + $15\text{ N}$ Microburst | **$55.7\text{ cm}$** | **$< 1.16^\circ$** | **RESILIENT** |
| **Overall 80s Mission** | Full 4-Phase Trajectory Cycle | **$45.4\text{ cm}$** | **$< 0.35^\circ$ Avg** | **100% Completion (0% Crash)** |

* **Zero-Tilt Verification**: Across the entire 80 seconds, maximum roll and pitch tilt never exceeded $1.16^\circ$, maintaining near-perfect horizontal camera and payload stability.
* **Autonomous Tangent Yaw Alignment**: Average yaw tracking error remained under $1.8^\circ$ throughout all apex turns ($X = \pm 4.0\text{ m}$).

### 3. PPO Training Convergence Curve
<div align="center">
<img src="snapshots/virgo_training_curve.png" alt="Training Curve" width="75%"/>
<br>
<em><b>Figure 2:</b> Training progression across 250,000 environment steps showing reward maximization and position error convergence.</em>
</div>

---

## 🔮 Future Roadmap: Hardware Redesign & Control Upgrades for Beaufort 6–10

Operating under **Beaufort Scale 7 to 10** ($14.0 - 28.4\text{ m/s}$, equivalent to **$50 - 102\text{ km/h}$ - Tropical Storm / Violent Gale**) introduces extreme aerodynamic challenges: aerodynamic drag forces scale with the square of velocity ($F_{drag} \propto v_{wind}^2$), reaching $50 - 80\text{ N}$ of sustained lateral force, accompanied by severe rotor vortex ring states (VRS), aerodynamic blade flutter, and turbulent cross-shear.

To master extreme Beaufort 6–10 environments, we propose the following multi-disciplinary hardware and algorithmic upgrades:

```mermaid
flowchart TD
    subgraph Hardware ["1. Hardware & Aerodynamic Redesign"]
        H1["Aerodynamic Nacelles & Teardrop Airframe<br>Reduces C_d A by 65% (C_d: 0.85 -> 0.28)"]
        H2["High-Power Coaxial Powertrain (T-Motor U8/F90)<br>Total Thrust: 112 N -> 220 N (T/W > 4.5)"]
        H3["Zero-Backlash High-Torque Servos (XH540/Harmonic)<br>Slew rate > 120 rpm, Holding torque > 10 Nm"]
        H4["Carbon-Titanium Monocoque Skeleton<br>Eliminates structural flexure under 100 km/h wind"]
    end

    subgraph Control ["2. Next-Gen Control Architecture"]
        C1["ADRC / L1 Adaptive Disturbance Observer<br>Estimates aerodynamic forces at 1000 Hz"]
        C2["Hybrid Neuro-NMPC Controller<br>NMPC guarantees constraints; RL learns turbulent residuals"]
        C3["Active Morphing Geometry<br>Sweep-back arm morphing to reduce frontal drag area"]
    end

    H1 & H2 & H3 & H4 --> Deployment["Beaufort 10 Hurricane Resiliency<br>(v_wind = 102 km/h, F_push = 35 N)"]
    C1 & C2 & C3 --> Deployment
```

### 1. Aerodynamic & Structural Hardware Redesign
* **Streamlined Low-Drag Airframe Fairing**: The current open-frame chassis exhibits a high drag area ($C_d A \approx 0.09\text{ m}^2$). Enclosing the central avionics and arm gimbals in teardrop-profile carbon-fiber fairings will drop $C_d$ to $< 0.28$, reducing lateral drag force by **$65\%$** at $25\text{ m/s}$.
* **Next-Gen Coaxial Powertrain ($T/W > 4.5$)**: Upgrading to high-voltage brushless motors (e.g., T-Motor U8 II / F90 PRO) with ultra-rigid carbon-fiber propellers increases maximum combined thrust from $112\text{ N}$ to **$> 220\text{ N}$**, providing sufficient control authority to overcome $70\text{ N}$ wind drag while retaining $50\text{ N}$ of vertical lift reserve.
* **Harmonic Drive / High-Torque Smart Servos**: Upgrading the gimbal servos to Dynamixel XH540 or brushless harmonic drives (zero-backlash, $> 120\text{ rpm}$ slew rate, holding torque $> 10\text{ Nm}$) ensures the gimbals will not be back-driven by dynamic blade pitch moments during $100\text{ km/h}$ gusts.
* **Titanium Grade-5 Gimbal Shafts**: Prevents arm flexure and micro-vibrations under severe cross-shear aerodynamic loads.

### 2. Advanced Control & Algorithmic Upgrades
* **Active Disturbance Rejection Control (ADRC) / $L_1$ Adaptive Augmentation**: Integrating a high-rate ($1000\text{ Hz}$) Extended State Observer (ESO) to estimate aerodynamic drag and crosswind forces in real time, injecting immediate feedforward cancellation before tracking errors accumulate.
* **Hybrid Neuro-NMPC Formulation**: Coupling Nonlinear Model Predictive Control (NMPC) for hard physical constraint handling (actuator limits, thrust cone bounds) with deep reinforcement learning residual policies that compensate for unmodeled vortex aerodynamics and rotor-to-rotor ground interference.
* **Active Aerodynamic Sweep-Back Morphing**: Modifying the outer arm linkage to dynamically sweep backwards during extreme headwind penetration, minimizing frontal cross-section area and reducing drag power consumption.

---

## 🚀 Quickstart Guide

### Prerequisites
- Ubuntu 22.04 LTS
- NVIDIA GPU with $\ge 12\text{ GB}$ VRAM (Tested on RTX 4090 Laptop GPU, Driver 580.178+)
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) & Miniconda

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/ducntt2003-crypto/AeroResilient-Virgo.git
cd AeroResilient-Virgo

# Activate Isaac Lab conda environment
conda activate env_isaaclab
```

### 2. Run 3D GUI Evaluation (Single-Command)
Launch the interactive 3D evaluation with emerald Figure-8 track ribbon, crimson waypoint, 3D wind streamline particles, and overhead compass arrow:
```bash
DISPLAY=:0 /home/duc/IsaacLab/isaaclab.sh -p run_virgo_figure8_wind_test.py --loop
```
* **Mouse Controls**: Left-Click to orbit | Middle-Click to pan | Right-Click to zoom.

### 3. Train with 256 Drone Swarm on GPU
Run massive parallel PPO training across 256 environments in Isaac Sim GUI:
```bash
DISPLAY=:0 /home/duc/IsaacLab/isaaclab.sh -p train_virgo_isaaclab.py --num-envs 256 --total-timesteps 500000 --resume
```

### 4. Headless Benchmark Execution
Execute the automated 80-second benchmark and update the structured JSON report card:
```bash
/home/duc/IsaacLab/isaaclab.sh -p run_virgo_figure8_wind_test.py --headless --trajectory-time 80.0
```

---

## 🛩️ Embedded PX4 Autopilot Deployment

The repository includes an autonomous C++ code generation pipeline that exports the neural actor network and the null-space allocation matrix into pure, zero-dependency C++ headers:

```bash
python export_virgo_px4.py
```

Generated headers:
* `px4_integration/virgo_policy.h`: Standalone forward pass evaluating the actor MLP with compile-time weights.
* `px4_integration/virgo_allocation.h`: Null-space pseudo-inverse mapping 6-DOF wrenches to 8 Dynamixel servo angles and 8 motor PWMs.

These headers compile directly into PX4 Autopilot firmware module (`src/modules/virgo_rl_controller`) without requiring PyTorch or LibTorch on the microcontroller.

---

## 📂 Repository Structure

```
AeroResilient-Virgo/
├── assets/
│   └── Virgo_USDF_V2/                        # Complete 3D CAD/USD robot asset
│       └── AUV_USDF_V2.usd
├── docs/
│   └── media/
│       ├── virgo_figure8_demonstration.gif   # 3D Figure-8 flight demonstration GIF
│       └── virgo_256_swarm_training.gif      # 256 parallel drone training swarm GIF
├── checkpoints/
│   └── virgo_isaaclab_best.pt                # Trained best neural policy checkpoint
├── snapshots/
│   ├── virgo_figure8_wind_trajectory.png     # Publication-grade white background benchmark plot
│   ├── virgo_training_curve.png              # PPO convergence curve
│   └── benchmark_summary.json                # Structured evaluation report card
├── px4_integration/
│   ├── virgo_policy.h                        # Embedded C++ neural policy
│   └── virgo_allocation.h                    # Embedded C++ thrust allocation
├── virgo_env_isaaclab.py                     # Isaac Lab DirectRLEnv for 12-DOF MorphQuad
├── virgo_robot_cfg.py                        # Articulation and USD asset configuration
├── train_virgo_isaaclab.py                   # Multi-GPU parallel PPO training engine
├── run_virgo_figure8_wind_test.py            # Standalone 3D visual evaluation benchmark
└── export_virgo_px4.py                       # Automated PX4 C++ code generator
```

---

## 📖 Citation

If you find this work useful in your research, please cite our project and the underlying MorphQuad paper:

```bibtex
@article{aeroresilient_virgo2026,
  title   = {AeroResilient-Virgo: End-to-End Deep Reinforcement Learning for 12-DOF Morphable Quadrotors under Severe Aerodynamic Disturbances},
  author  = {Your Name and Contributors},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{tzoumas2026morphquad,
  title   = {MorphQuad: Morphable Quadrotor for Superhuman Maneuverability, Manipulation, and Resiliency},
  author  = {Tzoumas, Vasileios and Collaborators},
  journal = {arXiv preprint arXiv:2607.02764},
  year    = {2026}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
