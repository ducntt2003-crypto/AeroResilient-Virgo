import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_diagram():
    fig, ax = plt.subplots(figsize=(15.5, 9.0), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    # Title Banner
    ax.text(7.75, 8.85, "AeroResilient-Virgo: System Architecture & Control Pipeline", 
            ha='center', va='center', fontsize=16.5, fontweight='bold', color='#0D47A1')
    ax.text(7.75, 8.48, "Decoupled 6-DOF Flight, Lyapunov Dead-Leveling, Constrained Null-Space Thrust Allocation & PPO Disturbance Rejection", 
            ha='center', va='center', fontsize=10.5, style='italic', color='#37474F')

    # Helper function for rounded fancy boxes
    def draw_box(x, y, w, h, box_color, border_color, title, subtitle_lines, tag=""):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.25", 
                             facecolor=box_color, edgecolor=border_color, linewidth=2.0, zorder=2)
        ax.add_patch(box)
        
        # Tag Badge
        if tag:
            badge = FancyBboxPatch((x + 0.25, y + h - 0.42), 0.7, 0.32, boxstyle="round,pad=0.05,rounding_size=0.1",
                                   facecolor=border_color, edgecolor='none', zorder=3)
            ax.add_patch(badge)
            ax.text(x + 0.60, y + h - 0.26, tag, ha='center', va='center', 
                    fontsize=8.5, fontweight='bold', color='white', zorder=4)
        
        # Title
        title_x = x + 1.15 if tag else x + w/2
        title_align = 'left' if tag else 'center'
        ax.text(title_x, y + h - 0.26, title, ha=title_align, va='center', 
                fontsize=11.5, fontweight='bold', color=border_color, zorder=3)
        
        # Divider line
        ax.plot([x + 0.2, x + w - 0.2], [y + h - 0.58, y + h - 0.58], color=border_color, lw=1.0, alpha=0.4, zorder=3)
        
        # Content
        line_y = y + h - 0.88
        for line, is_code in subtitle_lines:
            weight = 'bold' if is_code else 'normal'
            font = 'monospace' if is_code else 'sans-serif'
            size = 8.6 if is_code else 8.8
            col = '#1A237E' if is_code else '#263238'
            ax.text(x + w/2, line_y, line, ha='center', va='center', 
                    fontsize=size, fontweight=weight, fontfamily=font, color=col, zorder=3)
            line_y -= 0.32

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2, label="", label_pos=0.5, label_offset=(0, 0.18), color='#263238'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=15),
                    zorder=4)
        if label:
            lx = x1 + (x2 - x1) * label_pos + label_offset[0]
            ly = y1 + (y2 - y1) * label_pos + label_offset[1]
            ax.text(lx, ly, label, ha='center', va='center', fontsize=8.5, 
                    fontweight='bold', color=color, 
                    bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='#B0BEC5', lw=0.6, alpha=0.95),
                    zorder=5)

    # 1. 3D Lemniscate Guidance
    draw_box(0.6, 4.6, 3.4, 3.4, '#E3F2FD', '#1565C0', "3D Lemniscate Guidance", [
        ("Desired Trajectory Generator:", False),
        ("r_d(t) = [Ax sin wt, Ay sin 2wt, z_d]", True),
        ("v_d(t), a_d(t) Feedforward", True),
        ("Autonomous Tangent Heading Alignment:", False),
        ("psi_d(t) = atan2(v_y, v_x)", True),
        ("Future Lookahead Preview Window:", False),
        ("e_lookahead = r_d(t + 0.6s) - p(t)", True)
    ], "M1")

    # 2. Translation Force Synthesis
    draw_box(4.8, 4.6, 4.2, 3.4, '#E8F5E9', '#2E7D32', "Translation Force Synthesis", [
        ("PID + Acceleration Feedforward (World):", False),
        ("F_cmd,w = m a_d + Kp e_p + Kd e_v + Ki ∫e_p dt", True),
        ("Decoupled Link-Frame Body Rotation:", False),
        ("F_cmd,b = R_BW(q_w) * F_cmd,w", True),
        ("Steady-State Anti-Windup Clamp:", False),
        ("clamp(F_cmd,b^xy, -32.0 N, +32.0 N)", True),
        ("Full Omnidirectional Thrust Reserve", False)
    ], "M2")

    # 3. Attitude & Heading Control
    draw_box(4.8, 0.7, 4.2, 3.4, '#FFF3E0', '#E65100', "Attitude Leveling & Yaw", [
        ("Global Lyapunov Attitude Leveling:", False),
        ("Dead-Level: Roll = 0.0°, Pitch = 0.0°", True),
        ("tau_x = -8.0 z_by - 1.5 w_x", True),
        ("tau_y =  8.0 z_bx - 1.5 w_y", True),
        ("Tangent Heading Yaw Torque:", False),
        ("tau_z = 4.0(psi_d - psi) + 1.0(r_d - w_z)", True),
        ("Decoupled from Lateral Wind Resiliency", False)
    ], "M3")

    # 4. Constrained Null-Space Allocator
    draw_box(9.8, 1.8, 5.1, 6.2, '#EDE7F6', '#512DA8', "Constrained Null-Space Allocator", [
        ("Commanded 6-DOF Wrench Vector:", False),
        ("w_des = [f_cmd,b^T,  tau_cmd,b^T]^T in R^6", True),
        ("Pseudo-Inverse Virtual Thrust Mapping:", False),
        ("t* = M_pinv * w_des + (I - M_pinv*M)*c_null", True),
        ("Tilt-Rotor Physical Decomposition:", False),
        ("t_bx,i =  T_i * sin(a_i) * cos(b_i)", True),
        ("t_by,i = -T_i * sin(b_i)", True),
        ("t_bz,i =  T_i * cos(a_i) * cos(b_i)", True),
        ("Physical Hardware Constraints:", False),
        ("t_bz,i >= 1.0 N  (Positive Lift Constraint)", True),
        ("alpha_i, beta_i in [-37.0°, +37.0°]", True),
        ("T_i in [0.0 N, 28.0 N] per arm", True),
        ("1st-Order Low-Pass Filter (~15 Hz):", False),
        ("alpha_k = (1-g)*alpha_k-1 + g*alpha_raw", True)
    ], "M4")

    # 5. PhysX Multi-Body Simulation / Real Hardware
    draw_box(0.6, 0.7, 3.4, 3.4, '#E0F2F1', '#00695C', "PhysX Dynamics / Robot", [
        ("NVIDIA Isaac Sim 5.1 / Isaac Lab:", False),
        ("Permanent Wrench Composer", True),
        ("Link-Frame Application (is_global=False):", False),
        ("Applies wrench strictly in BODY frame", True),
        ("16 Actuators Total Control Authority:", False),
        ("8 Coaxial Motors (112 N total thrust)", True),
        ("8 Dynamixel Servos (Active Tilt)", True)
    ], "M5")

    # Connective Arrows
    # 1 -> 2 (Guidance to Force Synth)
    draw_arrow(4.0, 6.3, 4.8, 6.3, "r_d, v_d, a_d")
    
    # 1 -> 3 (Guidance to Yaw/Attitude)
    draw_arrow(2.3, 4.6, 2.3, 3.8)
    draw_arrow(2.3, 3.8, 4.8, 2.4, "psi_d(t), r_d")
    
    # 2 -> 4 (Force Synth to Allocator)
    draw_arrow(9.0, 6.3, 9.8, 6.3, "f_cmd,b (3D)")
    
    # 3 -> 4 (Attitude to Allocator)
    draw_arrow(9.0, 2.4, 9.8, 3.8, "tau_cmd,b (3D)")
    
    # 4 -> 5 (Allocator down and left to Physics/Robot)
    ax.annotate('', xy=(4.0, 2.0), xytext=(9.8, 2.2),
                arrowprops=dict(arrowstyle="-|>", color='#512DA8', lw=2.0, mutation_scale=16, 
                                connectionstyle="arc3,rad=0.12"),
                zorder=4)
    ax.text(6.8, 1.45, "Actuator Commands: [T_i, alpha_i, beta_i]", ha='center', va='center', 
            fontsize=9.0, fontweight='bold', color='#512DA8',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='#512DA8', lw=1.0, alpha=0.95),
            zorder=5)

    # 5 feedback -> 1 & 2 & 3
    ax.annotate('', xy=(0.6, 6.3), xytext=(0.6, 2.4),
                arrowprops=dict(arrowstyle="-|>", color='#00695C', lw=1.8, mutation_scale=14,
                                connectionstyle="arc3,rad=-0.38"),
                zorder=4)
    ax.text(0.18, 4.3, "Full State Feedback [p, v, q, w]", ha='center', va='center', rotation=90,
            fontsize=8.8, fontweight='bold', color='#00695C',
            bbox=dict(boxstyle='square,pad=0.18', facecolor='white', edgecolor='#00695C', lw=0.8, alpha=0.95),
            zorder=5)

    plt.tight_layout()
    output_path = '/home/duc/Raptor_UAV/Virgo_RL/snapshots/virgo_system_architecture.png'
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path} successfully without warnings!")

if __name__ == '__main__':
    create_diagram()
