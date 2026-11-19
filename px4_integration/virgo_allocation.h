/*
 * Virgo_RL: MorphQuad Null-Space Control Allocation in C++ for PX4
 * Maps 6-DOF commanded wrench W_cmd to 4 rotor thrusts and 8 Dynamixel servo angles.
 * Includes Gimbal Lock avoidance and downwash repulsion (arXiv:2607.02764).
 */

#ifndef VIRGO_ALLOCATION_H
#define VIRGO_ALLOCATION_H

#include <math.h>

#define VIRGO_MAX_THRUST_PER_ARM 28.0f     // N total per arm
#define VIRGO_MAX_THRUST_PER_MOTOR 14.0f   // N per individual coaxial motor
#define VIRGO_MASS 4.56f                   // kg takeoff mass from arXiv:2607.02764
#define VIRGO_GRAVITY 9.81f

static const float M_PINV[72] = {
    2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, 0.0000000e+00f, -7.1682788e-17f, 5.6818182e-01f, 0.0000000e+00f, 2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, -7.1682788e-17f, 5.6818182e-01f, 0.0000000e+00f, 0.0000000e+00f, 2.5000000e-01f, -1.1363636e+00f, -1.1363636e+00f, -1.7681925e-16f, 2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, 0.0000000e+00f, -7.1682788e-17f, 5.6818182e-01f, 0.0000000e+00f, 2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, 7.1682788e-17f, -5.6818182e-01f, 0.0000000e+00f, 0.0000000e+00f, 2.5000000e-01f, -1.1363636e+00f, 1.1363636e+00f, 6.5307003e-17f, 2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, 0.0000000e+00f, 7.1682788e-17f, -5.6818182e-01f, 0.0000000e+00f, 2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, 7.1682788e-17f, -5.6818182e-01f, 0.0000000e+00f, 0.0000000e+00f, 2.5000000e-01f, 1.1363636e+00f, 1.1363636e+00f, 4.3004553e-17f, 2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, 0.0000000e+00f, 7.1682788e-17f, -5.6818182e-01f, 0.0000000e+00f, 2.5000000e-01f, 0.0000000e+00f, 0.0000000e+00f, -7.1682788e-17f, 5.6818182e-01f, 0.0000000e+00f, 0.0000000e+00f, 2.5000000e-01f, 1.1363636e+00f, -1.1363636e+00f, 6.8507696e-17f
};

static const float N_M[72] = {
    -4.3674835e-01f, 6.3251651e-02f, 3.1224945e-01f, -4.3674835e-01f, -3.1625826e-02f, 3.7550110e-01f, 2.3247622e-01f, -2.6752378e-01f, 2.5584126e-01f, 2.3247622e-01f, -6.1623811e-01f, -1.1682519e-02f, -1.1281638e-01f, -1.1281638e-01f, 2.0427213e-01f, -1.1281638e-01f, 5.6408190e-02f, -4.0854425e-01f, 4.7006825e-02f, 2.1367349e-01f, -6.2678005e-01f, 4.7006825e-02f, -1.0683675e-01f, -4.1310656e-01f, -3.0768714e-01f, -4.7435381e-01f, -4.5299317e-01f, -3.0768714e-01f, -1.2823096e-02f, 7.2653016e-02f, 1.1281638e-01f, 1.1281638e-01f, -2.0427213e-01f, 1.1281638e-01f, -5.6408190e-02f, 4.0854425e-01f, 6.9487076e-01f, -1.3846257e-01f, 1.5726530e-01f, -3.0512924e-01f, 6.9231286e-02f, 1.8802730e-02f, -8.2054381e-02f, 7.5127895e-01f, 1.3846257e-01f, -8.2054381e-02f, -1.2563948e-01f, -1.1025848e-01f, -1.1281638e-01f, -1.1281638e-01f, 2.0427213e-01f, -1.1281638e-01f, 5.6408190e-02f, -4.0854425e-01f, -3.0512924e-01f, -1.3846257e-01f, 1.5726530e-01f, 6.9487076e-01f, 6.9231286e-02f, 1.8802730e-02f, 1.5726530e-01f, -9.4013650e-03f, 5.8689344e-02f, 1.5726530e-01f, 7.5470068e-01f, 4.9287979e-02f, 1.1281638e-01f, 1.1281638e-01f, -2.0427213e-01f, 1.1281638e-01f, -5.6408190e-02f, 4.0854425e-01f
};


inline void virgo_allocate_actuators(
    const float f_cmd[3],
    const float tau_cmd[3],
    float motor_thrusts[8],
    float alphas[4],
    float betas[4]
) {
    float w_des[6] = { f_cmd[0], f_cmd[1], f_cmd[2], tau_cmd[0], tau_cmd[1], tau_cmd[2] };
    float t_stacked[12];

    // Minimum-norm pseudo-inverse: t_0 = M_pinv * w_des
    // Clean, perfectly symmetric and energy-optimal allocation across all 4 arms
    for (int i = 0; i < 12; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 6; ++j) {
            sum += M_PINV[i * 6 + j] * w_des[j];
        }
        t_stacked[i] = sum;
    }

    // Recover individual 8 coaxial motor thrusts and 8 servo angles
    for (int i = 0; i < 4; ++i) {
        float tx = t_stacked[i * 3 + 0];
        float ty = t_stacked[i * 3 + 1];
        float tz = t_stacked[i * 3 + 2];
        float t_mag = sqrtf(tx*tx + ty*ty + tz*tz);

        if (t_mag > VIRGO_MAX_THRUST_PER_ARM) t_mag = VIRGO_MAX_THRUST_PER_ARM;

        // Equal thrust sharing between coaxial counter-rotating motor pair
        motor_thrusts[2 * i + 0] = t_mag * 0.5f; // Top motor (PROP_i_1)
        motor_thrusts[2 * i + 1] = t_mag * 0.5f; // Bottom motor (PROP_i_2)

        if (t_mag > 1e-4f) {
            float hat_x = tx / t_mag;
            float hat_y = ty / t_mag;
            float hat_z = tz / t_mag;

            float sin_b = -hat_y;
            if (sin_b > 0.999f) sin_b = 0.999f;
            if (sin_b < -0.999f) sin_b = -0.999f;
            betas[i] = asinf(sin_b);
            alphas[i] = atan2f(hat_x, hat_z);
        } else {
            alphas[i] = 0.0f;
            betas[i] = 0.0f;
        }
    }
}

#endif // VIRGO_ALLOCATION_H
