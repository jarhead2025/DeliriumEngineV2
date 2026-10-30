#include "delirium_m4/basis_functions.hpp"

#include <cmath>

namespace delirium::m4 {

Vec3 evaluate_bilinear(const Vec3& p00, const Vec3& p10,
                       const Vec3& p01, const Vec3& p11,
                       float u, float v) {
    float w00 = (1.0f - u) * (1.0f - v);
    float w10 = u * (1.0f - v);
    float w01 = (1.0f - u) * v;
    float w11 = u * v;

    return p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;
}

Vec3 evaluate_basis_displacement(BasisType basis,
                                 const std::array<float, 8>& coefficients,
                                 float u, float v) {
    // Corner heights (c00-c11) are handled in bilinear base
    // Higher-order modes (m1-m4, indices 4-7) add displacement

    float m1 = coefficients[4];

    switch (basis) {
        case BasisType::Bilinear:
        case BasisType::Plano:
            return Vec3{};

        case BasisType::Cupula: {
            // Dome shape: sin-based bump centered at (0.5, 0.5)
            float h = m1 * std::sin(kPi * u) * std::sin(kPi * v);
            return Vec3{0.0f, 0.0f, h};
        }

        case BasisType::Silla: {
            // Saddle: hyperbolic shape
            float h = m1 * (u - 0.5f) * (v - 0.5f) * 4.0f;
            return Vec3{0.0f, 0.0f, h};
        }

        case BasisType::ValleU: {
            // Valley along u direction
            float h = -m1 * std::sin(kPi * u) * 0.5f;
            return Vec3{0.0f, 0.0f, h};
        }

        case BasisType::ValleV: {
            // Valley along v direction
            float h = -m1 * std::sin(kPi * v) * 0.5f;
            return Vec3{0.0f, 0.0f, h};
        }

        case BasisType::Eggbox: {
            // Egg-crate: double saddle
            float h = m1 * std::sin(2.0f * kPi * u) * std::sin(2.0f * kPi * v) * 0.25f;
            return Vec3{0.0f, 0.0f, h};
        }

        case BasisType::Twist: {
            // Twisted surface: lateral displacement
            float twist = m1 * (u - 0.5f) * (v - 0.5f);
            return Vec3{twist * 0.1f, -twist * 0.1f, 0.0f};
        }

        default:
            return Vec3{};
    }
}

Vec3 evaluate_residual_displacement(const std::array<Vec3, 4>& residuals,
                                    float u, float v) {
    float w00 = (1.0f - u) * (1.0f - v);
    float w10 = u * (1.0f - v);
    float w01 = (1.0f - u) * v;
    float w11 = u * v;

    return residuals[0] * w00 + residuals[1] * w10 +
           residuals[2] * w01 + residuals[3] * w11;
}

Vec3 evaluate_patch_position(const Vec3& p00, const Vec3& p10,
                             const Vec3& p01, const Vec3& p11,
                             BasisType basis,
                             const std::array<float, 8>& coefficients,
                             const std::array<Vec3, 4>& residuals,
                             float u, float v) {
    // 1. Bilinear base
    Vec3 pos = evaluate_bilinear(p00, p10, p01, p11, u, v);

    // 2. Higher-order mode displacement
    pos += evaluate_basis_displacement(basis, coefficients, u, v);

    // 3. Vector residual displacement
    pos += evaluate_residual_displacement(residuals, u, v);

    return pos;
}

Vec3 evaluate_patch_normal(const Vec3& p00, const Vec3& p10,
                           const Vec3& p01, const Vec3& p11,
                           BasisType basis,
                           const std::array<float, 8>& coefficients,
                           const std::array<Vec3, 4>& residuals,
                           float u, float v) {
    float h = 0.001f;

    // Finite difference: tangent_u = dP/du, tangent_v = dP/dv
    Vec3 pu_plus = evaluate_patch_position(p00, p10, p01, p11, basis,
                                            coefficients, residuals,
                                            u + h, v);
    Vec3 pu_minus = evaluate_patch_position(p00, p10, p01, p11, basis,
                                             coefficients, residuals,
                                             u - h, v);
    Vec3 pv_plus = evaluate_patch_position(p00, p10, p01, p11, basis,
                                            coefficients, residuals,
                                            u, v + h);
    Vec3 pv_minus = evaluate_patch_position(p00, p10, p01, p11, basis,
                                             coefficients, residuals,
                                             u, v - h);

    Vec3 tangent_u = (pu_plus - pu_minus) * (0.5f / h);
    Vec3 tangent_v = (pv_plus - pv_minus) * (0.5f / h);

    return normalize_vec3(cross(tangent_u, tangent_v));
}

} // namespace delirium::m4