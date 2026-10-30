#include "delirium_m3/basis_functions.hpp"

#include <cmath>

namespace delirium::m3 {

// ---------------------------------------------------------------------------
// Bilinear base evaluation
// ---------------------------------------------------------------------------
Vec3 evaluate_bilinear(const Vec3& p00, const Vec3& p10,
                       const Vec3& p01, const Vec3& p11,
                       float u, float v) {
    // Clamp u,v to [0,1]
    if (u < 0.0f) u = 0.0f;
    if (u > 1.0f) u = 1.0f;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    // Bilinear interpolation
    Vec3 a = p00 * (1.0f - u) + p10 * u;           // bottom edge
    Vec3 b = p01 * (1.0f - u) + p11 * u;           // top edge
    return a * (1.0f - v) + b * v;
}

// ---------------------------------------------------------------------------
// Higher-order basis displacement
// ---------------------------------------------------------------------------
Vec3 evaluate_basis_displacement(BasisType basis,
                                 const std::array<float, 8>& coefficients,
                                 float u, float v) {
    Vec3 displacement{};

    // The higher-order modes modify the z-component (surface height)
    // relative to the bilinear base plane. The displacement is computed
    // in the patch's local frame — the caller adds it to the bilinear position.
    //
    // For M3, we compute displacement along the surface normal direction.
    // The actual normal direction is applied by the surface evaluator.

    float height = 0.0f;

    switch (basis) {
        case BasisType::Bilinear:
            // No higher-order displacement
            break;

        case BasisType::Plano:
            // Flat plane with higher-order correction
            // m1-m4 add subtle curvature
            height = coefficients[4] * u * v +
                     coefficients[5] * u * u * (1.0f - v) +
                     coefficients[6] * (1.0f - u) * v * v +
                     coefficients[7] * u * v * (1.0f - u - v + u * v);
            break;

        case BasisType::Cupula:
            // Dome/cap: maximum at center, zero at edges
            height = coefficients[4] * 4.0f * u * v * (1.0f - u) * (1.0f - v);
            height += coefficients[5] * std::sin(kPi * u) * std::sin(kPi * v);
            break;

        case BasisType::Silla:
            // Saddle: positive in two corners, negative in the other two
            height = coefficients[4] * (2.0f * u - 1.0f) * (2.0f * v - 1.0f);
            height += coefficients[5] * std::sin(kPi * u) * std::cos(kPi * v);
            break;

        case BasisType::ValleU:
            // Valley along u direction
            height = -coefficients[4] * 4.0f * v * (1.0f - v);
            height += coefficients[5] * std::sin(kPi * v);
            break;

        case BasisType::ValleV:
            // Valley along v direction
            height = -coefficients[4] * 4.0f * u * (1.0f - u);
            height += coefficients[5] * std::sin(kPi * u);
            break;

        case BasisType::Eggbox:
            // Egg-crate: double saddle
            height = coefficients[4] * std::sin(kPi * u) * std::sin(kPi * v);
            height += coefficients[5] * std::sin(2.0f * kPi * u) * std::sin(2.0f * kPi * v) * 0.5f;
            break;

        case BasisType::Twist:
            // Twisted surface
            height = coefficients[4] * u * v * (2.0f * u - 1.0f) * (2.0f * v - 1.0f);
            height += coefficients[5] * std::sin(kPi * (u + v) * 0.5f);
            break;
    }

    // Height displacement is along the patch normal (z-axis in local frame)
    displacement.z = height;
    return displacement;
}

// ---------------------------------------------------------------------------
// Vector residual displacement
// ---------------------------------------------------------------------------
Vec3 evaluate_residual_displacement(const std::array<Vec3, 4>& residuals,
                                    float u, float v) {
    // Bilinear interpolation of corner residuals
    Vec3 r00 = residuals[0];
    Vec3 r10 = residuals[1];
    Vec3 r01 = residuals[2];
    Vec3 r11 = residuals[3];

    Vec3 a = r00 * (1.0f - u) + r10 * u;
    Vec3 b = r01 * (1.0f - u) + r11 * u;
    return a * (1.0f - v) + b * v;
}

// ---------------------------------------------------------------------------
// Full patch position: bilinear + basis displacement + residuals
// ---------------------------------------------------------------------------
Vec3 evaluate_patch_position(const Vec3& p00, const Vec3& p10,
                             const Vec3& p01, const Vec3& p11,
                             BasisType basis,
                             const std::array<float, 8>& coefficients,
                             const std::array<Vec3, 4>& residuals,
                             float u, float v) {
    Vec3 base = evaluate_bilinear(p00, p10, p01, p11, u, v);

    // Basis displacement is in local z (along surface normal)
    Vec3 basis_disp = evaluate_basis_displacement(basis, coefficients, u, v);

    // To apply basis displacement along the surface normal, compute the
    // approximate normal from the bilinear tangent vectors
    Vec3 tan_u = (evaluate_bilinear(p00, p10, p01, p11, u + 0.001f, v) -
                  evaluate_bilinear(p00, p10, p01, p11, u - 0.001f, v)) * 500.0f;
    Vec3 tan_v = (evaluate_bilinear(p00, p10, p01, p11, u, v + 0.001f) -
                  evaluate_bilinear(p00, p10, p01, p11, u, v - 0.001f)) * 500.0f;

    Vec3 normal = cross(tan_u, tan_v);
    float nl = length(normal);
    if (nl > 1e-8f) {
        normal = normal * (1.0f / nl);
    } else {
        normal = Vec3{0.0f, 0.0f, 1.0f};
    }

    // Apply basis displacement along the normal
    base = base + normal * basis_disp.z;

    // Add vector residuals
    Vec3 residual_disp = evaluate_residual_displacement(residuals, u, v);
    base = base + residual_disp;

    return base;
}

// ---------------------------------------------------------------------------
// Finite-difference normal
// ---------------------------------------------------------------------------
Vec3 evaluate_patch_normal(const Vec3& p00, const Vec3& p10,
                           const Vec3& p01, const Vec3& p11,
                           BasisType basis,
                           const std::array<float, 8>& coefficients,
                           const std::array<Vec3, 4>& residuals,
                           float u, float v) {
    float eps = 0.001f;

    Vec3 pos_u_p = evaluate_patch_position(p00, p10, p01, p11, basis,
                                           coefficients, residuals, u + eps, v);
    Vec3 pos_u_m = evaluate_patch_position(p00, p10, p01, p11, basis,
                                           coefficients, residuals, u - eps, v);
    Vec3 pos_v_p = evaluate_patch_position(p00, p10, p01, p11, basis,
                                           coefficients, residuals, u, v + eps);
    Vec3 pos_v_m = evaluate_patch_position(p00, p10, p01, p11, basis,
                                           coefficients, residuals, u, v - eps);

    Vec3 tangent_u = (pos_u_p - pos_u_m) * 0.5f / eps;
    Vec3 tangent_v = (pos_v_p - pos_v_m) * 0.5f / eps;

    Vec3 normal = cross(tangent_u, tangent_v);
    float nl = length(normal);
    return nl > 1e-8f ? normal * (1.0f / nl) : Vec3{0.0f, 0.0f, 1.0f};
}

} // namespace delirium::m3