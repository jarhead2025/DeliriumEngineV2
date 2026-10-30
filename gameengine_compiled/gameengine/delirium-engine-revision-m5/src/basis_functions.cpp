#include "delirium_m5/basis_functions.hpp"
#include <cmath>

namespace delirium::m5 {

Vec3 evaluate_bilinear(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11, float u, float v) {
    return p00 * ((1.0f-u)*(1.0f-v)) + p10 * (u*(1.0f-v)) + p01 * ((1.0f-u)*v) + p11 * (u*v);
}

Vec3 evaluate_basis_displacement(BasisType basis, const std::array<float, 8>& coefficients, float u, float v) {
    float m1 = coefficients[4];
    switch (basis) {
        case BasisType::Bilinear: case BasisType::Plano: return Vec3{};
        case BasisType::Cupula: return {0.0f, 0.0f, m1 * std::sin(kPi*u) * std::sin(kPi*v)};
        case BasisType::Silla: return {0.0f, 0.0f, m1 * (u-0.5f) * (v-0.5f) * 4.0f};
        case BasisType::ValleU: return {0.0f, 0.0f, -m1 * std::sin(kPi*u) * 0.5f};
        case BasisType::ValleV: return {0.0f, 0.0f, -m1 * std::sin(kPi*v) * 0.5f};
        case BasisType::Eggbox: return {0.0f, 0.0f, m1 * std::sin(2.0f*kPi*u) * std::sin(2.0f*kPi*v) * 0.25f};
        case BasisType::Twist: {
            float t = m1 * (u-0.5f) * (v-0.5f);
            return {t*0.1f, -t*0.1f, 0.0f};
        }
        default: return Vec3{};
    }
}

Vec3 evaluate_residual_displacement(const std::array<Vec3, 4>& residuals, float u, float v) {
    return residuals[0]*((1.0f-u)*(1.0f-v)) + residuals[1]*(u*(1.0f-v)) +
           residuals[2]*((1.0f-u)*v) + residuals[3]*(u*v);
}

Vec3 evaluate_patch_position(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11,
                             BasisType basis, const std::array<float, 8>& coefficients,
                             const std::array<Vec3, 4>& residuals, float u, float v) {
    return evaluate_bilinear(p00,p10,p01,p11,u,v) +
           evaluate_basis_displacement(basis,coefficients,u,v) +
           evaluate_residual_displacement(residuals,u,v);
}

Vec3 evaluate_patch_normal(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11,
                           BasisType basis, const std::array<float, 8>& coefficients,
                           const std::array<Vec3, 4>& residuals, float u, float v) {
    float h = 0.001f;
    Vec3 tu = (evaluate_patch_position(p00,p10,p01,p11,basis,coefficients,residuals,u+h,v) -
               evaluate_patch_position(p00,p10,p01,p11,basis,coefficients,residuals,u-h,v)) * (0.5f/h);
    Vec3 tv = (evaluate_patch_position(p00,p10,p01,p11,basis,coefficients,residuals,u,v+h) -
               evaluate_patch_position(p00,p10,p01,p11,basis,coefficients,residuals,u,v-h)) * (0.5f/h);
    return normalize_vec3(cross(tu, tv));
}

} // namespace delirium::m5