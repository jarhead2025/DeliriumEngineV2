#include "delirium_m6/basis_functions.hpp"
#include <cmath>
namespace delirium::m6 {
Vec3 evaluate_bilinear(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11, float u, float v) {
    return p00*((1-u)*(1-v)) + p10*(u*(1-v)) + p01*((1-u)*v) + p11*(u*v);
}
Vec3 evaluate_basis_displacement(BasisType basis, const std::array<float, 8>& c, float u, float v) {
    float m1 = c[4];
    switch (basis) {
        case BasisType::Bilinear: case BasisType::Plano: return Vec3{};
        case BasisType::Cupula: return {0,0,m1*std::sin(kPi*u)*std::sin(kPi*v)};
        case BasisType::Silla: return {0,0,m1*(u-0.5f)*(v-0.5f)*4.0f};
        case BasisType::ValleU: return {0,0,-m1*std::sin(kPi*u)*0.5f};
        case BasisType::ValleV: return {0,0,-m1*std::sin(kPi*v)*0.5f};
        case BasisType::Eggbox: return {0,0,m1*std::sin(2*kPi*u)*std::sin(2*kPi*v)*0.25f};
        case BasisType::Twist: { float t=m1*(u-0.5f)*(v-0.5f); return {t*0.1f,-t*0.1f,0}; }
        default: return Vec3{};
    }
}
Vec3 evaluate_residual_displacement(const std::array<Vec3, 4>& r, float u, float v) {
    return r[0]*((1-u)*(1-v)) + r[1]*(u*(1-v)) + r[2]*((1-u)*v) + r[3]*(u*v);
}
Vec3 evaluate_patch_position(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11,
                             BasisType basis, const std::array<float, 8>& c,
                             const std::array<Vec3, 4>& r, float u, float v) {
    return evaluate_bilinear(p00,p10,p01,p11,u,v) + evaluate_basis_displacement(basis,c,u,v) + evaluate_residual_displacement(r,u,v);
}
Vec3 evaluate_patch_normal(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11,
                           BasisType basis, const std::array<float, 8>& c,
                           const std::array<Vec3, 4>& r, float u, float v) {
    float h = 0.001f;
    Vec3 tu = (evaluate_patch_position(p00,p10,p01,p11,basis,c,r,u+h,v) - evaluate_patch_position(p00,p10,p01,p11,basis,c,r,u-h,v)) * (0.5f/h);
    Vec3 tv = (evaluate_patch_position(p00,p10,p01,p11,basis,c,r,u,v+h) - evaluate_patch_position(p00,p10,p01,p11,basis,c,r,u,v-h)) * (0.5f/h);
    return normalize_vec3(cross(tu, tv));
}
} // namespace delirium::m6