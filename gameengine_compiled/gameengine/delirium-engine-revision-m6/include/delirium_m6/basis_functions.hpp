#pragma once
#include "delirium_m6/types.hpp"
#include <array>
namespace delirium::m6 {
Vec3 evaluate_bilinear(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11, float u, float v);
Vec3 evaluate_basis_displacement(BasisType basis, const std::array<float, 8>& coefficients, float u, float v);
Vec3 evaluate_residual_displacement(const std::array<Vec3, 4>& residuals, float u, float v);
Vec3 evaluate_patch_position(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11,
                             BasisType basis, const std::array<float, 8>& coefficients,
                             const std::array<Vec3, 4>& residuals, float u, float v);
Vec3 evaluate_patch_normal(const Vec3& p00, const Vec3& p10, const Vec3& p01, const Vec3& p11,
                           BasisType basis, const std::array<float, 8>& coefficients,
                           const std::array<Vec3, 4>& residuals, float u, float v);
} // namespace delirium::m6