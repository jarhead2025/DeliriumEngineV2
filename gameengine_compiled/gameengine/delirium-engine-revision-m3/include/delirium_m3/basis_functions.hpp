#pragma once

#include "delirium_m3/types.hpp"

#include <array>
#include <vector>

namespace delirium::m3 {

// ---------------------------------------------------------------------------
// BasisEvaluator — higher-order PSF basis functions
// ---------------------------------------------------------------------------
//
// Evaluates surface position and normal for a patch using the specified basis
// function type. The bilinear case is the M2 baseline; higher-order bases
// add lateral geometric detail through the 8 scalar coefficients.
//
// Composition order:
//   1. Bilinear base from anchor positions
//   2. Higher-order mode displacement from scalar coefficients
//   3. Vector residual displacement (sub-anchor detail)
// ---------------------------------------------------------------------------

// Evaluate the bilinear base position from 4 anchor positions at (u,v)
Vec3 evaluate_bilinear(const Vec3& p00, const Vec3& p10,
                       const Vec3& p01, const Vec3& p11,
                       float u, float v);

// Evaluate the higher-order mode displacement for a given basis type
// Returns a displacement vector to add to the bilinear base
Vec3 evaluate_basis_displacement(BasisType basis,
                                 const std::array<float, 8>& coefficients,
                                 float u, float v);

// Evaluate vector residual displacement at (u,v)
Vec3 evaluate_residual_displacement(const std::array<Vec3, 4>& residuals,
                                    float u, float v);

// Full surface position: bilinear + basis displacement + residuals
Vec3 evaluate_patch_position(const Vec3& p00, const Vec3& p10,
                             const Vec3& p01, const Vec3& p11,
                             BasisType basis,
                             const std::array<float, 8>& coefficients,
                             const std::array<Vec3, 4>& residuals,
                             float u, float v);

// Finite-difference normal at (u,v)
Vec3 evaluate_patch_normal(const Vec3& p00, const Vec3& p10,
                           const Vec3& p01, const Vec3& p11,
                           BasisType basis,
                           const std::array<float, 8>& coefficients,
                           const std::array<Vec3, 4>& residuals,
                           float u, float v);

} // namespace delirium::m3