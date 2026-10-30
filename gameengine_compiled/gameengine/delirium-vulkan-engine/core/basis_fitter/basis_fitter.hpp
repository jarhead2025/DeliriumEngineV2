#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace delirium {

/// 8 basis surface types from the original Delirium engine
enum class BasisType : std::uint8_t {
    Bilinear = 0,  /// Standard bilinear interpolation
    Plano = 1,     /// Planar (flat) surface
    Cupula = 2,    /// Dome/cup shape (bulges outward)
    Silla = 3,     /// Saddle shape (hyperbolic)
    ValleU = 4,    /// Valley along U axis
    ValleV = 5,    /// Valley along V axis
    Eggbox = 6,    /// Egg-crate (double saddle)
    Twist = 7,     /// Twisted surface
};

/// Coefficients for higher-order basis evaluation
struct PatchCoefficients {
    std::array<float, 8> scalars{};     /// 8 scalar coefficients for basis blend
    std::array<std::array<float, 3>, 4> residuals{};  /// 4 residual correction vectors (one per corner)
    
    bool has_residuals() const {
        for (const auto& r : residuals) {
            float len = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
            if (len > 1e-12f) return true;
        }
        return false;
    }
};

/// Result of fitting a basis to a quad
struct BasisFitResult {
    BasisType type = BasisType::Bilinear;
    PatchCoefficients coefficients;
    float error = 1e10f;  /// RMS error of the fit
};

/// Vertex with position and UV for basis fitting
struct FitVertex {
    float x, y, z;
    float u, v;
};

/// Fit the best basis type to a quad defined by 4 vertices.
/// Samples the quad at a grid, evaluates each basis type, and picks
/// the one with lowest RMS error.
BasisFitResult fit_basis(const FitVertex& A, const FitVertex& B,
                         const FitVertex& C, const FitVertex& D,
                         int sample_resolution = 8);

/// Evaluate a surface point using the fitted basis.
/// Returns the 3D position at (u, v) in patch-local coordinates.
std::array<float, 3> evaluate_basis(const FitVertex& A, const FitVertex& B,
                                     const FitVertex& C, const FitVertex& D,
                                     BasisType type, const PatchCoefficients& coeffs,
                                     float u, float v);

} // namespace delirium