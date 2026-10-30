#pragma once
#include "delirium/math/vec3.hpp"
#include <cmath>
#include <array>
#include <span>

namespace delirium {

/// ============================================================================
/// BASIS FUNCTIONS — Σ Kᵢ · typeᵢ(u,v)
/// ============================================================================
///
/// Every surface is a weighted sum of basis types. The types are simple
/// mathematical functions evaluated at patch parameters (u,v) ∈ [0,1]².
///
/// Layer 1 (shape): low-frequency bases that define silhouette
/// Layer 2 (detail): high-frequency bases that add relief
/// Layer 3 (color): RGB spectral field
///
/// The same mechanism — weighted sum of types — applies to ALL layers.
/// ============================================================================

/// A basis function: evaluates to a scalar at (u,v)
using BasisFn = float(*)(float u, float v);

/// --- Layer 1: Shape bases (low frequency, orders 0-2) ---
///
/// These define the silhouette. Stable, smooth, no oscillations.
/// Cannot explode because there are no high bands to inject energy.

/// Bilinear corner bases — the patch interpolates its 4 anchors regardless
/// of coefficients. Corner cracks are impossible by construction.
inline float shape_base_00(float u, float v) { return (1.0f - u) * (1.0f - v); }  // corner 0
inline float shape_base_01(float u, float v) { return u * (1.0f - v); }           // corner 1
inline float shape_base_10(float u, float v) { return (1.0f - u) * v; }           // corner 2
inline float shape_base_11(float u, float v) { return u * v; }                     // corner 3

/// Low-frequency shape deformation — smooth bulging, no high-frequency artifacts
inline float shape_base_bulge(float u, float v) { return std::sin(M_PI * u) * std::sin(M_PI * v); }
inline float shape_base_saddle(float u, float v) { return (u - 0.5f) * (v - 0.5f); }
inline float shape_base_twist(float u, float v) { return std::sin(M_PI * (u + v)); }
inline float shape_base_cylinder(float u, float v) { return std::cos(M_PI * u); }

/// Shape basis registry (8 low-frequency bases)
constexpr std::array<BasisFn, 8> SHAPE_BASES = {
    shape_base_00,   // 0: corner bilinear 0
    shape_base_01,   // 1: corner bilinear 1
    shape_base_10,   // 2: corner bilinear 2
    shape_base_11,   // 3: corner bilinear 3
    shape_base_bulge,// 4: smooth bulge
    shape_base_saddle,// 5: saddle
    shape_base_twist, // 6: twist
    shape_base_cylinder // 7: cylindrical
};

/// --- Layer 2: Detail bases (high frequency, 2D Fourier on residuals) ---
///
/// These add relief/detail on top of the shape layer.
/// Evaluated as: detail(u,v) = Σ Kⱼ · detail_baseⱼ(u,v)

inline float detail_base_0(float u, float v) { return std::cos(2.0f * M_PI * u); }
inline float detail_base_1(float u, float v) { return std::sin(2.0f * M_PI * u); }
inline float detail_base_2(float u, float v) { return std::cos(2.0f * M_PI * v); }
inline float detail_base_3(float u, float v) { return std::sin(2.0f * M_PI * v); }
inline float detail_base_4(float u, float v) { return std::cos(4.0f * M_PI * u); }
inline float detail_base_5(float u, float v) { return std::sin(4.0f * M_PI * u); }
inline float detail_base_6(float u, float v) { return std::cos(4.0f * M_PI * v); }
inline float detail_base_7(float u, float v) { return std::sin(4.0f * M_PI * v); }

constexpr std::array<BasisFn, 8> DETAIL_BASES = {
    detail_base_0, detail_base_1, detail_base_2, detail_base_3,
    detail_base_4, detail_base_5, detail_base_6, detail_base_7
};

/// --- Layer 3: Color bases (RGB spectral field, DCT-like) ---
///
/// Color is a sum of types too: color(u,v) = Σ Kₖ · color_baseₖ(u,v)
/// Each Kₖ is a vec3 (R,G,B coefficient)

inline float color_base_0(float, float) { return 1.0f; }  // DC component (base color)
inline float color_base_1(float u, float) { return std::cos(M_PI * u); }
inline float color_base_2(float, float v) { return std::cos(M_PI * v); }
inline float color_base_3(float u, float) { return std::cos(2.0f * M_PI * u); }
inline float color_base_4(float, float v) { return std::cos(2.0f * M_PI * v); }
inline float color_base_5(float u, float v) { return std::cos(M_PI * (u + v)); }
inline float color_base_6(float u, float v) { return std::sin(M_PI * (u + v)); }
inline float color_base_7(float u, float v) { return std::cos(M_PI * (u - v)); }

constexpr std::array<BasisFn, 8> COLOR_BASES = {
    color_base_0, color_base_1, color_base_2, color_base_3,
    color_base_4, color_base_5, color_base_6, color_base_7
};

/// ============================================================================
/// EVALUATION — Σ Kᵢ · typeᵢ(u,v) for each layer
/// ============================================================================

/// Evaluate shape layer: returns scalar displacement from bilinear base
inline float evaluate_shape(float u, float v, const std::span<const float, 8>& K) {
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i)
        sum += K[i] * SHAPE_BASES[i](u, v);
    return sum;
}

/// Evaluate detail layer: returns scalar relief on top of shape
inline float evaluate_detail(float u, float v, const std::span<const float, 8>& K) {
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i)
        sum += K[i] * DETAIL_BASES[i](u, v);
    return sum;
}

/// Evaluate color layer: returns vec3 (R,G,B)
inline vec3 evaluate_color(float u, float v, const std::span<const float, 8>& K) {
    vec3 color;
    for (int i = 0; i < 8; ++i) {
        float basis = COLOR_BASES[i](u, v);
        color.x += K[i * 3 + 0] * basis;  // R
        color.y += K[i * 3 + 1] * basis;  // G
        color.z += K[i * 3 + 2] * basis;  // B
    }
    return color;
}

/// Evaluate all layers at once
struct SurfaceSample {
    vec3 position;    // world position from shape + bilinear anchors
    vec3 normal;      // derived from surface geometry
    vec3 color;       // from color layer
    float shape_val;  // shape layer scalar
    float detail_val; // detail layer scalar
};

} // namespace delirium