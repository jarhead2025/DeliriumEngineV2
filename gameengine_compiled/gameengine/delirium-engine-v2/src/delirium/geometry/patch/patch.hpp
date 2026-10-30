#pragma once
#include "delirium/math/vec3.hpp"
#include "delirium/basis/basis_functions.hpp"
#include "delirium/coefficient/records.hpp"
#include <array>
#include <span>

namespace delirium {

/// ============================================================================
/// PATCH — the fundamental geometry primitive
/// ============================================================================
///
/// A patch is a quadrilateral surface region defined by:
/// - 4 anchor indices (shared with neighboring patches at boundaries)
/// - Surface coefficients (Σ Kᵢ · typeᵢ for shape, detail, color)
/// - Material ID (which material region this patch belongs to)
///
/// Surface evaluation at (u,v):
///   1. Bilinear interpolation of 4 anchor positions → base position
///   2. Shape layer: Σ K_shape · type_shape(u,v) → displacement along normal
///   3. Detail layer: Σ K_detail · type_detail(u,v) → high-frequency relief
///   4. Color layer: Σ K_color · type_color(u,v) → RGB
///
/// The bilinear base ensures corner cracks are impossible — the surface
/// always interpolates its anchors regardless of coefficient values.
/// ============================================================================

class Patch {
public:
    /// 4 corner anchor indices (shared with neighbors)
    std::array<uint32_t, 4> anchor_indices;

    /// Surface coefficients
    SurfaceRecord surface;

    /// Material region ID
    uint32_t material_id;

    /// Patch ID for identification
    uint32_t patch_id;

    /// Generation ID — increments when coefficients change (dirty tracking)
    uint32_t generation_id;

    Patch()
        : anchor_indices{0, 0, 0, 0}, surface{},
          material_id(0), patch_id(0), generation_id(0) {}

    /// --- Bilinear interpolation of anchor positions ---
    ///
    /// P_bilinear(u,v) = (1-u)(1-v)·A₀ + u(1-v)·A₁ + (1-u)v·A₂ + uv·A₃
    ///
    /// This is the base position. Shape/detail layers add displacement.
    vec3 evaluate_bilinear(float u, float v,
                           const std::span<const AnchorRecord>& anchors) const;

    /// --- Normal derivation ---
    ///
    /// Computed from finite differences of the surface, NOT stored.
    /// This is structurally superior: the normal cannot contradict the surface.
    vec3 evaluate_normal(float u, float v,
                         const std::span<const AnchorRecord>& anchors) const;

    /// --- Full surface evaluation ---
    ///
    /// Returns world position + normal + color at (u,v).
    /// This is the function that both physics AND rendering call.
    SurfaceSample evaluate(float u, float v,
                           const std::span<const AnchorRecord>& anchors) const;

    /// --- Hit test ---
    ///
    /// Given a world position, is there surface here?
    /// Used by BOTH ray marching (render) and collision detection (physics).
    /// Returns distance to surface (0 = hit) and the hit (u,v) parameters.
    bool test_hit(const vec3& world_pos,
                  const std::span<const AnchorRecord>& anchors,
                  float& out_u, float& out_v, float& out_distance) const;

    /// --- AABB bounds ---
    ///
    /// Axis-aligned bounding box for frustum culling and spatial grid.
    struct AABB {
        vec3 min;
        vec3 max;
    };
    AABB compute_bounds(const std::span<const AnchorRecord>& anchors) const;

    /// Mark this patch as dirty (coefficients changed)
    void dirty() { generation_id++; surface.generation_id = generation_id; }
};

} // namespace delirium