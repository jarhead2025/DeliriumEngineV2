#include "delirium_m2/surface_evaluator.hpp"

#include <algorithm>
#include <cmath>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// Bilinear patch evaluation
// ---------------------------------------------------------------------------
//
// A patch is a quadrilateral defined by 4 anchors. The bilinear interpolation
// provides a simple, continuous surface within each patch. M3 will upgrade
// this to higher-order PSF basis functions.
// ---------------------------------------------------------------------------

Vec3 SurfaceEvaluator::evaluate_patch_position(const BodyState& state,
                                                const PatchTopology& patch,
                                                float u, float v) const {
    // Clamp to valid parametric range
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);

    const Vec3& p00 = state.anchors[patch.anchors[0]].position;
    const Vec3& p10 = state.anchors[patch.anchors[1]].position;
    const Vec3& p01 = state.anchors[patch.anchors[2]].position;
    const Vec3& p11 = state.anchors[patch.anchors[3]].position;

    // Bilinear interpolation
    Vec3 p0 = p00 * (1.0f - u) + p10 * u;
    Vec3 p1 = p01 * (1.0f - u) + p11 * u;
    return p0 * (1.0f - v) + p1 * v;
}

Vec3 SurfaceEvaluator::evaluate_patch_normal(const BodyState& state,
                                              const PatchTopology& patch,
                                              float u, float v) const {
    // Finite-difference normal
    float eps = 1e-4f;
    Vec3 pos_u = evaluate_patch_position(state, patch, u + eps, v);
    Vec3 pos_u0 = evaluate_patch_position(state, patch, u - eps, v);
    Vec3 pos_v = evaluate_patch_position(state, patch, u, v + eps);
    Vec3 pos_v0 = evaluate_patch_position(state, patch, u, v - eps);

    Vec3 tangent_u = (pos_u - pos_u0) * 0.5f / eps;
    Vec3 tangent_v = (pos_v - pos_v0) * 0.5f / eps;

    return normalize(cross(tangent_u, tangent_v));
}

SurfacePoint SurfaceEvaluator::evaluate_patch(const BodyState& state,
                                               const BodyTopology& topology,
                                               PatchId patch_id,
                                               float u, float v) const {
    const PatchTopology& patch = topology.patch(patch_id);
    Vec3 pos = evaluate_patch_position(state, patch, u, v);
    Vec3 normal = evaluate_patch_normal(state, patch, u, v);

    return {pos, normal, patch.material, patch_id, u, v};
}

std::vector<SurfacePoint> SurfaceEvaluator::evaluate_dirty_patches(
    const BodyState& state,
    const BodyTopology& topology,
    const DirtySet& dirty_patches,
    std::uint32_t grid_resolution) const {

    std::vector<SurfacePoint> points;
    std::vector<std::uint32_t> dirty_indices = dirty_patches.indices();

    float step = 1.0f / grid_resolution;
    points.reserve(dirty_indices.size() * grid_resolution * grid_resolution);

    for (PatchId patch_id : dirty_indices) {
        for (std::uint32_t i = 0; i <= grid_resolution; ++i) {
            for (std::uint32_t j = 0; j <= grid_resolution; ++j) {
                float u = i * step;
                float v = j * step;
                points.push_back(evaluate_patch(state, topology, patch_id, u, v));
            }
        }
    }

    return points;
}

float SurfaceEvaluator::compute_seam_error(const BodyState& state,
                                            const BodyTopology& topology,
                                            PatchId patch_a, PatchId patch_b) const {
    const PatchTopology& pa = topology.patch(patch_a);
    const PatchTopology& pb = topology.patch(patch_b);

    // Find shared anchors (the seam)
    std::vector<std::pair<AnchorId, AnchorId>> shared;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (pa.anchors[i] == pb.anchors[j]) {
                shared.push_back({pa.anchors[i], pb.anchors[j]});
            }
        }
    }

    if (shared.empty()) return 0.0f; // no shared edge

    // For a bilinear patch, shared anchors are already at the same position
    // (they reference the same anchor in BodyState). The seam error is zero
    // by construction for shared-anchor topology.
    // In M3, we'll add non-shared seams (welded boundaries) that can have error.
    return 0.0f;
}

} // namespace delirium::m2