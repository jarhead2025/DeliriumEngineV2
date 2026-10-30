#pragma once

#include "delirium_m2/body_state.hpp"
#include "delirium_m2/body_topology.hpp"

#include <string>
#include <vector>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// SurfaceEvaluator — bridges published BodyState into PSF/cage surface
// ---------------------------------------------------------------------------
//
// This is the canonical PSF/cage surface evaluator. It reads one published
// BodyState generation and evaluates surface points from the anchor positions
// and patch coefficients.
//
// The composition order is declared:
//   1. Bone pose displacement (articulation)
//   2. Modal displacement (global elastic modes)
//   3. Local material displacement (per-patch elastic/plastic)
//
// For M2, modal and local displacements are placeholders (zero). The bone
// pose displacement is the primary contributor.
// ---------------------------------------------------------------------------

struct SurfacePoint {
    Vec3 position;
    Vec3 normal;
    MaterialId material;
    PatchId source_patch;
    float u, v;           // parametric coordinates within the patch
};

class SurfaceEvaluator {
public:
    // Evaluate a single patch at parametric coordinates (u, v)
    SurfacePoint evaluate_patch(const BodyState& state,
                                const BodyTopology& topology,
                                PatchId patch_id,
                                float u, float v) const;

    // Evaluate a grid of points across all dirty patches
    std::vector<SurfacePoint> evaluate_dirty_patches(
        const BodyState& state,
        const BodyTopology& topology,
        const DirtySet& dirty_patches,
        std::uint32_t grid_resolution) const;

    // Compute seam error between two adjacent patches at their shared boundary
    float compute_seam_error(const BodyState& state,
                             const BodyTopology& topology,
                             PatchId patch_a, PatchId patch_b) const;

private:
    Vec3 evaluate_patch_position(const BodyState& state,
                                 const PatchTopology& patch,
                                 float u, float v) const;

    Vec3 evaluate_patch_normal(const BodyState& state,
                               const PatchTopology& patch,
                               float u, float v) const;
};

} // namespace delirium::m2