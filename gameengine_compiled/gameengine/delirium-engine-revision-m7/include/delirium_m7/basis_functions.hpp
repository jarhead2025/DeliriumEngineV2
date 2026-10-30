#pragma once
#include "delirium_m7/types.hpp"
#include "delirium_m7/body_topology.hpp"
#include "delirium_m7/body_state.hpp"
#include <string>
namespace delirium::m7 {
struct SurfacePoint {
    Vec3 position, normal;
    MaterialId material;
    PatchId source_patch;
    float u, v;
};
class BasisEvaluator {
public:
    SurfacePoint evaluate(const BodyState& state, const BodyTopology& topology,
                          PatchId patch_id, float u, float v) const;
    float compute_seam_error(const BodyState& state, const BodyTopology& topology,
                             PatchId patch_a, PatchId patch_b) const;
};
} // namespace delirium::m7