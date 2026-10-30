#pragma once

#include "types.h"
#include "body_topology.h"
#include "body_state.h"
#include <string>

namespace delirium {

struct SurfacePoint {
    Vec3 position, normal;
    MaterialId material;
    PatchId source_patch;
    float u, v;
};

class BasisEvaluator {
public:
    /* Evaluate position, normal, and colour at (u,v) on a patch.
     * Uses the 8-term corner-vanishing basis from the proven kernel. */
    SurfacePoint evaluate(const BodyState& state, const BodyTopology& topology,
                          PatchId patch_id, float u, float v) const;

    /* Measure seam gap between two adjacent patches (as fraction of bbox diag). */
    float compute_seam_error(const BodyState& state, const BodyTopology& topology,
                             PatchId patch_a, PatchId patch_b) const;

public:
    /* 8-term corner-vanishing basis (FMA chain, order matters).
     * Verbatim mirror of kernels/psf_splat.cl:70-79 and Patch.cpp. */
    static void deform_basis(float u, float v, float b[8]);
};

} // namespace delirium