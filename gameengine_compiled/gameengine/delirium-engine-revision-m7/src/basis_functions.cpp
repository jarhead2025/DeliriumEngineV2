#include "delirium_m7/basis_functions.hpp"
#include <stdexcept>

namespace delirium::m7 {

SurfacePoint BasisEvaluator::evaluate(const BodyState& state, const BodyTopology& topology,
                                       PatchId patch_id, float u, float v) const {
    const auto& patch = topology.patch(patch_id);
    SurfacePoint sp;
    sp.material = patch.material;
    sp.source_patch = patch_id;
    sp.u = u;
    sp.v = v;

    float bu00 = (1.0f - u) * (1.0f - v);
    float bu10 = u * (1.0f - v);
    float bu01 = (1.0f - u) * v;
    float bu11 = u * v;

    const auto& a00 = state.anchors[patch.anchors[0]];
    const auto& a10 = state.anchors[patch.anchors[1]];
    const auto& a01 = state.anchors[patch.anchors[2]];
    const auto& a11 = state.anchors[patch.anchors[3]];

    sp.position = a00.position * bu00 + a10.position * bu10 +
                  a01.position * bu01 + a11.position * bu11;

    Vec3 du = (a10.position - a00.position) * (1.0f - v) +
              (a11.position - a01.position) * v;
    Vec3 dv = (a01.position - a00.position) * (1.0f - u) +
              (a11.position - a10.position) * u;
    sp.normal = normalize_vec3(cross(du, dv));

    return sp;
}

float BasisEvaluator::compute_seam_error(const BodyState& state, const BodyTopology& topology,
                                          PatchId patch_a, PatchId patch_b) const {
    auto pa = evaluate(state, topology, patch_a, 1.0f, 0.5f);
    auto pb = evaluate(state, topology, patch_b, 0.0f, 0.5f);
    return length(pa.position - pb.position);
}

} // namespace delirium::m7