#include "delirium_m6/welded_seam.hpp"
#include "delirium_m6/basis_functions.hpp"
#include <algorithm>
#include <cmath>
namespace delirium::m6 {
WeldedSeamSolver::WeldedSeamSolver(float t) : default_tolerance_(t) {}
bool WeldedSeamSolver::solve(BodyState& state, const BodyTopology& topology, std::string* error) {
    for (const auto& seam : topology.welded_seams()) {
        SeamError se = sample_seam(state, topology, seam, 16);
        if (!se.within_tolerance) adjust_seam(state, topology, seam, se.max_error * 0.5f);
    }
    for (const auto& seam : topology.welded_seams()) {
        if (!sample_seam(state, topology, seam, 16).within_tolerance) {
            if (error) *error = "seam error exceeds tolerance";
            return false;
        }
    }
    return true;
}
SeamError WeldedSeamSolver::sample_seam(const BodyState& state, const BodyTopology& topology,
                                         const WeldedSeam& seam, std::uint32_t n) const {
    SeamError r; r.patch_a = seam.patch_a; r.patch_b = seam.patch_b; r.samples = n;
    const auto& pa = topology.patch(seam.patch_a);
    const auto& pb = topology.patch(seam.patch_b);
    float me = 0.0f, se2 = 0.0f;
    for (std::uint32_t i = 0; i <= n; ++i) {
        float t = static_cast<float>(i) / n;
        float ua = seam.a_u0 + (seam.a_u1-seam.a_u0)*t, va = seam.a_v0 + (seam.a_v1-seam.a_v0)*t;
        float ub = seam.b_u0 + (seam.b_u1-seam.b_u0)*t, vb = seam.b_v0 + (seam.b_v1-seam.b_v0)*t;
        Vec3 posa = evaluate_patch_position(state.anchors[pa.anchors[0]].position, state.anchors[pa.anchors[1]].position,
                                            state.anchors[pa.anchors[2]].position, state.anchors[pa.anchors[3]].position,
                                            pa.basis, pa.coefficients.scalars, pa.coefficients.residuals, ua, va);
        Vec3 posb = evaluate_patch_position(state.anchors[pb.anchors[0]].position, state.anchors[pb.anchors[1]].position,
                                            state.anchors[pb.anchors[2]].position, state.anchors[pb.anchors[3]].position,
                                            pb.basis, pb.coefficients.scalars, pb.coefficients.residuals, ub, vb);
        float e = length(posa - posb); me = std::max(me, e); se2 += e;
    }
    r.max_error = me; r.avg_error = se2 / (n+1); r.within_tolerance = me <= seam.tolerance;
    return r;
}
void WeldedSeamSolver::adjust_seam(BodyState& state, const BodyTopology& topology,
                                    const WeldedSeam& seam, float ma) const {
    const auto& pb = topology.patch(seam.patch_b);
    for (AnchorId a : pb.anchors) {
        if (a >= state.anchors.size()) continue;
        Vec3 d = state.anchors[a].position - state.anchors[a].rest_position;
        float dl = length(d);
        if (dl > 0.0f && dl < ma) state.anchors[a].position += -d * 0.5f;
    }
}
} // namespace delirium::m6