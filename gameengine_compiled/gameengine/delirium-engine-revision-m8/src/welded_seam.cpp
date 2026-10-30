#include "delirium_m8/welded_seam.hpp"
#include <cmath>
#include <stdexcept>

namespace delirium::m8 {

WeldedSeamSolver::WeldedSeamSolver(float tolerance) : tolerance_(tolerance) {}

float WeldedSeamSolver::compute_error(const BodyState& state, const BodyTopology& topology,
                                       const WeldedSeam& seam) const {
    if (!seam.is_valid()) return 0.0f;
    const auto& pa = topology.patch(seam.patch_a);
    const auto& pb = topology.patch(seam.patch_b);

    Vec3 pa_pos = state.anchors[pa.anchors[0]].position * 0.25f +
                  state.anchors[pa.anchors[1]].position * 0.25f +
                  state.anchors[pa.anchors[2]].position * 0.25f +
                  state.anchors[pa.anchors[3]].position * 0.25f;
    Vec3 pb_pos = state.anchors[pb.anchors[0]].position * 0.25f +
                  state.anchors[pb.anchors[1]].position * 0.25f +
                  state.anchors[pb.anchors[2]].position * 0.25f +
                  state.anchors[pb.anchors[3]].position * 0.25f;

    return length(pa_pos - pb_pos);
}

bool WeldedSeamSolver::solve(BodyState& state, const BodyTopology& topology,
                              std::string* error) {
    for (const auto& seam : topology.welded_seams()) {
        if (!seam.is_valid()) {
            if (error) *error = "invalid seam";
            return false;
        }

        float err = compute_error(state, topology, seam);
        if (err < seam.tolerance) continue;

        const auto& pa = topology.patch(seam.patch_a);
        const auto& pb = topology.patch(seam.patch_b);

        Vec3 correction = Vec3{};
        for (AnchorId a : pa.anchors) {
            correction += state.anchors[a].velocity;
        }
        for (AnchorId b : pb.anchors) {
            state.anchors[b].velocity += correction * 0.1f;
        }
    }
    return true;
}

} // namespace delirium::m8