#include "delirium_m4/welded_seam.hpp"
#include "delirium_m4/basis_functions.hpp"

#include <algorithm>
#include <cmath>

namespace delirium::m4 {

WeldedSeamSolver::WeldedSeamSolver(float default_tolerance)
    : default_tolerance_(default_tolerance) {}

std::vector<SeamError> WeldedSeamSolver::compute_errors(const BodyState& state,
                                                         const BodyTopology& topology) const {
    std::vector<SeamError> errors;
    for (const auto& seam : topology.welded_seams()) {
        errors.push_back(compute_error(state, topology, seam));
    }
    return errors;
}

SeamError WeldedSeamSolver::compute_error(const BodyState& state,
                                           const BodyTopology& topology,
                                           const WeldedSeam& seam) const {
    return sample_seam(state, topology, seam, 16);
}

bool WeldedSeamSolver::solve(BodyState& state,
                              const BodyTopology& topology,
                              std::string* error) {
    for (const auto& seam : topology.welded_seams()) {
        SeamError se = sample_seam(state, topology, seam, 16);
        if (!se.within_tolerance) {
            adjust_seam(state, topology, seam, se.max_error * 0.5f);
        }
    }

    // Verify all seams after adjustment
    for (const auto& seam : topology.welded_seams()) {
        SeamError se = sample_seam(state, topology, seam, 16);
        if (!se.within_tolerance) {
            if (error) *error = "seam error exceeds tolerance after adjustment";
            return false;
        }
    }

    return true;
}

SeamError WeldedSeamSolver::sample_seam(const BodyState& state,
                                         const BodyTopology& topology,
                                         const WeldedSeam& seam,
                                         std::uint32_t num_samples) const {
    SeamError result;
    result.patch_a = seam.patch_a;
    result.patch_b = seam.patch_b;
    result.samples = num_samples;

    const PatchTopology& pa = topology.patch(seam.patch_a);
    const PatchTopology& pb = topology.patch(seam.patch_b);

    float max_err = 0.0f;
    float sum_err = 0.0f;

    for (std::uint32_t i = 0; i <= num_samples; ++i) {
        float t = static_cast<float>(i) / num_samples;

        // Parametric position on patch A edge
        float ua = seam.a_u0 + (seam.a_u1 - seam.a_u0) * t;
        float va = seam.a_v0 + (seam.a_v1 - seam.a_v0) * t;

        // Parametric position on patch B edge
        float ub = seam.b_u0 + (seam.b_u1 - seam.b_u0) * t;
        float vb = seam.b_v0 + (seam.b_v1 - seam.b_v0) * t;

        // Evaluate positions
        Vec3 pos_a = evaluate_patch_position(
            state.anchors[pa.anchors[0]].position,
            state.anchors[pa.anchors[1]].position,
            state.anchors[pa.anchors[2]].position,
            state.anchors[pa.anchors[3]].position,
            pa.basis, pa.coefficients.scalars, pa.coefficients.residuals,
            ua, va);

        Vec3 pos_b = evaluate_patch_position(
            state.anchors[pb.anchors[0]].position,
            state.anchors[pb.anchors[1]].position,
            state.anchors[pb.anchors[2]].position,
            state.anchors[pb.anchors[3]].position,
            pb.basis, pb.coefficients.scalars, pb.coefficients.residuals,
            ub, vb);

        float err = length(pos_a - pos_b);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    result.max_error = max_err;
    result.avg_error = sum_err / (num_samples + 1);
    result.within_tolerance = max_err <= seam.tolerance;

    return result;
}

void WeldedSeamSolver::adjust_seam(BodyState& state,
                                    const BodyTopology& topology,
                                    const WeldedSeam& seam,
                                    float max_adjustment) const {
    const PatchTopology& pb = topology.patch(seam.patch_b);

    // Adjust patch B anchors toward patch A to close the gap
    for (AnchorId a : pb.anchors) {
        if (a >= state.anchors.size()) continue;

        Vec3 displacement = state.anchors[a].position - state.anchors[a].rest_position;
        float disp_len = length(displacement);

        if (disp_len > 0.0f && disp_len < max_adjustment) {
            Vec3 adjustment = -displacement * 0.5f;
            state.anchors[a].position += adjustment;
        }
    }
}

} // namespace delirium::m4