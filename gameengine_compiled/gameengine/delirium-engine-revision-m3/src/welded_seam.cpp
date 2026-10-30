#include "delirium_m3/welded_seam.hpp"

#include "delirium_m3/basis_functions.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace delirium::m3 {

WeldedSeamSolver::WeldedSeamSolver(float default_tolerance)
    : default_tolerance_(default_tolerance) {}

SeamError WeldedSeamSolver::sample_seam(const BodyState& state,
                                         const BodyTopology& topology,
                                         const WeldedSeam& seam,
                                         std::uint32_t num_samples) const {
    SeamError error;
    error.patch_a = seam.patch_a;
    error.patch_b = seam.patch_b;
    error.samples = num_samples;

    if (!seam.is_valid()) {
        error.within_tolerance = false;
        error.max_error = 1e6f;
        return error;
    }

    const PatchTopology& pa = topology.patch(seam.patch_a);
    const PatchTopology& pb = topology.patch(seam.patch_b);

    float max_err = 0.0f;
    float sum_err = 0.0f;

    for (std::uint32_t i = 0; i <= num_samples; ++i) {
        float t = static_cast<float>(i) / num_samples;

        // Sample patch A
        float ua = seam.a_u0 + (seam.a_u1 - seam.a_u0) * t;
        float va = seam.a_v0 + (seam.a_v1 - seam.a_v0) * t;

        // Sample patch B
        float ub = seam.b_u0 + (seam.b_u1 - seam.b_u0) * t;
        float vb = seam.b_v0 + (seam.b_v1 - seam.b_v0) * t;

        // Get anchor positions for each patch
        const Vec3& a_p00 = state.anchors[pa.anchors[0]].position;
        const Vec3& a_p10 = state.anchors[pa.anchors[1]].position;
        const Vec3& a_p01 = state.anchors[pa.anchors[2]].position;
        const Vec3& a_p11 = state.anchors[pa.anchors[3]].position;

        const Vec3& b_p00 = state.anchors[pb.anchors[0]].position;
        const Vec3& b_p10 = state.anchors[pb.anchors[1]].position;
        const Vec3& b_p01 = state.anchors[pb.anchors[2]].position;
        const Vec3& b_p11 = state.anchors[pb.anchors[3]].position;

        Vec3 pos_a = evaluate_patch_position(a_p00, a_p10, a_p01, a_p11,
                                              pa.basis, pa.coefficients.scalars,
                                              pa.coefficients.residuals, ua, va);
        Vec3 pos_b = evaluate_patch_position(b_p00, b_p10, b_p01, b_p11,
                                              pb.basis, pb.coefficients.scalars,
                                              pb.coefficients.residuals, ub, vb);

        float d = length(pos_a - pos_b);
        if (d > max_err) max_err = d;
        sum_err += d;
    }

    error.max_error = max_err;
    error.avg_error = sum_err / (num_samples + 1);
    error.within_tolerance = max_err <= seam.tolerance;

    return error;
}

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

void WeldedSeamSolver::adjust_seam(BodyState& state,
                                    const BodyTopology& topology,
                                    const WeldedSeam& seam,
                                    float max_adjustment) const {
    // Simple approach: compute average gap at seam endpoints and
    // adjust the non-shared anchors toward each other
    const PatchTopology& pa = topology.patch(seam.patch_a);
    const PatchTopology& pb = topology.patch(seam.patch_b);

    // Sample endpoints
    Vec3 a_start = state.anchors[pa.anchors[0]].position;
    Vec3 a_end = state.anchors[pa.anchors[1]].position;
    Vec3 b_start = state.anchors[pb.anchors[0]].position;
    Vec3 b_end = state.anchors[pb.anchors[1]].position;

    Vec3 gap_start = b_start - a_start;
    Vec3 gap_end = b_end - a_end;
    Vec3 avg_gap = (gap_start + gap_end) * 0.5f;

    float gap_len = length(avg_gap);
    if (gap_len < seam.tolerance) return;  // already within tolerance

    // Adjust anchors by half the gap each (distributed)
    float adjustment = std::min(gap_len * 0.5f, max_adjustment);
    Vec3 direction = avg_gap * (1.0f / gap_len);
    Vec3 delta = direction * adjustment;

    // Move patch A anchors toward patch B
    for (AnchorId aid : pa.anchors) {
        state.anchors[aid].position = state.anchors[aid].position + delta;
    }
    // Move patch B anchors toward patch A
    for (AnchorId bid : pb.anchors) {
        state.anchors[bid].position = state.anchors[bid].position - delta;
    }
}

bool WeldedSeamSolver::solve(BodyState& state,
                              const BodyTopology& topology,
                              std::string* error) {
    if (topology.welded_seams().empty()) return true;

    // Iterative solve: up to 8 passes
    for (int pass = 0; pass < 8; ++pass) {
        bool all_ok = true;
        for (const auto& seam : topology.welded_seams()) {
            SeamError se = compute_error(state, topology, seam);
            if (!se.within_tolerance) {
                all_ok = false;
                adjust_seam(state, topology, seam, seam.tolerance * 2.0f);
            }
        }
        if (all_ok) break;
    }

    // Final check
    for (const auto& seam : topology.welded_seams()) {
        SeamError se = compute_error(state, topology, seam);
        if (!se.within_tolerance) {
            if (error) {
                *error = "welded seam between patch " +
                         std::to_string(seam.patch_a) + " and " +
                         std::to_string(seam.patch_b) +
                         " exceeds tolerance (error=" +
                         std::to_string(se.max_error) + ")";
            }
            return false;
        }
    }

    return true;
}

} // namespace delirium::m3