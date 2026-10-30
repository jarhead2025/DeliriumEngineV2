#pragma once

#include "delirium_m5/types.hpp"
#include "delirium_m5/body_topology.hpp"
#include "delirium_m5/body_state.hpp"
#include <string>
#include <vector>

namespace delirium::m5 {

struct SeamError {
    PatchId patch_a = kInvalidPatch, patch_b = kInvalidPatch;
    float max_error = 0.0f, avg_error = 0.0f;
    std::uint32_t samples = 0;
    bool within_tolerance = true;
};

class WeldedSeamSolver {
public:
    explicit WeldedSeamSolver(float default_tolerance = 1e-4f);
    std::vector<SeamError> compute_errors(const BodyState& state, const BodyTopology& topology) const;
    SeamError compute_error(const BodyState& state, const BodyTopology& topology, const WeldedSeam& seam) const;
    bool solve(BodyState& state, const BodyTopology& topology, std::string* error = nullptr);
private:
    float default_tolerance_;
    SeamError sample_seam(const BodyState& state, const BodyTopology& topology, const WeldedSeam& seam, std::uint32_t num_samples) const;
    void adjust_seam(BodyState& state, const BodyTopology& topology, const WeldedSeam& seam, float max_adjustment) const;
};

} // namespace delirium::m5