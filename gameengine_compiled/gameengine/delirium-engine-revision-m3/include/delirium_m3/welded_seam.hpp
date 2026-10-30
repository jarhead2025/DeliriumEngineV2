#pragma once

#include "delirium_m3/types.hpp"
#include "delirium_m3/body_topology.hpp"
#include "delirium_m3/body_state.hpp"

#include <string>
#include <vector>

namespace delirium::m3 {

// ---------------------------------------------------------------------------
// WeldedSeamSolver — stitches non-shared patch boundaries
// ---------------------------------------------------------------------------
//
// After articulation moves anchors, welded seams can develop gaps. The solver
// adjusts anchor positions minimally to keep the seam error within tolerance.
// ---------------------------------------------------------------------------

struct SeamError {
    PatchId patch_a = kInvalidPatch;
    PatchId patch_b = kInvalidPatch;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    std::uint32_t samples = 0;
    bool within_tolerance = true;
};

class WeldedSeamSolver {
public:
    explicit WeldedSeamSolver(float default_tolerance = 1e-4f);

    // Compute error for all welded seams in the current state
    std::vector<SeamError> compute_errors(const BodyState& state,
                                          const BodyTopology& topology) const;

    // Compute error for a single welded seam
    SeamError compute_error(const BodyState& state,
                            const BodyTopology& topology,
                            const WeldedSeam& seam) const;

    // Adjust anchor positions to reduce seam error within tolerance
    // Returns true if all seams are within tolerance after adjustment
    bool solve(BodyState& state,
               const BodyTopology& topology,
               std::string* error = nullptr);

private:
    float default_tolerance_;

    // Sample a welded seam at N points and compute error
    SeamError sample_seam(const BodyState& state,
                          const BodyTopology& topology,
                          const WeldedSeam& seam,
                          std::uint32_t num_samples) const;

    // Adjust anchor positions to close a seam gap
    void adjust_seam(BodyState& state,
                     const BodyTopology& topology,
                     const WeldedSeam& seam,
                     float max_adjustment) const;
};

} // namespace delirium::m3