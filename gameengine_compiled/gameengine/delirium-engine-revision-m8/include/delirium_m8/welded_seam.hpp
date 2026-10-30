#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_topology.hpp"
#include "delirium_m8/body_state.hpp"
#include <string>
namespace delirium::m8 {
class WeldedSeamSolver {
public:
    explicit WeldedSeamSolver(float tolerance = 1e-4f);
    bool solve(BodyState& state, const BodyTopology& topology, std::string* error = nullptr);
    float compute_error(const BodyState& state, const BodyTopology& topology,
                        const WeldedSeam& seam) const;
private:
    float tolerance_;
};
} // namespace delirium::m8