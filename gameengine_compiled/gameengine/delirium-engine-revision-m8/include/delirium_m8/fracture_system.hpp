#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_state.hpp"
#include "delirium_m8/body_topology.hpp"
#include <vector>
namespace delirium::m8 {
class FractureSystem {
public:
    explicit FractureSystem(FractureParams params = {});
    void set_params(FractureParams p);
    FractureParams params() const { return params_; }
    FractureEvent evaluate(const BodyState& state, const BodyTopology& topology,
                           BodyId body_id) const;
    bool should_fracture(const BodyState& state, const BodyTopology& topology) const;
    std::vector<PatchId> find_fractured_patches(const BodyState& state,
                                                const BodyTopology& topology) const;
private:
    FractureParams params_;
};
} // namespace delirium::m8