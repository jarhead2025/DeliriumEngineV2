#pragma once
#include "delirium_m7/types.hpp"
#include "delirium_m7/body_state.hpp"
#include "delirium_m7/body_topology.hpp"
#include "delirium_m7/material_registry.hpp"
#include <string>
namespace delirium::m7 {
class DamageAccumulator {
public:
    explicit DamageAccumulator(float damage_rate = 1e-6f);
    bool accumulate(BodyState& state, const BodyTopology& topology,
                    const MaterialRegistry& materials, float fixed_dt, std::string* error = nullptr);
private:
    float damage_rate_;
};
} // namespace delirium::m7