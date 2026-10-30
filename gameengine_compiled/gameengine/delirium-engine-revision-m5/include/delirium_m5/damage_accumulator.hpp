#pragma once

#include "delirium_m5/types.hpp"
#include "delirium_m5/body_topology.hpp"
#include "delirium_m5/body_state.hpp"
#include "delirium_m5/material_registry.hpp"
#include <string>

namespace delirium::m5 {

class DamageAccumulator {
public:
    DamageAccumulator();
    void accumulate(BodyState& state, const BodyTopology& topology,
                    const MaterialRegistry& materials) const;
    static float get_damage(const PatchState& patch);
    static float get_effective_stiffness(const PatchState& patch);
    static bool is_valid(float damage);
    static float clamp_damage(float damage);
private:
    static constexpr float kMaxCumulativeStrain = 1.0f;
    static constexpr float kDamageRate = 0.1f;
};

} // namespace delirium::m5