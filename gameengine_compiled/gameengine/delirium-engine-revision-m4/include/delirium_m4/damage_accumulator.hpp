#pragma once

#include "delirium_m4/types.hpp"
#include "delirium_m4/body_topology.hpp"
#include "delirium_m4/body_state.hpp"
#include "delirium_m4/material_registry.hpp"

#include <string>

namespace delirium::m4 {

// ---------------------------------------------------------------------------
// DamageAccumulator — M4: monotonic damage tracking
// ---------------------------------------------------------------------------
//
// Tracks cumulative plastic strain per patch. Damage is monotonic (never
// decreases within a single generation). Stiffness is reduced proportionally
// to damage.
//
// Invariant: damage never decreases within a generation.
// ---------------------------------------------------------------------------
class DamageAccumulator {
public:
    DamageAccumulator();

    // Accumulate damage for dirty patches based on plastic strain
    // Updates state.patches[i].damage and state.patches[i].effective_stiffness
    void accumulate(BodyState& state,
                    const BodyTopology& topology,
                    const MaterialRegistry& materials) const;

    // Get damage value for a patch (0.0 = undamaged, 1.0 = fully damaged)
    static float get_damage(const PatchState& patch);

    // Get effective stiffness multiplier (1.0 - damage)
    static float get_effective_stiffness(const PatchState& patch);

    // Check if damage is within valid range [0, 1]
    static bool is_valid(float damage);

    // Clamp damage to valid range
    static float clamp_damage(float damage);

private:
    // Maximum cumulative strain before full damage
    static constexpr float kMaxCumulativeStrain = 1.0f;

    // Damage rate: how quickly cumulative strain converts to damage
    static constexpr float kDamageRate = 0.1f;
};

} // namespace delirium::m4