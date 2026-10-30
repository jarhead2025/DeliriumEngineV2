#include "delirium_m5/damage_accumulator.hpp"
#include <algorithm>

namespace delirium::m5 {

DamageAccumulator::DamageAccumulator() {}

void DamageAccumulator::accumulate(BodyState& state, const BodyTopology& /*topology*/,
                                    const MaterialRegistry& /*materials*/) const {
    for (PatchId pid = 0; pid < state.patches.size(); ++pid) {
        PatchState& patch = state.patches[pid];
        if (patch.cumulative_plastic_strain <= 0.0f) continue;
        float raw = patch.cumulative_plastic_strain * kDamageRate;
        float new_d = clamp_damage(raw);
        patch.damage = std::max(patch.damage, new_d);
        patch.effective_stiffness = get_effective_stiffness(patch);
    }
}

float DamageAccumulator::get_damage(const PatchState& patch) { return clamp_damage(patch.damage); }
float DamageAccumulator::get_effective_stiffness(const PatchState& patch) {
    return std::max(0.0f, 1.0f - clamp_damage(patch.damage));
}
bool DamageAccumulator::is_valid(float damage) { return damage >= 0.0f && damage <= 1.0f; }
float DamageAccumulator::clamp_damage(float damage) { return std::max(0.0f, std::min(1.0f, damage)); }

} // namespace delirium::m5