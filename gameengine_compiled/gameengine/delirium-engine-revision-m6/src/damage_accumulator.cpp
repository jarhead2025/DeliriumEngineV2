#include "delirium_m6/damage_accumulator.hpp"
#include <algorithm>
namespace delirium::m6 {
DamageAccumulator::DamageAccumulator() {}
void DamageAccumulator::accumulate(BodyState& state, const BodyTopology& /*topology*/,
                                    const MaterialRegistry& /*materials*/) const {
    for (PatchId pid = 0; pid < state.patches.size(); ++pid) {
        PatchState& p = state.patches[pid];
        if (p.cumulative_plastic_strain <= 0.0f) continue;
        float raw = p.cumulative_plastic_strain * kDamageRate;
        float nd = clamp_damage(raw);
        p.damage = std::max(p.damage, nd);
        p.effective_stiffness = get_effective_stiffness(p);
    }
}
float DamageAccumulator::get_damage(const PatchState& p) { return clamp_damage(p.damage); }
float DamageAccumulator::get_effective_stiffness(const PatchState& p) {
    return std::max(0.0f, 1.0f - clamp_damage(p.damage));
}
bool DamageAccumulator::is_valid(float d) { return d >= 0.0f && d <= 1.0f; }
float DamageAccumulator::clamp_damage(float d) { return std::max(0.0f, std::min(1.0f, d)); }
} // namespace delirium::m6