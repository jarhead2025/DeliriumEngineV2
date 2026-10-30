#include "delirium_m7/damage_accumulator.hpp"
#include <algorithm>

namespace delirium::m7 {

DamageAccumulator::DamageAccumulator(float damage_rate) : damage_rate_(damage_rate) {}

bool DamageAccumulator::accumulate(BodyState& state, const BodyTopology& topology,
                                    const MaterialRegistry& materials, float fixed_dt,
                                    std::string* error) {
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        if (!materials.has(patch.material)) {
            if (error) *error = "missing material";
            return false;
        }
        const auto& mat = materials.get(patch.material);
        float strain = state.patches[pid].cumulative_plastic_strain;
        float new_damage = std::min(1.0f, state.patches[pid].damage +
                          strain * damage_rate_ * mat.youngs_modulus * fixed_dt);
        state.patches[pid].damage = new_damage;

        for (AnchorId a : patch.anchors) {
            state.anchors[a].damage = std::min(1.0f, state.anchors[a].damage +
                                  strain * damage_rate_ * fixed_dt);
        }
    }
    return true;
}

} // namespace delirium::m7