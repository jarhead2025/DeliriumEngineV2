#include "delirium_m5/thermal_simulator.hpp"
#include <algorithm>

namespace delirium::m5 {

ThermalSimulator::ThermalSimulator(float diffusion_rate) : diffusion_rate_(diffusion_rate) {}

void ThermalSimulator::diffuse(BodyState& state, const BodyTopology& topology,
                                const MaterialRegistry& materials, float fixed_dt) const {
    for (AnchorId anchor = 0; anchor < state.anchors.size(); ++anchor) {
        const auto& incident = topology.patches_for_anchor(anchor);
        if (incident.empty()) continue;

        float avg_temp = 0.0f;
        for (PatchId pid : incident) {
            if (pid < state.patches.size())
                avg_temp += state.patches[pid].temperature;
        }
        avg_temp /= incident.size();

        // Diffuse: anchor temperature moves toward average patch temperature
        float diff = avg_temp - state.anchors[anchor].temperature;
        state.anchors[anchor].temperature += diff * diffusion_rate_ * fixed_dt;
    }
}

void ThermalSimulator::apply_expansion(BodyState& state, const BodyTopology& topology,
                                        const MaterialRegistry& materials, float fixed_dt) const {
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const PatchTopology& patch = topology.patch(pid);
        const MaterialProperties& mat = materials.get(patch.material);
        if (mat.thermal_expansion <= 0.0f) continue;

        float temp = state.patches[pid].temperature;
        if (temp <= 0.0f) continue;

        float expansion = mat.thermal_expansion * temp * fixed_dt;
        for (AnchorId a : patch.anchors) {
            if (a >= state.anchors.size()) continue;
            Vec3 dir = state.anchors[a].position - state.anchors[a].rest_position;
            float d = length(dir);
            if (d > 1e-8f) {
                state.anchors[a].position += dir * (expansion / d);
            }
        }
    }
}

void ThermalSimulator::cool(BodyState& state, const BodyTopology& /*topology*/,
                             float ambient_temperature, float cooling_rate, float fixed_dt) const {
    for (auto& anchor : state.anchors) {
        float diff = ambient_temperature - anchor.temperature;
        anchor.temperature += diff * cooling_rate * fixed_dt;
    }
}

} // namespace delirium::m5