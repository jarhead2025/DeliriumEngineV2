#include "delirium_m7/thermal_simulator.hpp"

namespace delirium::m7 {

ThermalSimulator::ThermalSimulator(float ambient_temp) : ambient_temp_(ambient_temp) {}

void ThermalSimulator::diffuse(BodyState& state, const BodyTopology& topology,
                                [[maybe_unused]] const MaterialRegistry& materials,
                                float fixed_dt) const {
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        float avg_temp = 0.0f;
        for (AnchorId a : patch.anchors) avg_temp += state.anchors[a].temperature;
        avg_temp /= static_cast<float>(patch.anchors.size());

        for (AnchorId a : patch.anchors) {
            float diff = avg_temp - state.anchors[a].temperature;
            state.anchors[a].temperature += diff * 0.1f * fixed_dt;
        }
        state.patches[pid].temperature = avg_temp;
    }
}

void ThermalSimulator::cool(BodyState& state, float fixed_dt) const {
    for (auto& anchor : state.anchors) {
        float diff = ambient_temp_ - anchor.temperature;
        anchor.temperature += diff * 0.01f * fixed_dt;
    }
}

} // namespace delirium::m7