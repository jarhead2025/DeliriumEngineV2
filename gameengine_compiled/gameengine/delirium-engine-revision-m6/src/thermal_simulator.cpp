#include "delirium_m6/thermal_simulator.hpp"
namespace delirium::m6 {
ThermalSimulator::ThermalSimulator(float r) : diffusion_rate_(r) {}
void ThermalSimulator::diffuse(BodyState& state, const BodyTopology& topology,
                                const MaterialRegistry& /*materials*/, float dt) const {
    for (AnchorId anchor = 0; anchor < state.anchors.size(); ++anchor) {
        const auto& incident = topology.patches_for_anchor(anchor);
        if (incident.empty()) continue;
        float avg = 0.0f;
        for (PatchId pid : incident)
            if (pid < state.patches.size()) avg += state.patches[pid].temperature;
        avg /= incident.size();
        float diff = avg - state.anchors[anchor].temperature;
        state.anchors[anchor].temperature += diff * diffusion_rate_ * dt;
    }
}
void ThermalSimulator::apply_expansion(BodyState& state, const BodyTopology& topology,
                                        const MaterialRegistry& materials, float dt) const {
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        const auto& mat = materials.get(patch.material);
        if (mat.thermal_expansion <= 0.0f) continue;
        float temp = state.patches[pid].temperature;
        if (temp <= 0.0f) continue;
        float expansion = mat.thermal_expansion * temp * dt;
        for (AnchorId a : patch.anchors) {
            if (a >= state.anchors.size()) continue;
            Vec3 dir = state.anchors[a].position - state.anchors[a].rest_position;
            float d = length(dir);
            if (d > 1e-8f) state.anchors[a].position += dir * (expansion / d);
        }
    }
}
void ThermalSimulator::cool(BodyState& state, const BodyTopology& /*topology*/,
                             float ambient, float rate, float dt) const {
    for (auto& anchor : state.anchors) {
        float diff = ambient - anchor.temperature;
        anchor.temperature += diff * rate * dt;
    }
}
} // namespace delirium::m6