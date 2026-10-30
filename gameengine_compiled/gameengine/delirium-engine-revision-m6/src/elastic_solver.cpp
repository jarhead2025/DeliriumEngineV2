#include "delirium_m6/elastic_solver.hpp"
#include <cmath>
#include <algorithm>
namespace delirium::m6 {
ElasticSolver::ElasticSolver(float s) : default_stiffness_(s) {}
bool ElasticSolver::solve(BodyState& state, const BodyTopology& topology,
                           const MaterialRegistry& materials, float fixed_dt, std::string* error) {
    if (fixed_dt <= 0.0f) { if (error) *error = "fixed_dt must be positive"; return false; }
    auto dirty = state.dirty_patches.indices();
    for (PatchId pid : dirty) {
        const auto& patch = topology.patch(pid);
        const auto& mat = materials.get(patch.material);
        float strain = compute_strain(state, patch);
        float energy = compute_strain_energy(state, patch, mat);
        state.patches[pid].strain_energy = std::max(0.0f, energy);
        if (mat.youngs_modulus * strain > mat.yield_strength)
            apply_plastic_deformation(state, pid, strain);
        else
            apply_elastic_force(state, topology, pid, mat, fixed_dt);
    }
    propagate_strain(state, topology);
    return true;
}
float ElasticSolver::compute_strain(const BodyState& state, const PatchTopology& patch) const {
    float t = 0.0f;
    for (AnchorId a : patch.anchors)
        if (a < state.anchors.size()) t += length(state.anchors[a].position - state.anchors[a].rest_position);
    return t / 4.0f;
}
float ElasticSolver::compute_strain_energy(const BodyState& state, const PatchTopology& patch,
                                            const MaterialProperties& mat) const {
    float s = compute_strain(state, patch);
    return std::max(0.0f, 0.5f * mat.youngs_modulus * s * s);
}
void ElasticSolver::apply_elastic_force(BodyState& state, const BodyTopology& topology,
                                         PatchId pid, const MaterialProperties& mat, float dt) const {
    const auto& patch = topology.patch(pid);
    float eff = std::max(0.0f, state.patches[pid].effective_stiffness);
    float k = mat.youngs_modulus * eff * default_stiffness_;
    for (AnchorId a : patch.anchors) {
        if (a >= state.anchors.size()) continue;
        Vec3 disp = state.anchors[a].position - state.anchors[a].rest_position;
        Vec3 force = -disp * (k * dt) - state.anchors[a].velocity * (0.1f * mat.density * dt);
        state.anchors[a].velocity += force / std::max(1.0f, mat.density);
        state.anchors[a].position += state.anchors[a].velocity * dt;
    }
}
void ElasticSolver::apply_plastic_deformation(BodyState& state, PatchId pid, float strain) const {
    state.patches[pid].cumulative_plastic_strain += strain;
    float d = std::min(1.0f, state.patches[pid].cumulative_plastic_strain / 1.0f);
    state.patches[pid].effective_stiffness = std::max(0.0f, 1.0f - d);
}
void ElasticSolver::propagate_strain(BodyState& state, const BodyTopology& topology) const {
    for (AnchorId anchor = 0; anchor < state.anchors.size(); ++anchor) {
        const auto& incident = topology.patches_for_anchor(anchor);
        if (incident.empty()) continue;
        float total = 0.0f;
        for (PatchId pid : incident)
            if (pid < state.patches.size()) total += state.patches[pid].strain_energy;
        state.anchors[anchor].temperature = total / incident.size() * 1e-6f;
    }
}
} // namespace delirium::m6