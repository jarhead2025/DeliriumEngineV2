#include "delirium_m5/elastic_solver.hpp"
#include <cmath>
#include <algorithm>

namespace delirium::m5 {

ElasticSolver::ElasticSolver(float default_stiffness) : default_stiffness_(default_stiffness) {}

bool ElasticSolver::solve(BodyState& state, const BodyTopology& topology,
                           const MaterialRegistry& materials, float fixed_dt, std::string* error) {
    if (fixed_dt <= 0.0f) { if (error) *error = "fixed_dt must be positive"; return false; }
    auto dirty_indices = state.dirty_patches.indices();
    for (PatchId pid : dirty_indices) {
        const PatchTopology& patch = topology.patch(pid);
        const MaterialProperties& mat = materials.get(patch.material);
        float strain = compute_strain(state, patch);
        float energy = compute_strain_energy(state, patch, mat);
        state.patches[pid].strain_energy = std::max(0.0f, energy);
        if (is_plastic(state, patch, mat))
            apply_plastic_deformation(state, pid, strain, mat);
        else
            apply_elastic_force(state, topology, pid, mat, fixed_dt);
    }
    propagate_strain(state, topology);
    return true;
}

float ElasticSolver::compute_strain(const BodyState& state, const PatchTopology& patch) const {
    float total = 0.0f;
    for (AnchorId a : patch.anchors)
        if (a < state.anchors.size())
            total += length(state.anchors[a].position - state.anchors[a].rest_position);
    return total / 4.0f;
}

float ElasticSolver::compute_strain_energy(const BodyState& state, const PatchTopology& patch,
                                            const MaterialProperties& material) const {
    float strain = compute_strain(state, patch);
    return std::max(0.0f, 0.5f * material.youngs_modulus * strain * strain);
}

bool ElasticSolver::is_plastic(const BodyState& state, const PatchTopology& patch,
                                const MaterialProperties& material) const {
    return material.youngs_modulus * compute_strain(state, patch) > material.yield_strength;
}

void ElasticSolver::apply_elastic_force(BodyState& state, const BodyTopology& topology,
                                         PatchId patch_id, const MaterialProperties& material,
                                         float fixed_dt) const {
    const PatchTopology& patch = topology.patch(patch_id);
    float eff = std::max(0.0f, state.patches[patch_id].effective_stiffness);
    float k = material.youngs_modulus * eff * default_stiffness_;
    for (AnchorId a : patch.anchors) {
        if (a >= state.anchors.size()) continue;
        Vec3 disp = state.anchors[a].position - state.anchors[a].rest_position;
        Vec3 force = -disp * (k * fixed_dt) - state.anchors[a].velocity * (0.1f * material.density * fixed_dt);
        state.anchors[a].velocity += force / std::max(1.0f, material.density);
        state.anchors[a].position += state.anchors[a].velocity * fixed_dt;
    }
}

void ElasticSolver::apply_plastic_deformation(BodyState& state, PatchId patch_id,
                                                float strain, const MaterialProperties&) const {
    state.patches[patch_id].cumulative_plastic_strain += strain;
    float d = std::min(1.0f, state.patches[patch_id].cumulative_plastic_strain / 1.0f);
    state.patches[patch_id].effective_stiffness = std::max(0.0f, 1.0f - d);
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

} // namespace delirium::m5