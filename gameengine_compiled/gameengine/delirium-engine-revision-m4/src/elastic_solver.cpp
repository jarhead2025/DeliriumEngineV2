#include "delirium_m4/elastic_solver.hpp"

#include <cmath>
#include <algorithm>

namespace delirium::m4 {

ElasticSolver::ElasticSolver(float default_stiffness)
    : default_stiffness_(default_stiffness) {}

bool ElasticSolver::solve(BodyState& state,
                           const BodyTopology& topology,
                           const MaterialRegistry& materials,
                           float fixed_dt,
                           std::string* error) {
    if (fixed_dt <= 0.0f) {
        if (error) *error = "fixed_dt must be positive";
        return false;
    }

    // Solve only dirty patches
    auto dirty_indices = state.dirty_patches.indices();
    if (dirty_indices.empty()) {
        // No dirty patches — check all anchors for movement
        for (AnchorId anchor = 0; anchor < state.anchors.size(); ++anchor) {
            if (!nearly_equal(state.anchors[anchor].position,
                              state.anchors[anchor].rest_position)) {
                return false; // anchors moved, need solve
            }
        }
        return true;
    }

    for (PatchId pid : dirty_indices) {
        const PatchTopology& patch = topology.patch(pid);
        const MaterialProperties& mat = materials.get(patch.material);

        float strain = compute_strain(state, patch);
        float energy = compute_strain_energy(state, patch, mat);

        // Update patch strain energy (must be non-negative)
        state.patches[pid].strain_energy = std::max(0.0f, energy);

        if (is_plastic(state, patch, mat)) {
            // Plastic deformation: permanent displacement
            apply_plastic_deformation(state, pid, strain, mat);
        } else {
            // Elastic restoration
            apply_elastic_force(state, topology, pid, mat, fixed_dt);
        }
    }

    // Propagate strain between adjacent patches
    propagate_strain(state, topology);

    return true;
}

float ElasticSolver::compute_strain(const BodyState& state,
                                     const PatchTopology& patch) const {
    // Compute displacement-based strain as average anchor displacement from rest
    float total_displacement = 0.0f;
    for (AnchorId a : patch.anchors) {
        if (a < state.anchors.size()) {
            Vec3 disp = state.anchors[a].position - state.anchors[a].rest_position;
            total_displacement += length(disp);
        }
    }
    return total_displacement / 4.0f;
}

float ElasticSolver::compute_strain_energy(const BodyState& state,
                                            const PatchTopology& patch,
                                            const MaterialProperties& material) const {
    float strain = compute_strain(state, patch);
    // Strain energy = 0.5 * E * strain^2 (per-unit-volume approximation)
    float energy = 0.5f * material.youngs_modulus * strain * strain;
    return std::max(0.0f, energy); // invariant: non-negative
}

bool ElasticSolver::is_plastic(const BodyState& state,
                                const PatchTopology& patch,
                                const MaterialProperties& material) const {
    float strain = compute_strain(state, patch);
    // Stress = E * strain; plastic if stress exceeds yield strength
    float stress = material.youngs_modulus * strain;
    return stress > material.yield_strength;
}

void ElasticSolver::apply_elastic_force(BodyState& state,
                                         const BodyTopology& topology,
                                         PatchId patch_id,
                                         const MaterialProperties& material,
                                         float fixed_dt) const {
    const PatchTopology& patch = topology.patch(patch_id);

    // Effective stiffness reduced by patch damage
    float eff_stiffness = std::max(0.0f, state.patches[patch_id].effective_stiffness);
    float spring_k = material.youngs_modulus * eff_stiffness * default_stiffness_;

    for (AnchorId a : patch.anchors) {
        if (a >= state.anchors.size()) continue;

        Vec3 displacement = state.anchors[a].position - state.anchors[a].rest_position;
        Vec3 force = -displacement * (spring_k * fixed_dt);

        // Damping
        force -= state.anchors[a].velocity * (0.1f * material.density * fixed_dt);

        state.anchors[a].velocity += force / std::max(1.0f, material.density);
        state.anchors[a].position += state.anchors[a].velocity * fixed_dt;
    }
}

void ElasticSolver::apply_plastic_deformation(BodyState& state,
                                                PatchId patch_id,
                                                float strain,
                                                const MaterialProperties& material) const {
    // Update cumulative plastic strain (monotonic increase)
    state.patches[patch_id].cumulative_plastic_strain += strain;

    // Reduce effective stiffness based on cumulative strain
    float damage_factor = std::min(1.0f,
        state.patches[patch_id].cumulative_plastic_strain / 1.0f);
    state.patches[patch_id].effective_stiffness = std::max(0.0f, 1.0f - damage_factor);
}

void ElasticSolver::propagate_strain(BodyState& state,
                                      const BodyTopology& topology) const {
    // For each anchor, average strain energy from all incident patches
    for (AnchorId anchor = 0; anchor < state.anchors.size(); ++anchor) {
        const auto& incident = topology.patches_for_anchor(anchor);
        if (incident.empty()) continue;

        float total_energy = 0.0f;
        for (PatchId pid : incident) {
            if (pid < state.patches.size()) {
                total_energy += state.patches[pid].strain_energy;
            }
        }
        float avg_energy = total_energy / incident.size();

        // Propagate averaged strain to anchor temperature (heat from deformation)
        state.anchors[anchor].temperature = avg_energy * 1e-6f;
    }
}

} // namespace delirium::m4