#include "delirium_m7/elastic_solver.hpp"
#include <cmath>

namespace delirium::m7 {

ElasticSolver::ElasticSolver(float default_stiffness) : default_stiffness_(default_stiffness) {}

float ElasticSolver::compute_strain(const BodyState& state, const PatchTopology& patch) const {
    const auto& a00 = state.anchors[patch.anchors[0]];
    const auto& a10 = state.anchors[patch.anchors[1]];
    const auto& a01 = state.anchors[patch.anchors[2]];
    const auto& a11 = state.anchors[patch.anchors[3]];

    Vec3 rest0 = a10.rest_position - a00.rest_position;
    Vec3 rest1 = a01.rest_position - a00.rest_position;
    Vec3 cur0  = a10.position - a00.position;
    Vec3 cur1  = a01.position - a00.position;

    float rest_len0 = length(rest0);
    float rest_len1 = length(rest1);
    if (rest_len0 < 1e-6f || rest_len1 < 1e-6f) return 0.0f;

    float strain0 = std::abs(length(cur0) - rest_len0) / rest_len0;
    float strain1 = std::abs(length(cur1) - rest_len1) / rest_len1;

    return (strain0 + strain1) * 0.5f;
}

void ElasticSolver::apply_elastic_force(BodyState& state, const BodyTopology& topology,
                                         PatchId patch_id, const MaterialProperties& material,
                                         float fixed_dt) const {
    const auto& patch = topology.patch(patch_id);
    float strain = compute_strain(state, patch);

    for (AnchorId a : patch.anchors) {
        auto& anchor = state.anchors[a];
        Vec3 displacement = anchor.position - anchor.rest_position;
        float stiffness = default_stiffness_ * material.youngs_modulus * 1e-9f;
        Vec3 force = -displacement * stiffness * strain;
        anchor.velocity += force * fixed_dt;
    }
}

void ElasticSolver::apply_plastic_deformation(BodyState& state, PatchId patch_id,
                                               float strain) const {
    if (strain > 0.01f) {
        state.patches[patch_id].cumulative_plastic_strain += strain * 0.1f;
        state.patches[patch_id].effective_stiffness *= 0.99f;
    }
}

bool ElasticSolver::solve(BodyState& state, const BodyTopology& topology,
                           const MaterialRegistry& materials, float fixed_dt,
                           std::string* error) {
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        if (!materials.has(patch.material)) {
            if (error) *error = "missing material";
            return false;
        }
        const auto& mat = materials.get(patch.material);
        apply_elastic_force(state, topology, pid, mat, fixed_dt);

        float strain = compute_strain(state, patch);
        state.patches[pid].strain_energy = compute_strain_energy(state, patch, mat);
        apply_plastic_deformation(state, pid, strain);
    }
    return true;
}

float ElasticSolver::compute_strain_energy(const BodyState& state, const PatchTopology& patch,
                                            const MaterialProperties& material) const {
    float strain = compute_strain(state, patch);
    return 0.5f * material.youngs_modulus * strain * strain;
}

void ElasticSolver::propagate_strain(BodyState& state, const BodyTopology& topology) const {
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        float energy = state.patches[pid].strain_energy;
        for (AnchorId a : patch.anchors) {
            state.anchors[a].temperature += energy * 1e-10f;
        }
    }
}

} // namespace delirium::m7