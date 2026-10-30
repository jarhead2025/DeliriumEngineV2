#pragma once
#include "delirium_m7/types.hpp"
#include "delirium_m7/body_topology.hpp"
#include "delirium_m7/body_state.hpp"
#include "delirium_m7/material_registry.hpp"
#include <string>
namespace delirium::m7 {
class ElasticSolver {
public:
    explicit ElasticSolver(float default_stiffness = 1.0f);
    bool solve(BodyState& state, const BodyTopology& topology,
               const MaterialRegistry& materials, float fixed_dt, std::string* error = nullptr);
    float compute_strain_energy(const BodyState& state, const PatchTopology& patch,
                                const MaterialProperties& material) const;
    void propagate_strain(BodyState& state, const BodyTopology& topology) const;
private:
    float default_stiffness_;
    float compute_strain(const BodyState& state, const PatchTopology& patch) const;
    void apply_elastic_force(BodyState& state, const BodyTopology& topology, PatchId patch_id,
                             const MaterialProperties& material, float fixed_dt) const;
    void apply_plastic_deformation(BodyState& state, PatchId patch_id, float strain) const;
};
} // namespace delirium::m7