#pragma once

#include "delirium_m4/types.hpp"
#include "delirium_m4/body_topology.hpp"
#include "delirium_m4/body_state.hpp"
#include "delirium_m4/material_registry.hpp"

#include <string>
#include <vector>

namespace delirium::m4 {

// ---------------------------------------------------------------------------
// ElasticSolver — M4: sparse local elastic/plastic solve
// ---------------------------------------------------------------------------
//
// Operates only on dirty patches. Computes strain energy per-patch based on
// material properties. Patches exceeding yield strength accumulate plastic
// deformation. Strain energy is non-negative.
//
// Strain propagation: adjacent patches sharing anchors distribute strain
// energy through shared anchor states.
// ---------------------------------------------------------------------------
class ElasticSolver {
public:
    explicit ElasticSolver(float default_stiffness = 1.0f);

    // Solve elastic/plastic deformation for dirty patches only
    // Returns true if all patches are within yield limits
    bool solve(BodyState& state,
               const BodyTopology& topology,
               const MaterialRegistry& materials,
               float fixed_dt,
               std::string* error = nullptr);

    // Compute strain energy for a single patch
    float compute_strain_energy(const BodyState& state,
                                const PatchTopology& patch,
                                const MaterialProperties& material) const;

    // Check if a patch has exceeded yield strength
    bool is_plastic(const BodyState& state,
                    const PatchTopology& patch,
                    const MaterialProperties& material) const;

    // Propagate strain energy between adjacent patches through shared anchors
    void propagate_strain(BodyState& state,
                          const BodyTopology& topology) const;

private:
    float default_stiffness_;

    // Compute displacement-based strain for a patch
    float compute_strain(const BodyState& state,
                         const PatchTopology& patch) const;

    // Apply elastic restoration force to a patch's anchors
    void apply_elastic_force(BodyState& state,
                             const BodyTopology& topology,
                             PatchId patch_id,
                             const MaterialProperties& material,
                             float fixed_dt) const;

    // Apply plastic deformation (permanent displacement) when yield exceeded
    void apply_plastic_deformation(BodyState& state,
                                   PatchId patch_id,
                                   float strain,
                                   const MaterialProperties& material) const;
};

} // namespace delirium::m4