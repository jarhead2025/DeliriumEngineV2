#pragma once

#include "delirium_m4/body_topology.hpp"
#include "delirium_m4/body_state.hpp"
#include "delirium_m4/surfel_cache.hpp"
#include "delirium_m4/basis_functions.hpp"
#include "delirium_m4/welded_seam.hpp"
#include "delirium_m4/material_registry.hpp"
#include "delirium_m4/modal_system.hpp"
#include "delirium_m4/elastic_solver.hpp"
#include "delirium_m4/damage_accumulator.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m4 {

// ---------------------------------------------------------------------------
// SurfacePoint — evaluated surface sample
// ---------------------------------------------------------------------------
struct SurfacePoint {
    Vec3 position;
    Vec3 normal;
    MaterialId material;
    PatchId source_patch;
    float u, v;
};

// ---------------------------------------------------------------------------
// SurfaceEvaluator — M4: basis + material-aware evaluation
// ---------------------------------------------------------------------------
class SurfaceEvaluator {
public:
    SurfacePoint evaluate_patch(const BodyState& state,
                                const BodyTopology& topology,
                                PatchId patch_id,
                                float u, float v) const;

    std::vector<SurfacePoint> evaluate_dirty_patches(
        const BodyState& state,
        const BodyTopology& topology,
        const DirtySet& dirty_patches,
        std::uint32_t grid_resolution) const;

    float compute_seam_error(const BodyState& state,
                             const BodyTopology& topology,
                             PatchId patch_a, PatchId patch_b) const;
};

// ---------------------------------------------------------------------------
// BodyRuntime — M4 orchestrator with full pipeline
// ---------------------------------------------------------------------------
//
// Pipeline order per step:
//   1. Mark dirty anchors (from rest position deviation)
//   2. Modal displacement (global elastic stage)
//   3. Local elastic/plastic solve (sparse, dirty patches only)
//   4. Damage accumulation (monotonic)
//   5. Strain propagation between adjacent patches
//   6. Publish new state
// ---------------------------------------------------------------------------
class BodyRuntime {
public:
    explicit BodyRuntime(BodyTopology topology);

    bool initialize(const std::vector<Vec3>& rest_positions,
                    std::string* error = nullptr);

    const BodyState& read_state() const { return states_[front_]; }
    BodyState& write_state() { return states_[1 - front_]; }
    Generation published_generation() const { return read_state().generation; }
    TopologyGeneration published_topology_generation() const {
        return read_state().topology_generation;
    }
    const BodyTopology& topology() const { return topology_; }

    // Set material registry (required before step for material-aware solve)
    void set_materials(MaterialRegistry materials);
    const MaterialRegistry& materials() const { return materials_; }

    // Set modal system (optional — no-op if empty)
    void set_modal_system(ModalSystem modal_system);
    const ModalSystem& modal_system() const { return modal_system_; }

    // Full pipeline step: mark dirty → modal → elastic → damage → strain propagation → publish
    bool step(float fixed_dt, std::string* error = nullptr);

    // Surface evaluation
    SurfacePoint evaluate_surface(PatchId patch_id, float u, float v) const;
    float compute_seam_error(PatchId patch_a, PatchId patch_b) const;

    // Welded seam solving
    bool solve_welded_seams(std::string* error = nullptr);

    // Cache update with material-driven colors
    bool update_cache(SurfelCache& cache, std::string* error = nullptr);
    bool rebuild_cache(SurfelCache& cache, std::string* error = nullptr);

    // Get current time for modal animation
    float current_time() const { return time_; }
    void set_time(float t) { time_ = t; }

private:
    void mark_incident_patches(BodyState& state, AnchorId anchor);

    BodyTopology topology_;
    std::array<BodyState, 2> states_;
    std::size_t front_ = 0;

    bool initialized_ = false;
    float time_ = 0.0f;

    MaterialRegistry materials_;
    ModalSystem modal_system_;
    ElasticSolver elastic_solver_;
    DamageAccumulator damage_accumulator_;
    SurfaceEvaluator evaluator_;
    CacheUpdater cache_updater_;
    WeldedSeamSolver seam_solver_;
};

} // namespace delirium::m4