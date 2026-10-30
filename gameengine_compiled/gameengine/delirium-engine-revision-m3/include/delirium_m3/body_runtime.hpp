#pragma once

#include "delirium_m3/body_topology.hpp"
#include "delirium_m3/body_state.hpp"
#include "delirium_m3/surfel_cache.hpp"
#include "delirium_m3/basis_functions.hpp"
#include "delirium_m3/welded_seam.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m3 {

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
// SurfaceEvaluator — M3: higher-order basis + welded seams
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
// BodyRuntime — M3 orchestrator
// ---------------------------------------------------------------------------
class BodyRuntime {
public:
    explicit BodyRuntime(BodyTopology topology);

    bool initialize(const std::vector<Vec3>& rest_positions,
                    std::string* error = nullptr);

    const BodyState& read_state() const { return states_[front_]; }
    Generation published_generation() const { return read_state().generation; }
    TopologyGeneration published_topology_generation() const {
        return read_state().topology_generation;
    }
    const BodyTopology& topology() const { return topology_; }

    // Step: mark dirty anchors, propagate to patches, publish
    bool step(float fixed_dt, std::string* error = nullptr);

    // Surface evaluation
    SurfacePoint evaluate_surface(PatchId patch_id, float u, float v) const;
    float compute_seam_error(PatchId patch_a, PatchId patch_b) const;

    // Welded seam solving
    bool solve_welded_seams(std::string* error = nullptr);

    // Cache update with temporal coherence
    bool update_cache(SurfelCache& cache, std::string* error = nullptr);
    bool rebuild_cache(SurfelCache& cache, std::string* error = nullptr);

private:
    void mark_incident_patches(BodyState& state, AnchorId anchor);

    BodyTopology topology_;
    std::array<BodyState, 2> states_;
    std::size_t front_ = 0;

    bool initialized_ = false;

    SurfaceEvaluator evaluator_;
    CacheUpdater cache_updater_;
    WeldedSeamSolver seam_solver_;
};

} // namespace delirium::m3