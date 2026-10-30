#pragma once

#include "delirium_m2/body_topology.hpp"
#include "delirium_m2/body_state.hpp"
#include "delirium_m2/articulation.hpp"
#include "delirium_m2/pose_controller.hpp"
#include "delirium_m2/surface_evaluator.hpp"
#include "delirium_m2/surfel_cache.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// BodyRuntime — M2 orchestrator
// ---------------------------------------------------------------------------
//
// Owns the full pipeline:
//   1. Pose controllers write target_pose into the rig
//   2. Articulation step advances bone springs (linear + angular)
//   3. Anchor positions are computed from bone influences
//   4. Dirty patches are propagated through adjacency
//   5. Surface evaluator reads the published state
//   6. Cache updater writes surfels tagged with the generation
//
// Double-buffered BodyState with atomic publication by buffer swap.
// ---------------------------------------------------------------------------

class BodyRuntime {
public:
    explicit BodyRuntime(BodyTopology topology);

    // Initialize both buffers with rest positions
    bool initialize(const std::vector<Vec3>& rest_positions,
                    std::string* error = nullptr);

    // Bind an articulation rig (validates against topology)
    bool bind_rig(ArticulationRig rig, std::string* error = nullptr);

    // Add a pose controller (IK, gameplay, animation)
    void add_controller(std::unique_ptr<PoseController> controller);

    // Read-only access to the published front state
    const BodyState& read_state() const { return states_[front_]; }
    Generation published_generation() const { return read_state().generation; }
    TopologyGeneration published_topology_generation() const {
        return read_state().topology_generation;
    }
    const BodyTopology& topology() const { return topology_; }

    // Full frame step:
    //   1. Apply all pose controllers to rig targets
    //   2. Step articulation springs (linear + angular)
    //   3. Compute anchor positions from bone influences
    //   4. Mark dirty anchors and propagate to patches
    //   5. Publish by swapping front/back buffers
    bool step(float fixed_dt, std::string* error = nullptr);

    // Surface evaluation helpers (read from published state)
    SurfacePoint evaluate_surface(PatchId patch_id, float u, float v) const;
    float compute_seam_error(PatchId patch_a, PatchId patch_b) const;

    // Cache update (reads published state, writes surfel cache)
    bool update_cache(SurfelCache& cache, std::string* error = nullptr);
    bool rebuild_cache(SurfelCache& cache, std::string* error = nullptr);

private:
    void mark_incident_patches(BodyState& state, AnchorId anchor);

    BodyTopology topology_;
    std::array<BodyState, 2> states_;
    std::size_t front_ = 0;

    ArticulationRig rig_;
    bool has_rig_ = false;
    bool initialized_ = false;

    std::vector<std::unique_ptr<PoseController>> controllers_;

    SurfaceEvaluator evaluator_;
    CacheUpdater cache_updater_;
};

} // namespace delirium::m2