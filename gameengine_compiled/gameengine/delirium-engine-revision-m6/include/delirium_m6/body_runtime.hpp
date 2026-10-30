#pragma once

#include "delirium_m6/body_topology.hpp"
#include "delirium_m6/body_state.hpp"
#include "delirium_m6/surfel_cache.hpp"
#include "delirium_m6/basis_functions.hpp"
#include "delirium_m6/welded_seam.hpp"
#include "delirium_m6/material_registry.hpp"
#include "delirium_m6/modal_system.hpp"
#include "delirium_m6/elastic_solver.hpp"
#include "delirium_m6/damage_accumulator.hpp"
#include "delirium_m6/contact_detector.hpp"
#include "delirium_m6/thermal_simulator.hpp"
#include "delirium_m6/keyframe_animator.hpp"
#include "delirium_m6/animation_timeline.hpp"
#include "delirium_m6/serializer.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m6 {

struct SurfacePoint {
    Vec3 position, normal;
    MaterialId material;
    PatchId source_patch;
    float u, v;
};

class SurfaceEvaluator {
public:
    SurfacePoint evaluate_patch(const BodyState& state, const BodyTopology& topology,
                                PatchId patch_id, float u, float v) const;
    float compute_seam_error(const BodyState& state, const BodyTopology& topology,
                             PatchId patch_a, PatchId patch_b) const;
};

class BodyRuntime {
public:
    explicit BodyRuntime(BodyTopology topology, BodyId id = 0);

    bool initialize(const std::vector<Vec3>& rest_positions, std::string* error = nullptr);

    const BodyState& read_state() const { return states_[front_]; }
    BodyState& write_state() { return states_[1 - front_]; }
    Generation published_generation() const { return read_state().generation; }
    TopologyGeneration published_topology_generation() const { return read_state().topology_generation; }
    const BodyTopology& topology() const { return topology_; }
    BodyId body_id() const { return body_id_; }

    void set_materials(MaterialRegistry materials);
    const MaterialRegistry& materials() const { return materials_; }

    void set_modal_system(ModalSystem modal_system);
    void set_animation_timeline(AnimationTimeline timeline);
    const AnimationTimeline& animation_timeline() const { return timeline_; }

    bool step(float fixed_dt, std::string* error = nullptr);

    void apply_contact_forces(const std::vector<ContactPair>& contacts,
                              float stiffness, float damping, std::string* error = nullptr);

    SurfacePoint evaluate_surface(PatchId patch_id, float u, float v) const;
    float compute_seam_error(PatchId patch_a, PatchId patch_b) const;
    bool solve_welded_seams(std::string* error = nullptr);

    bool rebuild_cache(SurfelCache& cache, std::string* error = nullptr);
    bool rebuild_cache_with_lod(SurfelCache& cache, float camera_distance,
                                std::string* error = nullptr);

    float current_time() const { return time_; }
    void set_time(float t) { time_ = t; }

private:
    void mark_incident_patches(BodyState& state, AnchorId anchor);

    BodyTopology topology_;
    BodyId body_id_ = 0;
    std::array<BodyState, 2> states_;
    std::size_t front_ = 0;
    bool initialized_ = false;
    float time_ = 0.0f;

    MaterialRegistry materials_;
    ModalSystem modal_system_;
    AnimationTimeline timeline_;
    ElasticSolver elastic_solver_;
    DamageAccumulator damage_accumulator_;
    ThermalSimulator thermal_simulator_;
    SurfaceEvaluator evaluator_;
    CacheUpdater cache_updater_;
    WeldedSeamSolver seam_solver_;
};

} // namespace delirium::m6