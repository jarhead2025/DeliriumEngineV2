#pragma once

#include "delirium_m7/body_topology.hpp"
#include "delirium_m7/body_state.hpp"
#include "delirium_m7/surfel_cache.hpp"
#include "delirium_m7/basis_functions.hpp"
#include "delirium_m7/welded_seam.hpp"
#include "delirium_m7/material_registry.hpp"
#include "delirium_m7/modal_system.hpp"
#include "delirium_m7/elastic_solver.hpp"
#include "delirium_m7/damage_accumulator.hpp"
#include "delirium_m7/contact_detector.hpp"
#include "delirium_m7/thermal_simulator.hpp"
#include "delirium_m7/keyframe_animator.hpp"
#include "delirium_m7/animation_timeline.hpp"
#include "delirium_m7/serializer.hpp"
#include "delirium_m7/constraint_solver.hpp"
#include "delirium_m7/collision_response.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m7 {

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

    // M7: constraint management
    void add_constraint(Constraint c);
    const ConstraintSolver& constraint_solver() const { return constraint_solver_; }

    // M7: collision response params
    void set_collision_params(CollisionParams params);
    CollisionParams collision_params() const { return collision_params_; }

    // M7: sub-stepping config
    void set_substep_config(SubstepConfig config);
    SubstepConfig substep_config() const { return substep_config_; }

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
    void substep(float sub_dt);

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
    BasisEvaluator evaluator_;
    CacheUpdater cache_updater_;
    WeldedSeamSolver seam_solver_;

    // M7
    ConstraintSolver constraint_solver_;
    CollisionResponse collision_response_;
    CollisionParams collision_params_;
    SubstepConfig substep_config_;
};

} // namespace delirium::m7