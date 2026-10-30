#pragma once

#include "delirium_m8/body_topology.hpp"
#include "delirium_m8/body_state.hpp"
#include "delirium_m8/surfel_cache.hpp"
#include "delirium_m8/basis_functions.hpp"
#include "delirium_m8/welded_seam.hpp"
#include "delirium_m8/material_registry.hpp"
#include "delirium_m8/modal_system.hpp"
#include "delirium_m8/elastic_solver.hpp"
#include "delirium_m8/damage_accumulator.hpp"
#include "delirium_m8/contact_detector.hpp"
#include "delirium_m8/thermal_simulator.hpp"
#include "delirium_m8/keyframe_animator.hpp"
#include "delirium_m8/animation_timeline.hpp"
#include "delirium_m8/serializer.hpp"
#include "delirium_m8/constraint_solver.hpp"
#include "delirium_m8/collision_response.hpp"
#include "delirium_m8/external_forces.hpp"
#include "delirium_m8/fracture_system.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m8 {

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

    void add_constraint(Constraint c);
    const ConstraintSolver& constraint_solver() const { return constraint_solver_; }

    void set_collision_params(CollisionParams params);
    CollisionParams collision_params() const { return collision_params_; }

    void set_substep_config(SubstepConfig config);
    SubstepConfig substep_config() const { return substep_config_; }

    void set_external_forces(ExternalForce f);
    ExternalForces& external_forces_mut() { return external_forces_; }

    void set_fracture_params(FractureParams p);
    FractureSystem& fracture_system_mut() { return fracture_system_; }
    FractureEvent check_fracture() const;

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

    // M8: bounding box for culling
    Vec3 bounding_center() const;
    float bounding_radius() const;

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

    ConstraintSolver constraint_solver_;
    CollisionResponse collision_response_;
    CollisionParams collision_params_;
    SubstepConfig substep_config_;

    ExternalForces external_forces_;
    FractureSystem fracture_system_;
};

} // namespace delirium::m8