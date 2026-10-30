#include "delirium_m8/body_runtime.hpp"
#include <algorithm>
#include <cmath>

namespace delirium::m8 {

BodyRuntime::BodyRuntime(BodyTopology topology, BodyId id)
    : topology_(std::move(topology)), body_id_(id) {}

bool BodyRuntime::initialize(const std::vector<Vec3>& rest_positions, std::string* error) {
    if (rest_positions.size() != topology_.anchor_count()) {
        if (error) *error = "rest position count mismatch";
        return false;
    }

    for (auto& s : states_) {
        s.resize(topology_.anchor_count(), topology_.patch_count());
        for (std::size_t i = 0; i < rest_positions.size(); ++i) {
            s.anchors[i].rest_position = rest_positions[i];
            s.anchors[i].position = rest_positions[i];
        }
    }

    initialized_ = true;
    return true;
}

void BodyRuntime::set_materials(MaterialRegistry materials) {
    materials_ = std::move(materials);
}

void BodyRuntime::set_modal_system(ModalSystem modal_system) {
    modal_system_ = std::move(modal_system);
}

void BodyRuntime::set_animation_timeline(AnimationTimeline timeline) {
    timeline_ = std::move(timeline);
}

void BodyRuntime::add_constraint(Constraint c) {
    constraint_solver_.add_constraint(std::move(c));
}

void BodyRuntime::set_collision_params(CollisionParams params) {
    collision_params_ = std::move(params);
    collision_response_ = CollisionResponse(collision_params_);
}

void BodyRuntime::set_substep_config(SubstepConfig config) {
    substep_config_ = std::move(config);
}

void BodyRuntime::set_external_forces(ExternalForce f) {
    external_forces_.set_force(std::move(f));
}

void BodyRuntime::set_fracture_params(FractureParams p) {
    fracture_system_.set_params(std::move(p));
}

FractureEvent BodyRuntime::check_fracture() const {
    return fracture_system_.evaluate(read_state(), topology_, body_id_);
}

void BodyRuntime::mark_incident_patches(BodyState& state, AnchorId anchor) {
    for (PatchId pid : topology_.patches_for_anchor(anchor)) {
        state.dirty_patches.mark(pid);
    }
}

void BodyRuntime::substep(float sub_dt) {
    BodyState& state = write_state();

    state.generation = read_state().generation + 1;
    state.topology_generation = topology_.topology_generation();

    for (std::size_t i = 0; i < state.anchors.size(); ++i) {
        state.anchors[i].position = read_state().anchors[i].position;
        state.anchors[i].velocity = read_state().anchors[i].velocity;
        state.anchors[i].rest_position = read_state().anchors[i].rest_position;
        state.anchors[i].temperature = read_state().anchors[i].temperature;
        state.anchors[i].damage = read_state().anchors[i].damage;
    }
    for (std::size_t i = 0; i < state.patches.size(); ++i) {
        state.patches[i] = read_state().patches[i];
    }

    timeline_.tick(sub_dt);
    for (std::size_t mi = 0; mi < modal_system_.count(); ++mi) {
        float amp = timeline_.get_amplitude(mi);
        for (auto& anchor : state.anchors) {
            float disp = modal_system_.evaluate_displacement(mi, time_);
            anchor.modal_displacement += anchor.rest_position * disp * amp * 0.01f;
        }
    }

    // M8: apply external forces
    external_forces_.apply(state, sub_dt);

    if (materials_.ids().empty() == false) {
        elastic_solver_.solve(state, topology_, materials_, sub_dt);
    }

    if (materials_.ids().empty() == false) {
        damage_accumulator_.accumulate(state, topology_, materials_, sub_dt);
    }

    if (materials_.ids().empty() == false) {
        thermal_simulator_.diffuse(state, topology_, materials_, sub_dt);
    }
    thermal_simulator_.cool(state, sub_dt);

    constraint_solver_.solve(state, sub_dt);

    for (auto& anchor : state.anchors) {
        anchor.velocity *= 0.99f;
        anchor.position += anchor.velocity * sub_dt;
    }

    elastic_solver_.propagate_strain(state, topology_);

    state.clear_dirty();
}

bool BodyRuntime::step(float fixed_dt, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }

    std::uint32_t num_substeps = substep_config_.max_substeps;
    float sub_dt = fixed_dt / static_cast<float>(num_substeps);

    for (std::uint32_t i = 0; i < num_substeps; ++i) {
        substep(sub_dt);
        front_ = 1 - front_;
    }

    time_ += fixed_dt;
    return true;
}

void BodyRuntime::apply_contact_forces(const std::vector<ContactPair>& contacts,
                                        float stiffness, float damping, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return;
    }

    BodyState& state = write_state();
    for (const auto& cp : contacts) {
        AnchorId nearest = 0;
        float min_dist = 1e30f;
        for (std::size_t i = 0; i < state.anchors.size(); ++i) {
            float d = length(state.anchors[i].position - cp.contact_point);
            if (d < min_dist) { min_dist = d; nearest = static_cast<AnchorId>(i); }
        }

        Vec3 force = cp.normal * cp.penetration * stiffness;
        force += state.anchors[nearest].velocity * damping;
        state.anchors[nearest].velocity -= force * 0.016f;
        mark_incident_patches(state, nearest);
    }
}

SurfacePoint BodyRuntime::evaluate_surface(PatchId patch_id, float u, float v) const {
    return evaluator_.evaluate(read_state(), topology_, patch_id, u, v);
}

float BodyRuntime::compute_seam_error(PatchId patch_a, PatchId patch_b) const {
    return evaluator_.compute_seam_error(read_state(), topology_, patch_a, patch_b);
}

bool BodyRuntime::solve_welded_seams(std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    return seam_solver_.solve(write_state(), topology_, error);
}

bool BodyRuntime::rebuild_cache(SurfelCache& cache, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    return cache_updater_.rebuild(cache, read_state(), topology_, materials_, error);
}

bool BodyRuntime::rebuild_cache_with_lod(SurfelCache& cache, float camera_distance,
                                          std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    return cache_updater_.rebuild_with_lod(cache, read_state(), topology_, materials_,
                                            camera_distance, error);
}

Vec3 BodyRuntime::bounding_center() const {
    const auto& state = read_state();
    if (state.anchors.empty()) return {};
    Vec3 sum{};
    for (const auto& a : state.anchors) {
        sum += a.position;
    }
    return sum * (1.0f / static_cast<float>(state.anchors.size()));
}

float BodyRuntime::bounding_radius() const {
    Vec3 center = bounding_center();
    float max_d = 0.0f;
    for (const auto& a : read_state().anchors) {
        float d = length(a.position - center);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

} // namespace delirium::m8