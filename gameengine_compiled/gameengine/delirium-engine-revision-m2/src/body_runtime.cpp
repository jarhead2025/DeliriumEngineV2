#include "delirium_m2/body_runtime.hpp"

#include <algorithm>
#include <utility>

namespace delirium::m2 {

BodyRuntime::BodyRuntime(BodyTopology topology)
    : topology_(std::move(topology)) {}

bool BodyRuntime::initialize(const std::vector<Vec3>& rest_positions,
                              std::string* error) {
    if (!topology_.finalized()) {
        if (error) *error = "topology must be finalized before runtime initialization";
        return false;
    }
    if (rest_positions.size() != topology_.anchor_count()) {
        if (error) *error = "rest position count does not match topology anchor count";
        return false;
    }

    for (BodyState& state : states_) {
        state.resize(topology_.anchor_count(), topology_.patch_count());
        state.generation = 0;
        state.topology_generation = topology_.topology_generation();
        for (AnchorId id = 0; id < rest_positions.size(); ++id) {
            state.anchors[id].rest_position = rest_positions[id];
            state.anchors[id].position = rest_positions[id];
        }
        state.clear_dirty();
    }

    front_ = 0;
    initialized_ = true;
    return true;
}

bool BodyRuntime::bind_rig(ArticulationRig rig, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime must be initialized before binding a rig";
        return false;
    }
    if (!rig.validate(topology_.anchor_count(), error)) return false;
    rig_ = std::move(rig);
    has_rig_ = true;
    return true;
}

void BodyRuntime::add_controller(std::unique_ptr<PoseController> controller) {
    controllers_.push_back(std::move(controller));
}

void BodyRuntime::mark_incident_patches(BodyState& state, AnchorId anchor) {
    for (PatchId patch : topology_.patches_for_anchor(anchor)) {
        state.dirty_patches.mark(patch);
    }
}

bool BodyRuntime::step(float fixed_dt, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    if (fixed_dt <= 0.0f) {
        if (error) *error = "fixed step must be positive";
        return false;
    }

    // 1. Apply all pose controllers to rig targets
    for (auto& ctrl : controllers_) {
        if (ctrl->is_active()) {
            ctrl->apply(rig_, fixed_dt);
        }
    }

    // 2. Step articulation springs (linear + angular)
    if (has_rig_) {
        rig_.step(fixed_dt);
    }

    // 3-4. Compute anchor positions, mark dirty
    const std::size_t back = 1 - front_;
    const BodyState& read = states_[front_];
    BodyState& write = states_[back];
    write = read;
    write.clear_dirty();

    if (has_rig_) {
        for (AnchorId anchor = 0; anchor < write.anchors.size(); ++anchor) {
            AnchorState& state = write.anchors[anchor];
            const Vec3 next = rig_.target_position_for_anchor(anchor, state.rest_position);

            if (!nearly_equal(next, state.position)) {
                state.velocity = (next - state.position) * (1.0f / fixed_dt);
                state.position = next;
                write.dirty_anchors.mark(anchor);
                mark_incident_patches(write, anchor);
            } else {
                state.velocity = {};
            }
        }
    }

    // 5. Publish by swapping front/back buffers
    write.generation = read.generation + 1;
    write.topology_generation = topology_.topology_generation();
    front_ = back;

    return true;
}

SurfacePoint BodyRuntime::evaluate_surface(PatchId patch_id, float u, float v) const {
    return evaluator_.evaluate_patch(read_state(), topology_, patch_id, u, v);
}

float BodyRuntime::compute_seam_error(PatchId patch_a, PatchId patch_b) const {
    return evaluator_.compute_seam_error(read_state(), topology_, patch_a, patch_b);
}

bool BodyRuntime::update_cache(SurfelCache& cache, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    return cache_updater_.update(cache, read_state(), topology_, error);
}

bool BodyRuntime::rebuild_cache(SurfelCache& cache, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    return cache_updater_.rebuild(cache, read_state(), topology_, error);
}

} // namespace delirium::m2