#include "delirium_m1/body_runtime.hpp"

#include <utility>

namespace delirium::m1 {

BodyRuntime::BodyRuntime(BodyTopology topology) : topology_(std::move(topology)) {}

bool BodyRuntime::initialize(const std::vector<Vec3>& rest_positions, std::string* error) {
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

void BodyRuntime::mark_incident_patches(BodyState& state, AnchorId anchor) {
    for (PatchId patch : topology_.patches_for_anchor(anchor)) state.dirty_patches.mark(patch);
}

bool BodyRuntime::step_articulation(float fixed_dt, std::string* error) {
    if (!initialized_ || !has_rig_) {
        if (error) *error = "runtime requires initialized body and bound rig";
        return false;
    }
    if (fixed_dt <= 0.0f) {
        if (error) *error = "fixed step must be positive";
        return false;
    }

    const std::size_t back = 1 - front_;
    const BodyState& read = states_[front_];
    BodyState& write = states_[back];
    write = read;                       // write buffer starts as last complete state
    write.clear_dirty();

    rig_.step(fixed_dt);
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

    write.generation = read.generation + 1;
    front_ = back;                      // publication: readers now observe a complete generation
    return true;
}

} // namespace delirium::m1
