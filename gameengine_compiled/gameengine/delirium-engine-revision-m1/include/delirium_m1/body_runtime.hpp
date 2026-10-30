#pragma once

#include "delirium_m1/articulation.hpp"
#include "delirium_m1/body_topology.hpp"

#include <array>
#include <string>

namespace delirium::m1 {

class BodyRuntime {
public:
    explicit BodyRuntime(BodyTopology topology);

    bool initialize(const std::vector<Vec3>& rest_positions, std::string* error = nullptr);
    bool bind_rig(ArticulationRig rig, std::string* error = nullptr);

    // Reads only the published front state. This state is immutable to callers
    // until the next successful publish.
    const BodyState& read_state() const { return states_[front_]; }
    Generation published_generation() const { return read_state().generation; }
    const BodyTopology& topology() const { return topology_; }

    // Runs articulation into the non-published state, marks only affected
    // anchors/patches, and atomically publishes by swapping front/back indices.
    bool step_articulation(float fixed_dt, std::string* error = nullptr);

private:
    void mark_incident_patches(BodyState& state, AnchorId anchor);

    BodyTopology topology_;
    std::array<BodyState, 2> states_;
    std::size_t front_ = 0;
    ArticulationRig rig_;
    bool initialized_ = false;
    bool has_rig_ = false;
};

} // namespace delirium::m1
