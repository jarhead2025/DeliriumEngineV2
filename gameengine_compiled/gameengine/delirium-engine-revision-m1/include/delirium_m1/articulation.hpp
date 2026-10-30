#pragma once

#include "delirium_m1/body_state.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m1 {

struct SpringParams {
    float stiffness = 120.0f;
    float damping = 18.0f;
    float mass = 1.0f;
};

struct BoneState {
    BoneId parent = kInvalidBone;
    Transform bind_pose{};
    Transform target_pose{};
    Transform current_pose{};
    Vec3 linear_velocity{};
    SpringParams spring{};
};

struct AnchorInfluence {
    std::array<BoneId, 4> bones{ kInvalidBone, kInvalidBone, kInvalidBone, kInvalidBone };
    std::array<std::uint16_t, 4> weights{ 0, 0, 0, 0 };
};

class ArticulationRig {
public:
    BoneId add_bone(BoneId parent, Transform bind_pose, SpringParams spring = {});
    void set_target_pose(BoneId id, Transform target);
    bool bind_anchor(AnchorId anchor, AnchorInfluence influence, std::string* error = nullptr);
    bool validate(std::size_t anchor_count, std::string* error = nullptr) const;

    // Fixed-step semi-implicit spring solve. Translation is intentional M1
    // scope; M2 adds rotation, joint limits, IK, and collision constraints.
    void step(float fixed_dt);

    Vec3 target_position_for_anchor(AnchorId anchor, const Vec3& rest_position) const;
    const std::vector<BoneState>& bones() const { return bones_; }

private:
    std::vector<BoneState> bones_;
    std::vector<AnchorInfluence> anchor_influences_;
};

} // namespace delirium::m1
