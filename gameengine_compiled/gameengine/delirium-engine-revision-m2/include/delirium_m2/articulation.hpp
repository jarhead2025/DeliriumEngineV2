#pragma once

#include "delirium_m2/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// Bone with full transform, joint limits, and angular spring behavior
// ---------------------------------------------------------------------------
struct BoneState {
    BoneId parent = kInvalidBone;
    Transform bind_pose{};
    Transform target_pose{};
    Transform current_pose{};

    // Linear state (from M1)
    Vec3 linear_velocity{};
    SpringParams linear_spring{};

    // Angular state (M2 addition)
    Vec3 angular_velocity{};       // expressed as axis-angle rate
    AngularSpringParams angular_spring{};

    // Joint limits (M2 addition)
    JointLimits limits{};
};

// ---------------------------------------------------------------------------
// Anchor influence — explicit bone weights (no inverse-distance skinning)
// ---------------------------------------------------------------------------
struct AnchorInfluence {
    std::array<BoneId, 4> bones{ kInvalidBone, kInvalidBone, kInvalidBone, kInvalidBone };
    std::array<std::uint16_t, 4> weights{ 0, 0, 0, 0 };
};

// ---------------------------------------------------------------------------
// ArticulationRig — full joint hierarchy with rotation + limits
// ---------------------------------------------------------------------------
class ArticulationRig {
public:
    BoneId add_bone(BoneId parent, Transform bind_pose,
                    SpringParams linear_spring = {},
                    AngularSpringParams angular_spring = {},
                    JointLimits limits = {});

    void set_target_pose(BoneId id, Transform target);

    bool bind_anchor(AnchorId anchor, AnchorInfluence influence,
                     std::string* error = nullptr);

    bool validate(std::size_t anchor_count, std::string* error = nullptr) const;

    // Fixed-step semi-implicit spring solve with rotation + joint limits.
    // Linear and angular springs are integrated independently; joint limits
    // are clamped after each step.
    void step(float fixed_dt);

    // Compute the target position for an anchor by blending bone displacements
    // through stored influences. M2 adds rotational displacement: the anchor's
    // rest position is rotated by each bone's current orientation relative to
    // its bind orientation, then translated.
    Vec3 target_position_for_anchor(AnchorId anchor, const Vec3& rest_position) const;

    const std::vector<BoneState>& bones() const { return bones_; }
    std::size_t bone_count() const { return bones_.size(); }

private:
    void clamp_joint_limits(BoneState& bone);

    std::vector<BoneState> bones_;
    std::vector<AnchorInfluence> anchor_influences_;
};

} // namespace delirium::m2