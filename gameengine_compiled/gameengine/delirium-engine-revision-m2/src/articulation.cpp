#include "delirium_m2/articulation.hpp"

#include <cmath>
#include <algorithm>

namespace delirium::m2 {

namespace {
constexpr float kWeightTotal = 65535.0f;

bool valid_bone(BoneId id, std::size_t count) {
    return id != kInvalidBone && id < count;
}
} // namespace

BoneId ArticulationRig::add_bone(BoneId parent, Transform bind_pose,
                                 SpringParams linear_spring,
                                 AngularSpringParams angular_spring,
                                 JointLimits limits) {
    if (parent != kInvalidBone && parent >= bones_.size()) return kInvalidBone;
    BoneState bone;
    bone.parent = parent;
    bone.bind_pose = bind_pose;
    bone.target_pose = bind_pose;
    bone.current_pose = bind_pose;
    bone.linear_spring = linear_spring;
    bone.angular_spring = angular_spring;
    bone.limits = limits;
    bones_.push_back(bone);
    return static_cast<BoneId>(bones_.size() - 1);
}

void ArticulationRig::set_target_pose(BoneId id, Transform target) {
    if (id < bones_.size()) bones_[id].target_pose = target;
}

bool ArticulationRig::bind_anchor(AnchorId anchor, AnchorInfluence influence,
                                   std::string* error) {
    if (anchor >= anchor_influences_.size()) anchor_influences_.resize(anchor + 1);

    std::uint32_t sum = 0;
    for (std::size_t i = 0; i < influence.bones.size(); ++i) {
        const auto weight = influence.weights[i];
        if (weight == 0) continue;
        if (!valid_bone(influence.bones[i], bones_.size())) {
            if (error) *error = "anchor influence references invalid bone";
            return false;
        }
        sum += weight;
    }
    if (sum != static_cast<std::uint32_t>(kWeightTotal)) {
        if (error) *error = "anchor influence weights must sum to 65535";
        return false;
    }
    anchor_influences_[anchor] = influence;
    return true;
}

bool ArticulationRig::validate(std::size_t anchor_count, std::string* error) const {
    if (anchor_influences_.size() != anchor_count) {
        if (error) *error = "every anchor requires an explicit influence record";
        return false;
    }
    for (AnchorId anchor = 0; anchor < anchor_influences_.size(); ++anchor) {
        const AnchorInfluence& influence = anchor_influences_[anchor];
        std::uint32_t sum = 0;
        for (std::size_t i = 0; i < influence.bones.size(); ++i) {
            if (influence.weights[i] == 0) continue;
            if (!valid_bone(influence.bones[i], bones_.size())) {
                if (error) *error = "influence references invalid bone";
                return false;
            }
            sum += influence.weights[i];
        }
        if (sum != static_cast<std::uint32_t>(kWeightTotal)) {
            if (error) *error = "influence weights are not normalized";
            return false;
        }
    }
    return true;
}

void ArticulationRig::clamp_joint_limits(BoneState& bone) {
    if (bone.limits.is_unbounded()) return;

    // Clamp translation
    bone.current_pose.translation.x = std::clamp(bone.current_pose.translation.x,
                                                  bone.limits.min_translation.x,
                                                  bone.limits.max_translation.x);
    bone.current_pose.translation.y = std::clamp(bone.current_pose.translation.y,
                                                  bone.limits.min_translation.y,
                                                  bone.limits.max_translation.y);
    bone.current_pose.translation.z = std::clamp(bone.current_pose.translation.z,
                                                  bone.limits.min_translation.z,
                                                  bone.limits.max_translation.z);

    // Clamp rotation: extract Euler angles from quaternion, clamp, re-compose
    // For simplicity, clamp the angular velocity magnitude rather than
    // decomposing/quaternion-clamping (which has gimbal issues)
    float ang_speed = length(bone.angular_velocity);
    float max_ang_speed = std::min({
        std::min(std::abs(bone.limits.max_angle_x - bone.limits.min_angle_x),
                 std::abs(bone.limits.max_angle_y - bone.limits.min_angle_y)),
        std::abs(bone.limits.max_angle_z - bone.limits.min_angle_z)
    }) * 0.5f; // safety factor

    if (ang_speed > max_ang_speed && max_ang_speed > 0.0f) {
        bone.angular_velocity *= (max_ang_speed / ang_speed);
    }
}

void ArticulationRig::step(float fixed_dt) {
    if (fixed_dt <= 0.0f) return;

    for (BoneState& bone : bones_) {
        // --- Linear spring (semi-implicit Euler) ---
        const float mass = bone.linear_spring.mass > 1e-6f ? bone.linear_spring.mass : 1.0f;
        const Vec3 displacement = bone.target_pose.translation - bone.current_pose.translation;
        const Vec3 acceleration = displacement * (bone.linear_spring.stiffness / mass)
                                - bone.linear_velocity * (bone.linear_spring.damping / mass);
        bone.linear_velocity += acceleration * fixed_dt;
        bone.current_pose.translation += bone.linear_velocity * fixed_dt;

        // --- Angular spring (semi-implicit Euler on angular velocity) ---
        // Compute rotation error as the relative quaternion
        Quat relative = bone.current_pose.rotation.conjugate() * bone.target_pose.rotation;

        // Extract axis-angle from relative quaternion
        float ang = 0.0f;
        Vec3 axis{};
        Vec3 xyz{relative.x, relative.y, relative.z};
        float sin_half = length(xyz);

        if (sin_half > 1e-6f) {
            ang = 2.0f * std::atan2(sin_half, std::abs(relative.w));
            axis = xyz * (1.0f / sin_half);
            if (relative.w < 0.0f) ang = -ang; // handle sign
        }

        // Angular spring force
        const float inertia = bone.angular_spring.inertia > 1e-6f ? bone.angular_spring.inertia : 1.0f;
        Vec3 angular_accel = axis * (ang * bone.angular_spring.stiffness / inertia)
                           - bone.angular_velocity * (bone.angular_spring.damping / inertia);

        bone.angular_velocity += angular_accel * fixed_dt;

        // Integrate angular velocity into rotation
        float ang_speed = length(bone.angular_velocity);
        if (ang_speed > 1e-8f) {
            float delta_angle = ang_speed * fixed_dt;
            Vec3 delta_axis = bone.angular_velocity * (1.0f / ang_speed);
            Quat delta_quat = Quat::from_angle_axis(delta_angle, delta_axis);
            bone.current_pose.rotation = (bone.current_pose.rotation * delta_quat).normalized();
        }

        // --- Clamp joint limits ---
        clamp_joint_limits(bone);
    }
}

Vec3 ArticulationRig::target_position_for_anchor(AnchorId anchor, const Vec3& rest_position) const {
    if (anchor >= anchor_influences_.size()) return rest_position;
    const AnchorInfluence& influence = anchor_influences_[anchor];

    Vec3 blended_translation{};
    Vec3 blended_rotation_axis{};
    float blended_rotation_angle = 0.0f;

    for (std::size_t i = 0; i < influence.bones.size(); ++i) {
        if (influence.weights[i] == 0) continue;
        const BoneState& bone = bones_[influence.bones[i]];

        // Translation: delta from bind to current
        const Vec3 trans_delta = bone.current_pose.translation - bone.bind_pose.translation;

        // Rotation: relative rotation from bind to current
        Quat rel_rot = bone.bind_pose.rotation.conjugate() * bone.current_pose.rotation;

        // Accumulate weighted contributions
        float w = static_cast<float>(influence.weights[i]) / kWeightTotal;
        blended_translation += trans_delta * w;

        // For rotation blending, accumulate axis-angle weighted
        Vec3 rel_xyz{rel_rot.x, rel_rot.y, rel_rot.z};
        float sin_half = length(rel_xyz);
        if (sin_half > 1e-6f) {
            float a = 2.0f * std::atan2(sin_half, std::abs(rel_rot.w));
            Vec3 ax = rel_xyz * (1.0f / sin_half);
            if (rel_rot.w < 0.0f) a = -a;
            blended_rotation_angle += a * w;
            blended_rotation_axis += ax * w;
        }
    }

    // Apply blended rotation to rest position, then add blended translation
    Vec3 rotated_rest = rest_position;
    if (length(blended_rotation_axis) > 1e-6f && std::abs(blended_rotation_angle) > 1e-6f) {
        Vec3 norm_axis;
        float al = length(blended_rotation_axis);
        norm_axis = blended_rotation_axis * (1.0f / al);
        Quat blended_quat = Quat::from_angle_axis(blended_rotation_angle, norm_axis);
        rotated_rest = quat_rotate(blended_quat, rest_position);
    }

    return rotated_rest + blended_translation;
}

} // namespace delirium::m2