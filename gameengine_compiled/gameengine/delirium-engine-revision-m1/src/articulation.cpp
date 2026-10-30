#include "delirium_m1/articulation.hpp"

#include <cmath>

namespace delirium::m1 {

namespace {
constexpr float kWeightTotal = 65535.0f;

bool valid_bone(BoneId id, std::size_t count) {
    return id != kInvalidBone && id < count;
}
} // namespace

BoneId ArticulationRig::add_bone(BoneId parent, Transform bind_pose, SpringParams spring) {
    if (parent != kInvalidBone && parent >= bones_.size()) return kInvalidBone;
    BoneState bone;
    bone.parent = parent;
    bone.bind_pose = bind_pose;
    bone.target_pose = bind_pose;
    bone.current_pose = bind_pose;
    bone.spring = spring;
    bones_.push_back(bone);
    return static_cast<BoneId>(bones_.size() - 1);
}

void ArticulationRig::set_target_pose(BoneId id, Transform target) {
    if (id < bones_.size()) bones_[id].target_pose = target;
}

bool ArticulationRig::bind_anchor(AnchorId anchor, AnchorInfluence influence, std::string* error) {
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

void ArticulationRig::step(float fixed_dt) {
    if (fixed_dt <= 0.0f) return;
    for (BoneState& bone : bones_) {
        const float mass = bone.spring.mass > 1e-6f ? bone.spring.mass : 1.0f;
        const Vec3 displacement = bone.target_pose.translation - bone.current_pose.translation;
        const Vec3 acceleration = displacement * (bone.spring.stiffness / mass)
                                - bone.linear_velocity * (bone.spring.damping / mass);
        bone.linear_velocity += acceleration * fixed_dt;
        bone.current_pose.translation += bone.linear_velocity * fixed_dt;
    }
}

Vec3 ArticulationRig::target_position_for_anchor(AnchorId anchor, const Vec3& rest_position) const {
    if (anchor >= anchor_influences_.size()) return rest_position;
    const AnchorInfluence& influence = anchor_influences_[anchor];
    Vec3 displacement{};
    for (std::size_t i = 0; i < influence.bones.size(); ++i) {
        if (influence.weights[i] == 0) continue;
        const BoneState& bone = bones_[influence.bones[i]];
        const Vec3 delta = bone.current_pose.translation - bone.bind_pose.translation;
        displacement += delta * (static_cast<float>(influence.weights[i]) / kWeightTotal);
    }
    return rest_position + displacement;
}

} // namespace delirium::m1
