#include "delirium_m2/pose_controller.hpp"

#include <algorithm>
#include <cmath>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// DirectPoseController
// ---------------------------------------------------------------------------

void DirectPoseController::set_target(BoneId bone, Transform target) {
    // Update existing or add new
    for (auto& entry : targets_) {
        if (entry.bone == bone) {
            entry.transform = target;
            return;
        }
    }
    targets_.push_back({bone, target});
}

void DirectPoseController::clear_targets() {
    targets_.clear();
}

void DirectPoseController::apply(ArticulationRig& rig, float /*dt*/) {
    if (!active_) return;
    for (const auto& entry : targets_) {
        Transform blended = entry.transform;
        // Apply controller weight: blend toward identity when weight < 1
        if (get_weight() < 1.0f) {
            blended.translation *= get_weight();
            // For rotation, slerp toward identity
            if (get_weight() < 1.0f - 1e-5f) {
                // Simple approach: reduce rotation angle proportionally
                Vec3 rot_xyz{blended.rotation.x, blended.rotation.y, blended.rotation.z};
                float sin_half = length(rot_xyz);
                if (sin_half > 1e-6f) {
                    float angle = 2.0f * std::atan2(sin_half, std::abs(blended.rotation.w));
                    Vec3 axis = rot_xyz * (1.0f / sin_half);
                    if (blended.rotation.w < 0.0f) angle = -angle;
                    blended.rotation = Quat::from_angle_axis(angle * get_weight(), axis);
                }
            }
        }
        rig.set_target_pose(entry.bone, blended);
    }
}

// ---------------------------------------------------------------------------
// BlendPoseController
// ---------------------------------------------------------------------------

void BlendPoseController::set_source(BoneId bone, Transform a, Transform b) {
    for (auto& entry : sources_) {
        if (entry.bone == bone) {
            entry.source_a = a;
            entry.source_b = b;
            return;
        }
    }
    sources_.push_back({bone, a, b});
}

void BlendPoseController::clear_sources() {
    sources_.clear();
}

void BlendPoseController::apply(ArticulationRig& rig, float /*dt*/) {
    if (!active_) return;
    for (const auto& entry : sources_) {
        // Linear interpolation of translation
        Vec3 blended_trans = entry.source_a.translation * (1.0f - blend_)
                           + entry.source_b.translation * blend_;

        // Slerp of rotation
        Quat blended_rot;
        float cos_theta = entry.source_a.rotation.w * entry.source_b.rotation.w +
                         entry.source_a.rotation.x * entry.source_b.rotation.x +
                         entry.source_a.rotation.y * entry.source_b.rotation.y +
                         entry.source_a.rotation.z * entry.source_b.rotation.z;

        if (cos_theta > 0.9995f) {
            // Nearly identical — linear blend is fine
            blended_rot.w = entry.source_a.rotation.w * (1.0f - blend_) + entry.source_b.rotation.w * blend_;
            blended_rot.x = entry.source_a.rotation.x * (1.0f - blend_) + entry.source_b.rotation.x * blend_;
            blended_rot.y = entry.source_a.rotation.y * (1.0f - blend_) + entry.source_b.rotation.y * blend_;
            blended_rot.z = entry.source_a.rotation.z * (1.0f - blend_) + entry.source_b.rotation.z * blend_;
        } else {
            // Slerp
            float theta = std::acos(std::clamp(cos_theta, -1.0f, 1.0f));
            float sin_theta = std::sin(theta);
            if (sin_theta > 1e-6f) {
                float wa = std::sin((1.0f - blend_) * theta) / sin_theta;
                float wb = std::sin(blend_ * theta) / sin_theta;
                blended_rot.w = entry.source_a.rotation.w * wa + entry.source_b.rotation.w * wb;
                blended_rot.x = entry.source_a.rotation.x * wa + entry.source_b.rotation.x * wb;
                blended_rot.y = entry.source_a.rotation.y * wa + entry.source_b.rotation.y * wb;
                blended_rot.z = entry.source_a.rotation.z * wa + entry.source_b.rotation.z * wb;
            } else {
                blended_rot = entry.source_a.rotation;
            }
        }
        blended_rot.normalize();

        // Apply controller weight
        if (get_weight() < 1.0f) {
            blended_trans *= get_weight();
        }

        rig.set_target_pose(entry.bone, {blended_trans, blended_rot});
    }
}

} // namespace delirium::m2