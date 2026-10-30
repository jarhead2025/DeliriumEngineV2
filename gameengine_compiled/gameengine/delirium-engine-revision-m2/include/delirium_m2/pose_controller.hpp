#pragma once

#include "delirium_m2/articulation.hpp"

#include <string>
#include <vector>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// PoseController — writes target_pose into the rig, never raw anchors
// ---------------------------------------------------------------------------
//
// Controllers are the only path for external input (IK, gameplay, animation
// clips) to modify bone targets. They write PoseTarget entries that are
// blended into ArticulationRig::target_pose each frame.
//
// This keeps the invariant: animation / IK / gameplay → target_pose → spring
// solve → current_pose → anchor targets. Raw anchor positions are never
// written directly by a controller.
// ---------------------------------------------------------------------------

class PoseController {
public:
    virtual ~PoseController() = default;

    // Apply this controller's targets to the rig for the given frame.
    // dt is the simulation step; controllers may use it for time-based
    // interpolation or damping.
    virtual void apply(ArticulationRig& rig, float dt) = 0;

    // Enable/disable this controller
    virtual void set_active(bool active) { active_ = active; }
    virtual bool is_active() const { return active_; }

    // Blend weight for this controller [0, 1]
    virtual void set_weight(float w) { weight_ = std::max(0.0f, std::min(1.0f, w)); }
    virtual float get_weight() const { return weight_; }

protected:
    bool active_ = true;
    float weight_ = 1.0f;
};

// ---------------------------------------------------------------------------
// DirectPoseController — sets explicit target poses on named bones
// ---------------------------------------------------------------------------
class DirectPoseController : public PoseController {
public:
    void set_target(BoneId bone, Transform target);
    void clear_targets();
    void apply(ArticulationRig& rig, float dt) override;

private:
    struct TargetEntry {
        BoneId bone;
        Transform transform;
    };
    std::vector<TargetEntry> targets_;
};

// ---------------------------------------------------------------------------
// BlendPoseController — blends between two rigs (or two pose sets)
// ---------------------------------------------------------------------------
class BlendPoseController : public PoseController {
public:
    // Set the blend factor [0 = source A, 1 = source B]
    void set_blend(float t) { blend_ = std::max(0.0f, std::min(1.0f, t)); }

    // Define source A and source B target poses for a bone
    void set_source(BoneId bone, Transform a, Transform b);
    void clear_sources();
    void apply(ArticulationRig& rig, float dt) override;

private:
    struct BlendEntry {
        BoneId bone;
        Transform source_a;
        Transform source_b;
    };
    std::vector<BlendEntry> sources_;
    float blend_ = 0.0f;
};

} // namespace delirium::m2