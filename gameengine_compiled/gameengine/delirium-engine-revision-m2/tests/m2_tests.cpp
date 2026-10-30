#include "delirium_m2/body_runtime.hpp"
#include "delirium_m2/opencl_adapter.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace delirium::m2;

namespace {

int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) std::cout << "PASS: " << label << "\n";
    else { std::cerr << "FAIL: " << label << "\n"; ++failures; }
}

// Helper: create a simple 2-patch topology (shared edge at anchors 2,3)
BodyTopology make_two_patch_topology() {
    BodyTopology body;
    for (int i = 0; i < 6; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0, {0, 64});
    body.add_patch({2, 3, 4, 5}, 1, {64, 64});
    body.finalize();
    return body;
}

// Helper: single-bone influence
AnchorInfluence one_bone(BoneId bone) {
    AnchorInfluence influence;
    influence.bones[0] = bone;
    influence.weights[0] = 65535;
    return influence;
}

// Helper: two-bone influence (50/50 split)
AnchorInfluence two_bones(BoneId a, BoneId b) {
    AnchorInfluence influence;
    influence.bones[0] = a;
    influence.weights[0] = 32768; // ~half
    influence.bones[1] = b;
    influence.weights[1] = 32767; // ~half (sums to 65535)
    return influence;
}

} // namespace

// ============================================================================
// Test 1: Quaternion / rotation pose locality
// ============================================================================
void test_quaternion_rotation_locality() {
    // Create a 90-degree rotation around Z axis
    Quat q90 = Quat::from_angle_axis(M_PI_2, {0.0f, 0.0f, 1.0f});

    // Rotate (1, 0, 0) by 90° around Z → should be (0, 1, 0)
    Vec3 v = {1.0f, 0.0f, 0.0f};
    Vec3 rotated = quat_rotate(q90, v);
    expect(nearly_equal(rotated, {0.0f, 1.0f, 0.0f}), "90° Z rotation of (1,0,0) yields (0,1,0)");

    // Identity rotation should leave vector unchanged
    Vec3 v2 = {3.0f, -1.0f, 2.0f};
    Vec3 rotated2 = quat_rotate(Quat::identity(), v2);
    expect(nearly_equal(rotated2, v2), "identity quaternion preserves vector");

    // Compose two 90° rotations = 180° rotation
    Quat q180 = q90 * q90;
    Vec3 v3 = {1.0f, 0.0f, 0.0f};
    Vec3 rotated3 = quat_rotate(q180, v3);
    expect(nearly_equal(rotated3, {-1.0f, 0.0f, 0.0f}, 1e-4f),
           "two 90° Z rotations compose to 180°");

    // Transform::apply_to_point rotates then translates
    Transform t{{2.0f, 0.0f, 0.0f}, q90};
    Vec3 pt = t.apply_to_point({1.0f, 0.0f, 0.0f});
    expect(nearly_equal(pt, {2.0f, 1.0f, 0.0f}),
           "transform rotates (1,0,0)→(0,1,0) then adds (2,0,0) = (2,1,0)");

    // Transform::apply_to_direction rotates only (no translation)
    Vec3 dir = t.apply_to_direction({1.0f, 0.0f, 0.0f});
    expect(nearly_equal(dir, {0.0f, 1.0f, 0.0f}),
           "transform direction ignores translation");
}

// ============================================================================
// Test 2: Joint-limit convergence
// ============================================================================
void test_joint_limit_convergence() {
    ArticulationRig rig;

    // Bone with translation limits
    JointLimits limits;
    limits.min_translation = {-1.0f, -1.0f, -1.0f};
    limits.max_translation = {1.0f, 1.0f, 1.0f};

    // High-stiffness spring to drive the bone toward the limit
    SpringParams spring;
    spring.stiffness = 500.0f;
    spring.damping = 10.0f;
    spring.mass = 1.0f;

    BoneId bone = rig.add_bone(kInvalidBone, {}, spring, {}, limits);

    // Target is far outside the limit
    rig.set_target_pose(bone, {{10.0f, 0.0f, 0.0f}});

    // Step many times — position should be clamped within limits
    for (int i = 0; i < 300; ++i) {
        rig.step(1.0f / 120.0f);
    }

    const Vec3& pos = rig.bones()[bone].current_pose.translation;
    expect(pos.x >= limits.min_translation.x - 0.1f && pos.x <= limits.max_translation.x + 0.1f,
           "joint translation stays within limits after spring integration");
}

// ============================================================================
// Test 3: Deterministic fixed-step secondary motion
// ============================================================================
void test_deterministic_fixed_step_motion() {
    ArticulationRig rig;

    SpringParams spring;
    spring.stiffness = 100.0f;
    spring.damping = 15.0f;
    spring.mass = 1.0f;

    BoneId bone = rig.add_bone(kInvalidBone, {}, spring);
    rig.set_target_pose(bone, {{1.0f, 0.0f, 0.0f}});

    // Run two identical simulations
    auto run_simulation = [&]() -> Vec3 {
        ArticulationRig r;
        BoneId b = r.add_bone(kInvalidBone, {}, spring);
        r.set_target_pose(b, {{1.0f, 0.0f, 0.0f}});
        for (int i = 0; i < 120; ++i) r.step(1.0f / 120.0f);
        return r.bones()[b].current_pose.translation;
    };

    Vec3 result_a = run_simulation();
    Vec3 result_b = run_simulation();

    expect(nearly_equal(result_a, result_b),
           "fixed-step simulation is deterministic across runs");
    expect(result_a.x > 0.5f, "spring reaches significant displacement in 1 second");
}

// ============================================================================
// Test 4: Surface seam error after articulated pose
// ============================================================================
void test_surface_seam_error_after_pose() {
    BodyTopology topology = make_two_patch_topology();

    BodyRuntime runtime(std::move(topology));
    std::vector<Vec3> rest = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {2, 0, 0}
    };
    runtime.initialize(rest);

    ArticulationRig rig;
    BoneId bone_a = rig.add_bone(kInvalidBone, {});
    BoneId bone_b = rig.add_bone(kInvalidBone, {});

    // Anchors 0,1 driven by bone_a; anchors 4,5 driven by bone_b
    // Anchors 2,3 (shared) driven by both bones — this is the seam
    rig.bind_anchor(0, one_bone(bone_a));
    rig.bind_anchor(1, one_bone(bone_a));
    rig.bind_anchor(2, two_bones(bone_a, bone_b));
    rig.bind_anchor(3, two_bones(bone_a, bone_b));
    rig.bind_anchor(4, one_bone(bone_b));
    rig.bind_anchor(5, one_bone(bone_b));

    rig.set_target_pose(bone_a, {{0.5f, 0.0f, 0.0f}});
    rig.set_target_pose(bone_b, {{0.5f, 0.0f, 0.0f}});

    runtime.bind_rig(std::move(rig));

    // Step to move bones
    for (int i = 0; i < 120; ++i) runtime.step(1.0f / 120.0f);

    // Shared anchors belong to both patches → seam error should be zero
    // (they reference the same anchor in BodyState)
    float seam = runtime.compute_seam_error(0, 1);
    expect(seam < 1e-4f,
           "shared-anchor seam has zero error after articulation");

    // Verify surface evaluation returns valid points
    SurfacePoint sp = runtime.evaluate_surface(0, 0.5f, 0.5f);
    expect(sp.material == 0, "surface point carries correct material ID");
    expect(sp.source_patch == 0, "surface point carries correct patch ID");
    expect(length(sp.normal) > 0.5f, "surface normal has meaningful magnitude");
}

// ============================================================================
// Test 5: Dirty patch ranges → cache ranges
// ============================================================================
void test_dirty_patch_to_cache_ranges() {
    BodyTopology topology = make_two_patch_topology();

    BodyRuntime runtime(std::move(topology));
    std::vector<Vec3> rest = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {2, 0, 0}
    };
    runtime.initialize(rest);

    ArticulationRig rig;
    BoneId moving = rig.add_bone(kInvalidBone, {});
    BoneId fixed = rig.add_bone(kInvalidBone, {});

    // Only anchor 0 is driven by the moving bone
    rig.bind_anchor(0, one_bone(moving));
    for (AnchorId a = 1; a < 6; ++a) rig.bind_anchor(a, one_bone(fixed));

    rig.set_target_pose(moving, {{0.5f, 0.0f, 0.0f}});
    runtime.bind_rig(std::move(rig));

    runtime.step(1.0f / 60.0f);

    const BodyState& state = runtime.read_state();
    expect(state.dirty_anchors.marked(0), "moved anchor is dirty");
    expect(!state.dirty_anchors.marked(1), "unmoved anchor is clean");

    // Anchor 0 is in patch 0 only → only patch 0 should be dirty
    expect(state.dirty_patches.marked(0), "patch containing moved anchor is dirty");
    expect(!state.dirty_patches.marked(1), "unrelated patch is clean");

    // Cache rebuild should produce surfels for both patches (full rebuild)
    SurfelCache cache;
    runtime.rebuild_cache(cache);
    expect(cache.surfel_count() > 0, "rebuild produces surfels");
    expect(cache.matches_generation(state.generation, state.topology_generation),
           "cache generation matches body state generation");
}

// ============================================================================
// Test 6: Cache-generation mismatch rejected by upload adapter
// ============================================================================
void test_cache_generation_mismatch_rejected() {
    BodyTopology topology = make_two_patch_topology();

    BodyRuntime runtime(std::move(topology));
    std::vector<Vec3> rest = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {2, 0, 0}
    };
    runtime.initialize(rest);

    // Build a cache at generation 0
    SurfelCache cache;
    runtime.rebuild_cache(cache);
    Generation gen0 = cache.source_generation();

    // Step the runtime — generation advances
    runtime.step(1.0f / 60.0f);
    Generation gen1 = runtime.published_generation();

    expect(gen1 > gen0, "generation advances after step");

    // Try to upload old cache with new expected generation — should fail
    OpenCLUploadAdapter adapter(1); // stub: context handle = 1 (active)
    UploadResult result = adapter.upload_cache(cache, gen1, runtime.published_topology_generation());

    expect(!result.success, "upload rejected for cache-generation mismatch");
    expect(!result.error.empty(), "rejection includes error message");

    // Now rebuild cache at the new generation — should succeed
    runtime.rebuild_cache(cache);
    UploadResult result2 = adapter.upload_cache(cache, gen1, runtime.published_topology_generation());

    expect(result2.success, "upload accepted after cache rebuild at matching generation");
    expect(result2.source_generation == gen1, "uploaded generation matches body state");
}

// ============================================================================
// Test 7: Pose controllers write target_pose, not raw anchors
// ============================================================================
void test_pose_controller_invariant() {
    BodyTopology topology = make_two_patch_topology();
    BodyRuntime runtime(std::move(topology));

    std::vector<Vec3> rest = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {2, 0, 0}
    };
    runtime.initialize(rest);

    ArticulationRig rig;
    BoneId bone = rig.add_bone(kInvalidBone, {});
    for (AnchorId a = 0; a < 6; ++a) rig.bind_anchor(a, one_bone(bone));
    runtime.bind_rig(std::move(rig));

    // Add a direct pose controller
    auto ctrl = std::make_unique<DirectPoseController>();
    ctrl->set_target(0, {{1.0f, 0.0f, 0.0f}});
    runtime.add_controller(std::move(ctrl));

    // Step — controller should have written target_pose, spring should integrate
    runtime.step(1.0f / 60.0f);

    const BodyState& state = runtime.read_state();
    // Anchor should have moved toward the target (spring not fully converged in 1 step)
    expect(state.anchors[0].position.x > rest[0].x,
           "controller-driven bone moves influenced anchor");
    expect(state.generation == 1, "generation incremented after step with controller");
}

// ============================================================================
// Test 8: BlendPoseController
// ============================================================================
void test_blend_pose_controller() {
    ArticulationRig rig;
    BoneId bone = rig.add_bone(kInvalidBone, {});

    BlendPoseController blender;
    blender.set_source(0, {{0.0f, 0.0f, 0.0f}}, {{2.0f, 0.0f, 0.0f}});
    blender.set_blend(0.5f);
    blender.apply(rig, 1.0f / 60.0f);

    // Target should be blended: 0.5 * (0,0,0) + 0.5 * (2,0,0) = (1,0,0)
    expect(nearly_equal(rig.bones()[bone].target_pose.translation, {1.0f, 0.0f, 0.0f}),
           "blend controller interpolates between source poses");
}

// ============================================================================
// Test 9: Topology generation tracking
// ============================================================================
void test_topology_generation_tracking() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0, {0, 16});
    body.finalize();

    TopologyGeneration gen1 = body.topology_generation();
    expect(gen1 == 1, "first finalize increments topology generation");

    // Re-finalize (simulates topology change)
    body.finalize();
    TopologyGeneration gen2 = body.topology_generation();
    expect(gen2 > gen1, "re-finalize increments topology generation");
}

// ============================================================================
// Test 10: Angular spring behavior
// ============================================================================
void test_angular_spring_behavior() {
    ArticulationRig rig;

    AngularSpringParams ang_spring;
    ang_spring.stiffness = 300.0f;
    ang_spring.damping = 30.0f;
    ang_spring.inertia = 1.0f;

    BoneId bone = rig.add_bone(kInvalidBone, {}, {}, ang_spring);

    // Set a 90° target rotation around Z
    Quat target_q = Quat::from_angle_axis(M_PI_2, {0.0f, 0.0f, 1.0f});
    rig.set_target_pose(bone, {{}, target_q});

    // Step — angular spring should rotate toward target
    for (int i = 0; i < 120; ++i) rig.step(1.0f / 120.0f);

    // Check that rotation has progressed toward target
    // The current rotation should have a meaningful Z component
    const Quat& current = rig.bones()[bone].current_pose.rotation;
    expect(std::abs(current.z) > 0.1f || std::abs(current.w) < 0.99f,
           "angular spring rotates bone toward target orientation");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_quaternion_rotation_locality();
    test_joint_limit_convergence();
    test_deterministic_fixed_step_motion();
    test_surface_seam_error_after_pose();
    test_dirty_patch_to_cache_ranges();
    test_cache_generation_mismatch_rejected();
    test_pose_controller_invariant();
    test_blend_pose_controller();
    test_topology_generation_tracking();
    test_angular_spring_behavior();

    std::cout << (failures ? "M2 TESTS FAILED\n" : "M2 TESTS PASSED\n");
    return failures ? 1 : 0;
}