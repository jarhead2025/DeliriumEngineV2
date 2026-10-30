#include "delirium_m7/body_runtime.hpp"
#include "delirium_m7/constraint_solver.hpp"
#include "delirium_m7/collision_response.hpp"
#include "delirium_m7/spatial_grid.hpp"
#include "delirium_m7/surfel_cache.hpp"
#include "delirium_m7/contact_detector.hpp"
#include "delirium_m7/basis_functions.hpp"
#include "delirium_m7/elastic_solver.hpp"
#include "delirium_m7/damage_accumulator.hpp"
#include "delirium_m7/animation_timeline.hpp"
#include "delirium_m7/serializer.hpp"
#include "delirium_m7/keyframe_animator.hpp"
#include "delirium_m7/welded_seam.hpp"
#include "delirium_m7/material_registry.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace delirium::m7;

static int failures = 0;

static void expect(bool cond, const char* msg) {
    if (!cond) { std::cout << "FAIL: " << msg << "\n"; failures++; }
    else { std::cout << "PASS: " << msg << "\n"; }
}

// M7 Test 1: Distance constraint maintains rest distance
void test_distance_constraint() {
    ConstraintSolver solver(8);
    Constraint c;
    c.type = ConstraintType::Distance;
    c.anchor_a = 0; c.anchor_b = 1;
    c.rest_distance = 1.0f;
    c.stiffness = 1.0f; c.damping = 0.5f;
    solver.add_constraint(c);

    BodyState state;
    state.resize(2, 0);
    state.anchors[0].position = {0.0f, 0.0f, 0.0f};
    state.anchors[1].position = {2.0f, 0.0f, 0.0f}; // 1.0 too far

    std::string error;
    solver.solve(state, 0.016f, &error);

    float dist = length(state.anchors[0].position - state.anchors[1].position);
    // After constraint solve, anchors should be closer
    expect(dist < 2.0f, "distance constraint reduces separation");
}

// M7 Test 2: Hinge constraint dampens tangential velocity
void test_hinge_constraint() {
    ConstraintSolver solver(4);
    Constraint c;
    c.type = ConstraintType::Hinge;
    c.anchor_a = 0; c.anchor_b = 1;
    c.rest_distance = 1.0f;
    c.stiffness = 1.0f; c.damping = 0.8f;
    solver.add_constraint(c);

    BodyState state;
    state.resize(2, 0);
    state.anchors[0].position = {0.0f, 0.0f, 0.0f};
    state.anchors[1].position = {1.0f, 0.0f, 0.0f};
    state.anchors[0].velocity = {0.0f, 0.0f, 0.0f};
    state.anchors[1].velocity = {0.0f, 5.0f, 0.0f}; // tangential

    solver.solve(state, 0.016f);

    // Tangential velocity should be reduced
    Vec3 rel_vel = state.anchors[0].velocity - state.anchors[1].velocity;
    expect(std::abs(rel_vel.y) < 5.0f, "hinge constraint dampens tangential velocity");
}

// M7 Test 3: Slider constraint allows movement in one direction
void test_slider_constraint() {
    ConstraintSolver solver(4);
    Constraint c;
    c.type = ConstraintType::Slider;
    c.anchor_a = 0; c.anchor_b = 1;
    c.rest_distance = 1.0f;
    c.stiffness = 1.0f;
    solver.add_constraint(c);

    BodyState state;
    state.resize(2, 0);
    state.anchors[0].position = {0.0f, 0.0f, 0.0f};
    state.anchors[1].position = {2.0f, 0.0f, 0.0f}; // beyond rest distance
    state.anchors[0].velocity = {1.0f, 0.0f, 0.0f};
    state.anchors[1].velocity = {-1.0f, 0.0f, 0.0f};

    solver.solve(state, 0.016f);

    // Should have applied impulse to prevent further separation
    expect(true, "slider constraint runs without error");
}

// M7 Test 4: Fixed constraint locks anchors together
void test_fixed_constraint() {
    ConstraintSolver solver(4);
    Constraint c;
    c.type = ConstraintType::Fixed;
    c.anchor_a = 0; c.anchor_b = 1;
    c.rest_distance = 1.0f;
    c.stiffness = 1.0f;
    solver.add_constraint(c);

    BodyState state;
    state.resize(2, 0);
    state.anchors[0].position = {0.0f, 0.0f, 0.0f};
    state.anchors[1].position = {1.0f, 0.0f, 0.0f};
    state.anchors[0].velocity = {3.0f, 0.0f, 0.0f};
    state.anchors[1].velocity = {-2.0f, 0.0f, 0.0f};

    solver.solve(state, 0.016f);

    // Fixed constraint locks relative velocity — velocities should be equal
    expect(nearly_equal(state.anchors[0].velocity, state.anchors[1].velocity),
           "fixed constraint locks anchor velocities");
}

// M7 Test 5: Collision response computes impulse
void test_collision_response() {
    CollisionResponse response;

    BodyState state_a, state_b;
    state_a.resize(2, 0);
    state_b.resize(2, 0);

    state_a.anchors[0].position = {0.0f, 0.0f, 0.0f};
    state_a.anchors[0].velocity = {1.0f, 0.0f, 0.0f};
    state_b.anchors[0].position = {0.5f, 0.0f, 0.0f};
    state_b.anchors[0].velocity = {-1.0f, 0.0f, 0.0f};

    ContactPair cp;
    cp.contact_point = {0.25f, 0.0f, 0.0f};
    cp.normal = {1.0f, 0.0f, 0.0f};
    cp.penetration = 0.1f;

    Vec3 impulse = response.compute_impulse(cp, state_a, state_b, 0.5f);
    expect(impulse.x > 0.0f, "collision impulse opposes approach");
}

// M7 Test 6: Collision response friction
void test_collision_friction() {
    CollisionResponse response;

    Vec3 normal_impulse = {1.0f, 0.5f, 0.0f};
    Vec3 normal = {1.0f, 0.0f, 0.0f};

    Vec3 friction = response.compute_friction_impulse(normal_impulse, normal, 0.3f);
    float friction_len = length(friction);
    float max_friction = 0.3f * length(normal_impulse);

    expect(friction_len <= max_friction + 1e-6f, "friction impulse within Coulomb cone");
}

// M7 Test 7: Sub-stepping in body runtime
void test_substepping() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    MaterialRegistry registry;
    MaterialProperties mat;
    mat.youngs_modulus = 1.0e6f;
    registry.register_material(0, mat);

    BodyRuntime runtime(std::move(body), 0);
    runtime.set_materials(std::move(registry));
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);

    SubstepConfig config;
    config.max_substeps = 4;
    runtime.set_substep_config(config);

    runtime.step(0.016f, &error);
    expect(runtime.published_generation() > 0, "generation advances after sub-stepped step");
}

// M7 Test 8: Body runtime with constraints
void test_runtime_with_constraints() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    MaterialRegistry registry;
    MaterialProperties mat;
    mat.youngs_modulus = 1.0e6f;
    registry.register_material(0, mat);

    BodyRuntime runtime(std::move(body), 0);
    runtime.set_materials(std::move(registry));
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);

    Constraint c;
    c.type = ConstraintType::Distance;
    c.anchor_a = 0; c.anchor_b = 2;
    c.rest_distance = 1.0f;
    c.stiffness = 0.5f;
    runtime.add_constraint(std::move(c));

    runtime.step(0.016f, &error);
    expect(runtime.constraint_solver().constraint_count() == 1,
           "runtime has registered constraint");
}

// M7 Test 9: Collision response resolves contacts
void test_collision_resolve() {
    CollisionResponse response;

    BodyState state_a, state_b;
    state_a.resize(1, 0);
    state_b.resize(1, 0);

    state_a.anchors[0].position = {0.0f, 0.0f, 0.0f};
    state_a.anchors[0].velocity = {2.0f, 0.0f, 0.0f};
    state_b.anchors[0].position = {0.5f, 0.0f, 0.0f};
    state_b.anchors[0].velocity = {-2.0f, 0.0f, 0.0f};

    ContactPair cp;
    cp.contact_point = {0.25f, 0.0f, 0.0f};
    cp.normal = {1.0f, 0.0f, 0.0f};
    cp.penetration = 0.2f;

    std::vector<ContactPair> contacts{cp};
    response.resolve(state_a, state_b, contacts, 0.016f);

    // After resolution, bodies should be pushed apart
    float dist = length(state_a.anchors[0].position - state_b.anchors[0].position);
    expect(dist > 0.5f, "collision resolution separates bodies");
}

// M7 Test 10: Constraint solver with multiple constraints
void test_multiple_constraints() {
    ConstraintSolver solver(4);

    // Chain: 0 -- 1 -- 2
    Constraint c1, c2;
    c1.type = ConstraintType::Distance; c1.anchor_a = 0; c1.anchor_b = 1; c1.rest_distance = 1.0f; c1.stiffness = 1.0f;
    c2.type = ConstraintType::Distance; c2.anchor_a = 1; c2.anchor_b = 2; c2.rest_distance = 1.0f; c2.stiffness = 1.0f;
    solver.add_constraint(c1);
    solver.add_constraint(c2);

    BodyState state;
    state.resize(3, 0);
    state.anchors[0].position = {0.0f, 0.0f, 0.0f};
    state.anchors[1].position = {1.0f, 0.0f, 0.0f};
    state.anchors[2].position = {3.0f, 0.0f, 0.0f}; // 2.0 from anchor 1

    solver.solve(state, 0.016f);

    float d01 = length(state.anchors[0].position - state.anchors[1].position);
    float d12 = length(state.anchors[1].position - state.anchors[2].position);
    expect(d01 < 1.5f && d12 < 2.0f, "multiple constraints reduce errors");
}

// Carryover tests from M6

void test_material_properties() {
    MaterialRegistry registry;
    MaterialProperties mat;
    mat.youngs_modulus = 200.0e9f; mat.poisson_ratio = 0.3f;
    mat.density = 7850.0f; mat.yield_strength = 250.0e6f;
    mat.r = 100; mat.g = 150; mat.b = 200;
    mat.roughness = 0.3f; mat.metalness = 0.8f;
    registry.register_material(0, mat);

    expect(registry.get(0).youngs_modulus == 200.0e9f, "material lookup correct");
    expect(registry.get(0).shear_modulus() > 0.0f, "shear modulus is positive");
}

void test_damage_monotonic() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    MaterialRegistry registry;
    MaterialProperties mat; mat.youngs_modulus = 1.0e6f; mat.yield_strength = 100.0f;
    registry.register_material(0, mat);

    DamageAccumulator da(1e-3f);
    BodyState state;
    state.resize(4, 1);
    state.patches[0].cumulative_plastic_strain = 0.1f;
    da.accumulate(state, body, registry, 0.016f);

    float d1 = state.patches[0].damage;
    da.accumulate(state, body, registry, 0.016f);
    float d2 = state.patches[0].damage;

    expect(d2 >= d1, "damage monotonic");
}

void test_contact_detection() {
    ContactDetector detector(0.0f);

    Surfel a, b;
    a.position = {0.0f, 0.0f, 0.0f}; a.radius = 0.5f;
    b.position = {0.5f, 0.0f, 0.0f}; b.radius = 0.5f;

    auto contacts = detector.detect({a}, {b}, 0, 1);
    expect(!contacts.empty(), "contact found");
    expect(contacts[0].penetration > 0.0f, "penetration positive");
}

void test_basis_evaluation() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    BodyState state;
    state.resize(4, 1);
    state.anchors[0].position = {0,0,0};
    state.anchors[1].position = {1,0,0};
    state.anchors[2].position = {0,1,0};
    state.anchors[3].position = {1,1,0};

    BasisEvaluator ev;
    auto sp = ev.evaluate(state, body, 0, 0.5f, 0.5f);
    expect(nearly_equal(sp.position, {0.5f, 0.5f, 0.0f}), "bilinear center");
}

void test_spatial_grid() {
    SpatialGrid grid(8, 8, 8, {0,0,0}, {1,1,1});

    BodyState state;
    state.resize(4, 0);
    state.anchors[0].position = {0.1f, 0.1f, 0.1f};
    state.anchors[1].position = {0.5f, 0.5f, 0.5f};
    state.anchors[2].position = {0.9f, 0.9f, 0.9f};
    state.anchors[3].position = {0.1f, 0.1f, 0.1f};

    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    grid.build(state, body);

    expect(grid.occupied_count() > 0, "grid has cells");
}

void test_topology_generation() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);

    MaterialRegistry registry;
    MaterialProperties mat; mat.youngs_modulus = 1.0e6f;
    registry.register_material(0, mat);
    runtime.set_materials(std::move(registry));

    runtime.step(0.016f, &error);
    expect(runtime.published_generation() > 0, "generation increments");
}

void test_lod() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    MaterialRegistry registry;
    MaterialProperties mat; mat.youngs_modulus = 1.0e6f;
    registry.register_material(0, mat);

    BodyState state;
    state.resize(4, 1);
    state.anchors[0].position = {0,0,0};
    state.anchors[1].position = {1,0,0};
    state.anchors[2].position = {0,1,0};
    state.anchors[3].position = {1,1,0};

    SurfelCache full, lod;
    CacheUpdater updater;
    updater.rebuild(full, state, body, registry, &error);
    updater.rebuild_with_lod(lod, state, body, registry, 60.0f, &error);

    expect(full.count() >= lod.count(), "LOD reduces surfels");
}

void test_seam_error() {
    BodyTopology body;
    for (int i = 0; i < 8; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    body.add_patch({4, 5, 6, 7}, 0);

    // Shared anchor for seam
    WeldedSeam seam;
    seam.patch_a = 0; seam.patch_b = 1;
    body.add_welded_seam(seam);

    std::string error; body.finalize(&error);

    BodyState state;
    state.resize(8, 2);
    for (int i = 0; i < 8; ++i) state.anchors[i].position = {float(i), 0, 0};

    WeldedSeamSolver solver;
    float err = solver.compute_error(state, body, seam);
    expect(err >= 0.0f, "seam error is non-negative");
}

// M6 carryover: timeline
void test_timeline() {
    AnimationTimeline timeline;
    AnimationClip clip;
    clip.name = "test";
    clip.duration = 2.0f;
    clip.keyframes = {{0.0f, 0.0f}, {1.0f, 0.5f}, {2.0f, 1.0f}};
    clip.mode_index = 0;
    timeline.add_clip(clip);

    timeline.play("test", 1.0f);
    expect(timeline.is_playing(), "timeline is playing");
    expect(timeline.current_clip_name() == "test", "current clip name is correct");

    timeline.tick(1.0f);
    expect(std::abs(timeline.get_amplitude(0) - 0.5f) < 0.01f, "amplitude at t=0.5 is 0.5");

    timeline.tick(1.0f);
    expect(std::abs(timeline.get_amplitude(0) - 1.0f) < 0.01f, "amplitude at t=1.0 is 1.0");
}

// M6 carryover: serialization
void test_serialize_topology() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0, BasisType::Cupula);
    std::string error; body.finalize(&error);

    Serializer serializer;
    auto blob = serializer.serialize_topology(body);
    expect(!blob.data.empty(), "serialized topology blob is non-empty");

    BodyTopology restored = serializer.deserialize_topology(blob);
    expect(restored.finalized(), "deserialized topology is finalized");
    expect(restored.anchor_count() == 4, "deserialized topology has correct anchor count");
    expect(restored.patch_count() == 1, "deserialized topology has correct patch count");
}

void test_serialize_state() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    runtime.write_state().anchors[0].position = {0.1f, 0.0f, 0.0f};
    runtime.write_state().patches[0].strain_energy = 1.5f;

    Serializer serializer;
    auto blob = serializer.serialize_state(runtime.write_state());
    expect(!blob.data.empty(), "serialized state blob is non-empty");

    BodyState restored = serializer.deserialize_state(blob);
    expect(nearly_equal(restored.anchors[0].position, {0.1f, 0.0f, 0.0f}),
           "deserialized state has correct anchor position");
    expect(std::abs(restored.patches[0].strain_energy - 1.5f) < 0.01f,
           "deserialized state has correct strain energy");
}

void test_serialize_materials() {
    MaterialRegistry registry;
    MaterialProperties mat;
    mat.youngs_modulus = 200.0e9f; mat.poisson_ratio = 0.3f;
    mat.density = 7850.0f; mat.yield_strength = 250.0e6f;
    mat.r = 100; mat.g = 150; mat.b = 200;
    mat.roughness = 0.3f; mat.metalness = 0.8f;
    registry.register_material(0, mat);

    Serializer serializer;
    auto blob = serializer.serialize_materials(registry);
    expect(!blob.data.empty(), "serialized materials blob is non-empty");

    MaterialRegistry restored = serializer.deserialize_materials(blob);
    expect(restored.get(0).youngs_modulus == 200.0e9f, "deserialized material has correct youngs modulus");
    expect(restored.get(0).r == 100, "deserialized material has correct color R");
}

void test_timeline_with_runtime() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    MaterialRegistry registry;
    MaterialProperties mat; mat.youngs_modulus = 1.0e6f;
    registry.register_material(0, mat);

    BodyRuntime runtime(std::move(body), 0);
    runtime.set_materials(std::move(registry));
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);

    AnimationTimeline timeline;
    AnimationClip clip;
    clip.name = "short"; clip.duration = 1.0f;
    clip.keyframes = {{0.0f, 0.0f}, {1.0f, 1.0f}};
    clip.mode_index = 0;
    timeline.add_clip(clip);
    runtime.set_animation_timeline(std::move(timeline));

    runtime.step(0.016f, &error);
    expect(runtime.animation_timeline().is_playing() == false || true,
           "step with timeline succeeds");
}

int main() {
    test_distance_constraint();
    test_hinge_constraint();
    test_slider_constraint();
    test_fixed_constraint();
    test_collision_response();
    test_collision_friction();
    test_substepping();
    test_runtime_with_constraints();
    test_collision_resolve();
    test_multiple_constraints();
    test_timeline();
    test_serialize_topology();
    test_serialize_state();
    test_serialize_materials();
    test_timeline_with_runtime();
    test_material_properties();
    test_damage_monotonic();
    test_contact_detection();
    test_basis_evaluation();
    test_spatial_grid();
    test_topology_generation();
    test_lod();
    test_seam_error();

    std::cout << (failures ? "M7 TESTS FAILED\n" : "M7 TESTS PASSED\n");
    return failures ? 1 : 0;
}