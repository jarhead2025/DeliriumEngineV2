#include "delirium_m6/body_runtime.hpp"
#include "delirium_m6/spatial_grid.hpp"
#include "delirium_m6/basis_functions.hpp"
#include "delirium_m6/material_registry.hpp"
#include "delirium_m6/modal_system.hpp"
#include "delirium_m6/damage_accumulator.hpp"
#include "delirium_m6/contact_detector.hpp"
#include "delirium_m6/animation_timeline.hpp"
#include "delirium_m6/serializer.hpp"
#include "delirium_m6/keyframe_animator.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace delirium::m6;

namespace {
int failures = 0;
void expect(bool condition, const char* label) {
    if (condition) std::cout << "PASS: " << label << "\n";
    else { std::cerr << "FAIL: " << label << "\n"; ++failures; }
}
} // namespace

// M6 Test 1: Animation timeline plays and interpolates
void test_animation_timeline() {
    AnimationTimeline timeline;
    AnimationClip clip;
    clip.name = "breath";
    clip.duration = 2.0f;
    clip.mode_index = 0;
    clip.keyframes = {{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 0.0f}};
    timeline.add_clip(std::move(clip));

    timeline.play("breath", 1.0f);
    expect(timeline.is_playing(), "timeline is playing");
    expect(timeline.current_clip_name() == "breath", "current clip name is correct");

    timeline.tick(0.5f);
    expect(std::abs(timeline.get_amplitude(0) - 0.5f) < 0.01f, "amplitude at t=0.5 is 0.5");

    timeline.tick(0.5f);
    expect(std::abs(timeline.get_amplitude(0) - 1.0f) < 0.01f, "amplitude at t=1.0 is 1.0");

    timeline.pause();
    expect(!timeline.is_playing(), "timeline paused");

    timeline.stop();
    expect(timeline.current_time() == 0.0f, "timeline reset after stop");
}

// M6 Test 2: Timeline auto-stops at end of clip
void test_timeline_auto_stop() {
    AnimationTimeline timeline;
    AnimationClip clip;
    clip.name = "short";
    clip.duration = 1.0f;
    clip.mode_index = 0;
    clip.keyframes = {{0.0f, 0.0f}, {1.0f, 1.0f}};
    timeline.add_clip(std::move(clip));

    timeline.play("short", 1.0f);
    timeline.tick(1.0f);
    expect(!timeline.is_playing(), "timeline stopped at end of clip");
}

// M6 Test 3: Serialization round-trip for topology
void test_serialize_topology() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0, BasisType::Cupula);
    std::string error;
    expect(body.finalize(&error), "topology finalizes for serialization");

    Serializer serializer;
    auto blob = serializer.serialize_topology(body);
    expect(!blob.data.empty(), "serialized topology blob is non-empty");

    BodyTopology restored = serializer.deserialize_topology(blob);
    expect(restored.finalized(), "deserialized topology is finalized");
    expect(restored.anchor_count() == 4, "deserialized topology has correct anchor count");
    expect(restored.patch_count() == 1, "deserialized topology has correct patch count");
}

// M6 Test 4: Serialization round-trip for body state
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

// M6 Test 5: Serialization round-trip for materials
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
    expect(std::abs(restored.get(0).youngs_modulus - 200.0e9f) < 1.0f,
           "deserialized material has correct youngs modulus");
    expect(restored.get(0).r == 100, "deserialized material has correct color R");
}

// M6 Test 6: Animation timeline integrates with runtime
void test_timeline_with_runtime() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);

    ModalSystem modal;
    ModalMode breathe; breathe.type = ModalMode::Type::Breathing; breathe.amplitude = 1.0f;
    modal.add_mode(breathe);
    runtime.set_modal_system(std::move(modal));

    AnimationTimeline timeline;
    AnimationClip clip;
    clip.name = "fade";
    clip.duration = 2.0f;
    clip.mode_index = 0;
    clip.keyframes = {{0.0f, 0.0f}, {1.0f, 2.0f}, {2.0f, 0.0f}};
    timeline.add_clip(std::move(clip));
    timeline.play("fade", 1.0f);
    runtime.set_animation_timeline(std::move(timeline));

    runtime.set_materials(MaterialRegistry{});
    expect(runtime.step(0.016f, &error), "step with timeline succeeds");
    expect(runtime.animation_timeline().is_playing(), "timeline is playing after step");
}

// M4/M5 carryover tests (condensed)
void test_material_properties() {
    MaterialRegistry r;
    MaterialProperties m; m.youngs_modulus = 200.0e9f; m.poisson_ratio = 0.3f;
    m.density = 7850.0f; m.yield_strength = 250.0e6f;
    r.register_material(0, m);
    expect(std::abs(r.get(0).youngs_modulus - 200.0e9f) < 1.0f, "material lookup correct");
}

void test_damage_monotonic() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0,1,2,3}, 0);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    MaterialRegistry reg;
    MaterialProperties soft; soft.youngs_modulus = 1.0e3f; soft.yield_strength = 1.0f; soft.density = 100.0f;
    reg.register_material(0, soft); runtime.set_materials(std::move(reg));
    runtime.write_state().anchors[0].position = {0.5f, 0.0f, 0.0f};
    float prev = 0.0f; bool mono = true;
    for (int i = 0; i < 5; ++i) {
        runtime.step(0.016f, &error);
        if (runtime.read_state().patches[0].damage < prev - 1e-6f) { mono = false; break; }
        prev = runtime.read_state().patches[0].damage;
    }
    expect(mono, "damage monotonic");
}

void test_contact_detection() {
    std::vector<Surfel> a, b;
    Surfel sa; sa.position = {0,0,0}; sa.radius = 0.3f; sa.source_body = 0; a.push_back(sa);
    Surfel sb; sb.position = {0.2f,0,0}; sb.radius = 0.3f; sb.source_body = 1; b.push_back(sb);
    ContactDetector d;
    auto c = d.detect(a, b, 0, 1);
    expect(!c.empty(), "contact found");
    expect(c[0].penetration > 0.0f, "penetration positive");
}

void test_basis_evaluation() {
    Vec3 p00{0,0,0}, p10{1,0,0}, p01{0,1,0}, p11{1,1,0};
    std::array<float, 8> c{}; std::array<Vec3, 4> r{};
    Vec3 v = evaluate_patch_position(p00,p10,p01,p11,BasisType::Bilinear,c,r,0.5f,0.5f);
    expect(nearly_equal(v, {0.5f,0.5f,0.0f}), "bilinear center");
}

void test_spatial_grid() {
    std::vector<Surfel> s;
    for (int i = 0; i < 10; ++i) { Surfel x; x.position = {0.1f*i,0,0}; x.material = 0; s.push_back(x); }
    SpatialGrid g(8,8,8);
    expect(g.build(s) > 0, "grid has cells");
}

void test_topology_generation() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0,1,2,3}, 0);
    std::string error; body.finalize(&error);
    TopologyGeneration g1 = body.topology_generation();
    body.finalize(&error);
    expect(body.topology_generation() > g1, "generation increments");
}

void test_lod() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0,1,2,3}, 0);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    runtime.set_materials(MaterialRegistry{});
    SurfelCache cf, cl;
    runtime.rebuild_cache(cf, &error);
    runtime.rebuild_cache_with_lod(cl, 20.0f, &error);
    expect(cl.surfel_count() <= cf.surfel_count(), "LOD reduces surfels");
}

void test_seam_error() {
    BodyTopology body;
    for (int i = 0; i < 8; ++i) body.add_anchor();
    body.add_patch({0,1,2,3}, 0); body.add_patch({2,3,4,5}, 1);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0},{2,1,0},{2,0,0},{0,2,0},{1,2,0}}, &error);
    expect(runtime.compute_seam_error(0, 1) == 0.0f, "shared-anchor seam zero");
}

int main() {
    test_animation_timeline();
    test_timeline_auto_stop();
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

    std::cout << (failures ? "M6 TESTS FAILED\n" : "M6 TESTS PASSED\n");
    return failures ? 1 : 0;
}