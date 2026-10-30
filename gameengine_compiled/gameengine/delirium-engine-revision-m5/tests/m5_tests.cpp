#include "delirium_m5/body_runtime.hpp"
#include "delirium_m5/spatial_grid.hpp"
#include "delirium_m5/basis_functions.hpp"
#include "delirium_m5/material_registry.hpp"
#include "delirium_m5/modal_system.hpp"
#include "delirium_m5/damage_accumulator.hpp"
#include "delirium_m5/contact_detector.hpp"
#include "delirium_m5/thermal_simulator.hpp"
#include "delirium_m5/keyframe_animator.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace delirium::m5;

namespace {
int failures = 0;
void expect(bool condition, const char* label) {
    if (condition) std::cout << "PASS: " << label << "\n";
    else { std::cerr << "FAIL: " << label << "\n"; ++failures; }
}
} // namespace

// M5 Test 1: Contact detection finds overlapping surfels between two bodies
void test_contact_detection() {
    std::vector<Surfel> surfels_a, surfels_b;
    Surfel sa; sa.position = {0.0f, 0.0f, 0.0f}; sa.radius = 0.3f; sa.source_body = 0;
    surfels_a.push_back(sa);
    Surfel sb; sb.position = {0.2f, 0.0f, 0.0f}; sb.radius = 0.3f; sb.source_body = 1;
    surfels_b.push_back(sb);

    ContactDetector detector;
    auto contacts = detector.detect(surfels_a, surfels_b, 0, 1);
    expect(!contacts.empty(), "contact detection finds overlapping surfels");
    expect(contacts[0].penetration > 0.0f, "penetration is positive for overlapping surfels");
}

// M5 Test 2: Contact forces propagate through elastic solver
void test_contact_forces_propagate() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "topology finalizes for contact test");

    BodyRuntime runtime(std::move(body), 0);
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes for contact test");

    MaterialRegistry registry;
    MaterialProperties mat; mat.youngs_modulus = 1.0e6f; mat.yield_strength = 5.0e5f; mat.density = 1000.0f;
    registry.register_material(0, mat);
    runtime.set_materials(std::move(registry));

    ContactPair cp; cp.body_a = 0; cp.contact_point = {0.5f, 0.5f, 0.0f}; cp.penetration = 0.1f;
    std::vector<ContactPair> contacts = {cp};
    runtime.apply_contact_forces(contacts, 1000.0f, 0.5f);

    expect(runtime.step(0.016f, &error), "step after contact forces succeeds");
}

// M5 Test 3: Thermal diffusion reduces temperature gradients over time
void test_thermal_diffusion() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "topology finalizes for thermal test");

    BodyRuntime runtime(std::move(body), 0);
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes for thermal test");

    MaterialRegistry registry;
    MaterialProperties mat; mat.youngs_modulus = 1.0e6f; mat.yield_strength = 5.0e5f; mat.density = 1000.0f;
    mat.thermal_conductivity = 1.0f; mat.thermal_expansion = 1e-5f;
    registry.register_material(0, mat);
    runtime.set_materials(std::move(registry));

    // Set high temperature on one patch
    runtime.write_state().patches[0].temperature = 100.0f;
    runtime.write_state().anchors[0].temperature = 50.0f;

    ThermalSimulator thermal;
    thermal.diffuse(runtime.write_state(), runtime.topology(), runtime.materials(), 1.0f);

    expect(runtime.write_state().anchors[0].temperature > 50.0f,
           "thermal diffusion moves anchor temperature toward patch average");

    // Cooling test
    thermal.cool(runtime.write_state(), runtime.topology(), 20.0f, 0.5f, 1.0f);
    expect(runtime.write_state().anchors[0].temperature < 50.0f,
           "cooling reduces temperature toward ambient");
}

// M5 Test 4: Animation keyframes interpolate modal amplitudes correctly
void test_keyframe_interpolation() {
    KeyframeAnimator animator;
    std::vector<Keyframe> frames = {{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 0.0f}};
    animator.set_keyframes(0, std::move(frames));

    expect(animator.has_keyframes(0), "animator has keyframes for mode 0");
    expect(std::abs(animator.get_amplitude(0, 0.0f)) < 0.01f, "amplitude at t=0 is 0");
    expect(std::abs(animator.get_amplitude(0, 1.0f) - 1.0f) < 0.01f, "amplitude at t=1 is 1");
    expect(std::abs(animator.get_amplitude(0, 0.5f) - 0.5f) < 0.01f, "amplitude at t=0.5 is 0.5 (linear)");
    expect(std::abs(animator.get_amplitude(0, 3.0f)) < 0.01f, "amplitude clamps to last keyframe after range");
}

// M5 Test 5: LOD reduces surfel count at distance without breaking continuity
void test_lod_reduces_surfels() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "topology finalizes for LOD test");

    BodyRuntime runtime(std::move(body), 0);
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes for LOD test");

    MaterialRegistry registry;
    registry.register_material(0, MaterialRegistry::default_material());
    runtime.set_materials(std::move(registry));

    SurfelCache cache_full, cache_lod;
    runtime.rebuild_cache(cache_full, &error);
    runtime.rebuild_cache_with_lod(cache_lod, 20.0f, &error);

    expect(cache_full.surfel_count() > 0, "full cache has surfels");
    expect(cache_lod.surfel_count() > 0, "LOD cache has surfels");
    expect(cache_lod.surfel_count() <= cache_full.surfel_count(),
           "LOD at distance produces fewer or equal surfels");
}

// M5 Test 6: Multi-body contact detection
void test_multi_body_contact() {
    std::vector<Surfel> a, b;
    for (int i = 0; i < 5; ++i) {
        Surfel s; s.position = {0.1f * i, 0, 0}; s.radius = 0.06f; s.source_body = 0; a.push_back(s);
    }
    for (int i = 0; i < 5; ++i) {
        Surfel s; s.position = {0.1f * i, 0.05f, 0}; s.radius = 0.06f; s.source_body = 1; b.push_back(s);
    }

    ContactDetector detector;
    auto contacts = detector.detect(a, b, 0, 1);
    expect(!contacts.empty(), "multi-body contact finds contacts between surfel arrays");
    expect(contacts[0].body_a == 0 && contacts[0].body_b == 1, "contact pair has correct body IDs");
}

// M4 carryover tests (condensed)
void test_material_properties() {
    MaterialRegistry registry;
    MaterialProperties stiff; stiff.youngs_modulus = 200.0e9f; stiff.poisson_ratio = 0.3f;
    stiff.density = 7850.0f; stiff.yield_strength = 250.0e6f;
    registry.register_material(0, stiff);
    expect(std::abs(registry.get(0).youngs_modulus - 200.0e9f) < 1.0f, "material lookup returns correct youngs modulus");
    expect(registry.get(0).shear_modulus() > 0.0f, "shear modulus is positive");
}

void test_damage_monotonic() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    MaterialRegistry registry;
    MaterialProperties soft; soft.youngs_modulus = 1.0e3f; soft.yield_strength = 1.0f; soft.density = 100.0f;
    registry.register_material(0, soft); runtime.set_materials(std::move(registry));
    runtime.write_state().anchors[0].position = {0.5f, 0.0f, 0.0f};
    float prev = 0.0f; bool mono = true;
    for (int i = 0; i < 5; ++i) {
        runtime.step(0.016f, &error);
        if (runtime.read_state().patches[0].damage < prev - 1e-6f) { mono = false; break; }
        prev = runtime.read_state().patches[0].damage;
    }
    expect(mono, "damage is monotonic across steps");
}

void test_modal_composition() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    ModalSystem modal;
    ModalMode breathe; breathe.type = ModalMode::Type::Breathing; breathe.amplitude = 1.0f;
    modal.add_mode(breathe); runtime.set_modal_system(std::move(modal));
    runtime.set_materials(MaterialRegistry{});
    runtime.step(0.016f, &error);
    bool has_modal = false;
    for (const auto& a : runtime.read_state().anchors)
        if (length(a.modal_displacement) > 1e-6f) { has_modal = true; break; }
    expect(has_modal, "modal displacement is non-zero");
}

void test_strain_non_negative() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    MaterialRegistry registry;
    MaterialProperties mat; mat.youngs_modulus = 1.0e6f; mat.yield_strength = 1.0e6f; mat.density = 1000.0f;
    registry.register_material(0, mat); runtime.set_materials(std::move(registry));
    runtime.write_state().anchors[0].position = {0.1f, 0.0f, 0.0f};
    runtime.step(0.016f, &error);
    bool ok = true;
    for (std::size_t i = 0; i < runtime.read_state().patches.size(); ++i)
        if (runtime.read_state().patches[i].strain_energy < 0.0f) ok = false;
    expect(ok, "strain energy is non-negative");
}

void test_material_colors() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    MaterialRegistry registry;
    MaterialProperties red; red.r = 255; red.g = 0; red.b = 0; red.roughness = 0.2f; red.metalness = 0.9f;
    red.youngs_modulus = 1.0e6f; red.yield_strength = 5.0e5f; red.density = 1000.0f;
    registry.register_material(0, red); runtime.set_materials(std::move(registry));
    SurfelCache cache; runtime.rebuild_cache(cache, &error);
    bool found = false;
    for (std::size_t i = 0; i < cache.surfel_count(); ++i)
        if (cache.surfel(i).r == 255 && cache.surfel(i).g == 0) { found = true; break; }
    expect(found, "surfels have material-driven color");
}

void test_basis_evaluation() {
    Vec3 p00{0,0,0}, p10{1,0,0}, p01{0,1,0}, p11{1,1,0};
    std::array<float, 8> coeffs{}; std::array<Vec3, 4> residuals{};
    Vec3 c = evaluate_patch_position(p00,p10,p01,p11,BasisType::Bilinear,coeffs,residuals,0.5f,0.5f);
    expect(nearly_equal(c, {0.5f,0.5f,0.0f}), "bilinear center is average");
    coeffs[4] = 0.5f;
    Vec3 cu = evaluate_patch_position(p00,p10,p01,p11,BasisType::Cupula,coeffs,residuals,0.5f,0.5f);
    expect(std::abs(cu.z) > 0.01f, "cupula adds height");
}

void test_seam_error() {
    BodyTopology body;
    for (int i = 0; i < 8; ++i) body.add_anchor();
    body.add_patch({0,1,2,3}, 0); body.add_patch({2,3,4,5}, 1);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0},{2,1,0},{2,0,0},{0,2,0},{1,2,0}}, &error);
    expect(runtime.compute_seam_error(0, 1) == 0.0f, "shared-anchor seam has zero error");
}

void test_spatial_grid() {
    std::vector<Surfel> surfels;
    for (int i = 0; i < 10; ++i) { Surfel s; s.position = {0.1f*i,0,0}; s.material = 0; surfels.push_back(s); }
    for (int i = 0; i < 10; ++i) { Surfel s; s.position = {0.1f*i,0,0}; s.material = 1; surfels.push_back(s); }
    SpatialGrid grid(8,8,8);
    expect(grid.build(surfels) > 0, "grid has occupied cells");
}

void test_topology_generation() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0,1,2,3}, 0);
    std::string error; body.finalize(&error);
    TopologyGeneration g1 = body.topology_generation();
    body.finalize(&error);
    expect(body.topology_generation() > g1, "topology generation increments");
}

void test_temporal_coherence() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0,1,2,3}, 0);
    std::string error; body.finalize(&error);
    BodyRuntime runtime(std::move(body), 0);
    runtime.initialize({{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, &error);
    runtime.set_materials(MaterialRegistry{});
    SurfelCache cache; runtime.rebuild_cache(cache, &error);
    expect(cache.surfel_count() > 0, "cache has surfels");
    runtime.step(1.0f/60.0f, &error);
    runtime.update_cache(cache, &error);
    expect(cache.surfel_count() > 0, "cache retains surfels after step");
}

int main() {
    test_contact_detection();
    test_contact_forces_propagate();
    test_thermal_diffusion();
    test_keyframe_interpolation();
    test_lod_reduces_surfels();
    test_multi_body_contact();
    test_material_properties();
    test_damage_monotonic();
    test_modal_composition();
    test_strain_non_negative();
    test_material_colors();
    test_basis_evaluation();
    test_seam_error();
    test_spatial_grid();
    test_topology_generation();
    test_temporal_coherence();

    std::cout << (failures ? "M5 TESTS FAILED\n" : "M5 TESTS PASSED\n");
    return failures ? 1 : 0;
}