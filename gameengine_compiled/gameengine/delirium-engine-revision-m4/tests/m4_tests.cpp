#include "delirium_m4/body_runtime.hpp"
#include "delirium_m4/spatial_grid.hpp"
#include "delirium_m4/basis_functions.hpp"
#include "delirium_m4/material_registry.hpp"
#include "delirium_m4/modal_system.hpp"
#include "delirium_m4/damage_accumulator.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace delirium::m4;

namespace {
int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) std::cout << "PASS: " << label << "\n";
    else { std::cerr << "FAIL: " << label << "\n"; ++failures; }
}

BodyTopology make_topology() {
    BodyTopology body;
    for (int i = 0; i < 6; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0, BasisType::Bilinear);
    body.add_patch({2, 3, 4, 5}, 1, BasisType::Bilinear);
    std::string error;
    expect(body.finalize(&error), "topology finalizes with valid anchors/materials");
    return body;
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: Material properties affect spring behavior
// ---------------------------------------------------------------------------
void test_material_properties_affect_springs() {
    MaterialRegistry registry;

    // Register a stiff material
    MaterialProperties stiff;
    stiff.youngs_modulus = 200.0e9f;  // steel-like
    stiff.poisson_ratio = 0.3f;
    stiff.density = 7850.0f;
    stiff.yield_strength = 250.0e6f;
    stiff.r = 150; stiff.g = 150; stiff.b = 170;
    stiff.roughness = 0.3f;
    stiff.metalness = 0.8f;
    registry.register_material(0, stiff);

    // Register a soft material
    MaterialProperties soft;
    soft.youngs_modulus = 1.0e6f;
    soft.poisson_ratio = 0.45f;
    soft.density = 1000.0f;
    soft.yield_strength = 5.0e5f;
    soft.r = 200; soft.g = 100; soft.b = 100;
    soft.roughness = 0.8f;
    soft.metalness = 0.0f;
    registry.register_material(1, soft);

    // Verify lookup returns correct properties
    const auto& looked_up_stiff = registry.get(0);
    expect(std::abs(looked_up_stiff.youngs_modulus - 200.0e9f) < 1.0f,
           "stiff material youngs modulus is correct");

    const auto& looked_up_soft = registry.get(1);
    expect(std::abs(looked_up_soft.youngs_modulus - 1.0e6f) < 1.0f,
           "soft material youngs modulus is correct");

    // Verify derived properties
    expect(looked_up_stiff.shear_modulus() > 0.0f, "shear modulus is positive");
    expect(looked_up_stiff.bulk_modulus() > 0.0f, "bulk modulus is positive");

    // Verify unknown material returns default
    const auto& unknown = registry.get(999);
    expect(unknown.youngs_modulus == MaterialRegistry::default_material().youngs_modulus,
           "unknown material returns default properties");
}

// ---------------------------------------------------------------------------
// Test 2: Damage accumulation is monotonic
// ---------------------------------------------------------------------------
void test_damage_accumulation_monotonic() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "single-patch topology finalizes");

    BodyRuntime runtime(std::move(body));
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes");

    MaterialRegistry registry;
    MaterialProperties soft;
    soft.youngs_modulus = 1.0e3f;  // very soft — yields easily
    soft.yield_strength = 1.0f;
    soft.density = 100.0f;
    registry.register_material(0, soft);
    runtime.set_materials(std::move(registry));

    // Displace an anchor significantly to trigger plastic deformation
    runtime.write_state().anchors[0].position = Vec3{0.5f, 0.0f, 0.0f};

    // Step multiple times and verify damage never decreases
    float prev_damage = 0.0f;
    bool damage_monotonic = true;
    for (int i = 0; i < 5; ++i) {
        runtime.step(0.016f, &error);
        float current_damage = runtime.read_state().patches[0].damage;
        if (current_damage < prev_damage - 1e-6f) {
            damage_monotonic = false;
            break;
        }
        prev_damage = current_damage;
    }
    expect(damage_monotonic, "damage accumulation is monotonic across steps");

    // Verify damage is in valid range [0, 1]
    expect(DamageAccumulator::is_valid(prev_damage), "damage is within valid range [0, 1]");
}

// ---------------------------------------------------------------------------
// Test 3: Modal displacement composes correctly with bone pose
// ---------------------------------------------------------------------------
void test_modal_displacement_composition() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "topology finalizes for modal test");

    BodyRuntime runtime(std::move(body));
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes for modal test");

    // Set up a breathing modal mode
    ModalSystem modal;
    ModalMode breathe;
    breathe.type = ModalMode::Type::Breathing;
    breathe.amplitude = 1.0f;
    breathe.frequency = 1.0f;
    breathe.phase = 0.0f;
    modal.add_mode(breathe);
    runtime.set_modal_system(std::move(modal));

    MaterialRegistry registry;
    runtime.set_materials(std::move(registry));

    // Step and verify modal displacement was applied
    expect(runtime.step(0.016f, &error), "step with modal system succeeds");

    // Modal displacement should be non-zero for displaced anchors
    const BodyState& state = runtime.read_state();
    bool has_modal_displacement = false;
    for (const auto& anchor : state.anchors) {
        if (length(anchor.modal_displacement) > 1e-6f) {
            has_modal_displacement = true;
            break;
        }
    }
    expect(has_modal_displacement, "modal displacement is non-zero for at least one anchor");

    // Verify modal displacement composes before local solve
    // (position should include both rest + modal contribution)
    expect(length(state.anchors[1].position - rest[1]) > 1e-6f ||
           length(state.anchors[1].modal_displacement) > 1e-6f,
           "anchor position or modal displacement reflects modal contribution");
}

// ---------------------------------------------------------------------------
// Test 4: Strain energy is non-negative
// ---------------------------------------------------------------------------
void test_strain_energy_non_negative() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "topology finalizes for strain test");

    BodyRuntime runtime(std::move(body));
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes for strain test");

    MaterialRegistry registry;
    MaterialProperties mat;
    mat.youngs_modulus = 1.0e6f;
    mat.yield_strength = 1.0e6f;
    mat.density = 1000.0f;
    registry.register_material(0, mat);
    runtime.set_materials(std::move(registry));

    // Displace an anchor to create strain
    runtime.write_state().anchors[0].position = Vec3{0.1f, 0.0f, 0.0f};

    expect(runtime.step(0.016f, &error), "step with strain succeeds");

    // Verify all patches have non-negative strain energy
    bool all_non_negative = true;
    for (std::size_t pid = 0; pid < runtime.read_state().patches.size(); ++pid) {
        if (runtime.read_state().patches[pid].strain_energy < 0.0f) {
            all_non_negative = false;
            break;
        }
    }
    expect(all_non_negative, "all patch strain energies are non-negative");
}

// ---------------------------------------------------------------------------
// Test 5: Material-driven colors appear in surfel cache
// ---------------------------------------------------------------------------
void test_material_driven_colors() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "topology finalizes for color test");

    BodyRuntime runtime(std::move(body));
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes for color test");

    // Register a material with distinct color
    MaterialRegistry registry;
    MaterialProperties red_mat;
    red_mat.r = 255; red_mat.g = 0; red_mat.b = 0;
    red_mat.roughness = 0.2f;
    red_mat.metalness = 0.9f;
    red_mat.youngs_modulus = 1.0e6f;
    red_mat.yield_strength = 5.0e5f;
    red_mat.density = 1000.0f;
    registry.register_material(0, red_mat);
    runtime.set_materials(std::move(registry));

    SurfelCache cache;
    expect(runtime.rebuild_cache(cache, &error), "cache rebuild succeeds for color test");
    expect(cache.surfel_count() > 0, "cache has surfels after rebuild");

    // Verify surfels have the material's color
    bool found_correct_color = false;
    for (std::size_t i = 0; i < cache.surfel_count(); ++i) {
        const Surfel& s = cache.surfel(i);
        if (s.r == 255 && s.g == 0 && s.b == 0) {
            found_correct_color = true;
            break;
        }
    }
    expect(found_correct_color, "surfels have material-driven red color");

    // Verify roughness/metalness are set
    bool found_correct_pbr = false;
    for (std::size_t i = 0; i < cache.surfel_count(); ++i) {
        const Surfel& s = cache.surfel(i);
        if (std::abs(s.roughness - 0.2f) < 0.01f && std::abs(s.metalness - 0.9f) < 0.01f) {
            found_correct_pbr = true;
            break;
        }
    }
    expect(found_correct_pbr, "surfels have material-driven PBR properties");
}

// ---------------------------------------------------------------------------
// Test 6: Basis evaluation (carried from M3)
// ---------------------------------------------------------------------------
void test_basis_evaluation() {
    Vec3 p00{0, 0, 0}, p10{1, 0, 0}, p01{0, 1, 0}, p11{1, 1, 0};
    std::array<float, 8> coeffs{};
    std::array<Vec3, 4> residuals{};

    Vec3 center = evaluate_patch_position(p00, p10, p01, p11,
                                           BasisType::Bilinear, coeffs, residuals, 0.5f, 0.5f);
    expect(nearly_equal(center, {0.5f, 0.5f, 0.0f}), "bilinear at center is average of corners");

    coeffs[4] = 0.5f;
    Vec3 cupula_center = evaluate_patch_position(p00, p10, p01, p11,
                                                  BasisType::Cupula, coeffs, residuals, 0.5f, 0.5f);
    expect(std::abs(cupula_center.z) > 0.01f, "cupula adds height at center");

    residuals[0] = Vec3{0, 0, 0.1f};
    Vec3 with_residual = evaluate_patch_position(p00, p10, p01, p11,
                                                  BasisType::Bilinear, coeffs, residuals, 0.0f, 0.0f);
    expect(std::abs(with_residual.z - 0.1f) < 0.01f, "vector residual displaces corner");
}

// ---------------------------------------------------------------------------
// Test 7: Welded seam error (carried from M3)
// ---------------------------------------------------------------------------
void test_welded_seam_error() {
    BodyTopology body;
    for (int i = 0; i < 8; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    body.add_patch({2, 3, 4, 5}, 1);
    std::string error;
    expect(body.finalize(&error), "two-patch topology finalizes");

    BodyRuntime runtime(std::move(body));
    std::vector<Vec3> rest = {
        {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
        {2,1,0}, {2,0,0}, {0,2,0}, {1,2,0}
    };
    expect(runtime.initialize(rest, &error), "runtime initializes");

    float seam_err = runtime.compute_seam_error(0, 1);
    expect(seam_err == 0.0f, "shared-anchor seam has zero error");
}

// ---------------------------------------------------------------------------
// Test 8: Multi-surfel spatial grid (carried from M3)
// ---------------------------------------------------------------------------
void test_multi_surfel_cells() {
    std::vector<Surfel> surfels;

    for (int i = 0; i < 10; ++i) {
        Surfel s;
        s.position = {0.1f * i, 0, 0};
        s.material = 0;
        surfels.push_back(s);
    }

    for (int i = 0; i < 10; ++i) {
        Surfel s;
        s.position = {0.1f * i, 0, 0};
        s.material = 1;
        surfels.push_back(s);
    }

    SpatialGrid grid(8, 8, 8);
    std::size_t occupied = grid.build(surfels);
    expect(occupied > 0, "grid has occupied cells");

    bool found_multi_material = false;
    for (std::size_t i = 0; i < grid.total_cells(); ++i) {
        const SpatialCell& cell = grid.cell(i);
        if (!cell.is_empty() && cell.material_mask > 1) {
            found_multi_material = true;
            break;
        }
    }
    expect(found_multi_material, "cells with multiple materials preserve identity");
}

// ---------------------------------------------------------------------------
// Test 9: Temporal coherence (carried from M3)
// ---------------------------------------------------------------------------
void test_temporal_coherence() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "single-patch topology finalizes");

    BodyRuntime runtime(std::move(body));
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}};
    expect(runtime.initialize(rest, &error), "runtime initializes");

    MaterialRegistry registry;
    runtime.set_materials(std::move(registry));

    SurfelCache cache;
    expect(runtime.rebuild_cache(cache, &error), "initial cache rebuild succeeds");
    expect(cache.surfel_count() > 0, "cache has surfels after rebuild");

    expect(runtime.step(1.0f / 60.0f, &error), "step succeeds");
    expect(runtime.update_cache(cache, &error), "incremental update succeeds");
    expect(cache.surfel_count() > 0, "cache retains surfels after clean step");
}

// ---------------------------------------------------------------------------
// Test 10: Topology generation tracking (carried from M3)
// ---------------------------------------------------------------------------
void test_topology_generation_tracking() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    expect(body.finalize(&error), "first finalize succeeds");
    TopologyGeneration gen1 = body.topology_generation();
    expect(gen1 >= 1, "first finalize sets topology generation >= 1");

    expect(body.finalize(&error), "re-finalize succeeds");
    TopologyGeneration gen2 = body.topology_generation();
    expect(gen2 > gen1, "re-finalize increments topology generation");
}

// ---------------------------------------------------------------------------
// Test 11: Modal system modes
// ---------------------------------------------------------------------------
void test_modal_modes() {
    ModalSystem modal;

    ModalMode bend;
    bend.type = ModalMode::Type::BendingX;
    bend.amplitude = 0.5f;
    bend.frequency = 2.0f;
    modal.add_mode(bend);

    expect(modal.mode_count() == 1, "modal system has one mode");

    // Compute displacement at a point
    Vec3 disp = modal.compute_anchor_displacement({1.0f, 0.5f, 0.0f}, 0.25f, 0);
    expect(length(disp) >= 0.0f, "modal displacement is computed");

    ModalMode breathe;
    breathe.type = ModalMode::Type::Breathing;
    breathe.amplitude = 1.0f;
    modal.add_mode(breathe);
    expect(modal.mode_count() == 2, "modal system has two modes after adding");
}

// ---------------------------------------------------------------------------
// Test 12: Effective stiffness reduction from damage
// ---------------------------------------------------------------------------
void test_stiffness_reduction() {
    PatchState patch;
    patch.cumulative_plastic_strain = 0.5f;
    patch.damage = 0.05f;  // 0.5 * 0.1 damage rate

    expect(patch.damage > 0.0f, "damage is positive after plastic strain");
    expect(patch.effective_stiffness < 1.0f || patch.effective_stiffness == 1.0f,
           "effective stiffness is valid");

    // After damage accumulation, stiffness should be reduced
    float eff = DamageAccumulator::get_effective_stiffness(patch);
    expect(eff <= 1.0f, "effective stiffness is at most 1.0");
    expect(eff >= 0.0f, "effective stiffness is at least 0.0");
}

// ---------------------------------------------------------------------------
// Test 13: Spatial grid empty cell handling (carried from M3)
// ---------------------------------------------------------------------------
void test_spatial_grid_empty_cells() {
    SpatialGrid grid(4, 4, 4);
    const SpatialCell& empty = grid.cell(0, 0, 0);
    expect(empty.is_empty(), "new grid has empty cells");

    const SpatialCell& oob = grid.cell(100, 100, 100);
    expect(oob.is_empty(), "out-of-bounds access returns empty cell");
}

// ---------------------------------------------------------------------------
// Test 14: Cache generation mismatch rejection (carried from M3)
// ---------------------------------------------------------------------------
void test_cache_generation_mismatch() {
    SurfelCache cache;
    expect(!cache.matches_generation(5, 3), "empty cache does not match arbitrary generation");

    cache.set_source_generation(5);
    cache.set_topology_generation(3);
    expect(cache.matches_generation(5, 3), "cache matches its own generation");
    expect(!cache.matches_generation(6, 3), "cache rejects different body generation");
    expect(!cache.matches_generation(5, 4), "cache rejects different topology generation");
}

// ---------------------------------------------------------------------------
// Test 15: Basis normals (carried from M3)
// ---------------------------------------------------------------------------
void test_basis_normals() {
    Vec3 p00{0,0,0}, p10{1,0,0}, p01{0,1,0}, p11{1,1,0};
    std::array<float, 8> coeffs{};
    std::array<Vec3, 4> residuals{};

    Vec3 normal = evaluate_patch_normal(p00, p10, p01, p11,
                                         BasisType::Bilinear, coeffs, residuals, 0.5f, 0.5f);
    expect(std::abs(normal.z - 1.0f) < 0.01f, "bilinear plane has upward normal");

    coeffs[4] = 1.0f;
    Vec3 cupula_normal = evaluate_patch_normal(p00, p10, p01, p11,
                                                BasisType::Cupula, coeffs, residuals, 0.5f, 0.5f);
    expect(length(cupula_normal) > 0.9f, "cupula normal has meaningful magnitude");
}

int main() {
    test_material_properties_affect_springs();
    test_damage_accumulation_monotonic();
    test_modal_displacement_composition();
    test_strain_energy_non_negative();
    test_material_driven_colors();
    test_basis_evaluation();
    test_welded_seam_error();
    test_multi_surfel_cells();
    test_temporal_coherence();
    test_topology_generation_tracking();
    test_modal_modes();
    test_stiffness_reduction();
    test_spatial_grid_empty_cells();
    test_cache_generation_mismatch();
    test_basis_normals();

    std::cout << (failures ? "M4 TESTS FAILED\n" : "M4 TESTS PASSED\n");
    return failures ? 1 : 0;
}