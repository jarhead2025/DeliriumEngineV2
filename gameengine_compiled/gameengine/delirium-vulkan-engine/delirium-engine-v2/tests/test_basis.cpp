#include "core/types.h"
#include "core/body_topology.h"
#include "core/body_state.h"
#include "core/basis_functions.h"
#include "core/body_runtime.h"
#include "core/material_registry.h"
#include "core/surfel_cache.h"
#include "core/modal_system.h"
#include "core/scene.h"
#include "core/engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

using namespace delirium;

/* ── Simple test harness ─────────────────────────────────────────── */
static int passed = 0, failed = 0;
#define expect(cond) do { \
    if (cond) { ++passed; } \
    else { ++failed; fprintf(stderr, "FAIL: %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)

/* ── 64-byte record layout ──────────────────────────────────────── */
static void test_record_sizes() {
    expect(sizeof(AnchorRecord) == 64);
    expect(sizeof(SurfaceRecord) == 64);
}

/* ── Vec3 operators ──────────────────────────────────────────────── */
static void test_vec3() {
    Vec3 a{1, 2, 3};
    Vec3 b{4, 5, 6};

    Vec3 c = a + b;
    expect(c.x == 5 && c.y == 7 && c.z == 9);

    Vec3 d = a - b;
    expect(d.x == -3 && d.y == -3 && d.z == -3);

    Vec3 e = a * 2.0f;
    expect(e.x == 2 && e.y == 4 && e.z == 6);

    Vec3 f = a / 2.0f;
    expect(f.x == 0.5f && f.y == 1.0f && f.z == 1.5f);

    float d2 = dot(a, b);
    expect(d2 == 32.0f);

    Vec3 cr = cross(a, b);
    expect(cr.x == -3 && cr.y == 6 && cr.z == -3);

    float l = length(a);
    expect(std::abs(l - std::sqrt(14.0f)) < 1e-6f);

    Vec3 n = normalize_vec3(a);
    expect(std::abs(length(n) - 1.0f) < 1e-6f);
}

/* ── Corner-vanishing basis ──────────────────────────────────────── */
static void test_deform_basis() {
    float b[8];

    // At corners, all terms must be zero
    BasisEvaluator::deform_basis(0.0f, 0.0f, b);
    for (int i = 0; i < 8; ++i) expect(std::abs(b[i]) < 1e-7f);

    BasisEvaluator::deform_basis(1.0f, 0.0f, b);
    for (int i = 0; i < 8; ++i) expect(std::abs(b[i]) < 1e-7f);

    BasisEvaluator::deform_basis(0.0f, 1.0f, b);
    for (int i = 0; i < 8; ++i) expect(std::abs(b[i]) < 1e-7f);

    BasisEvaluator::deform_basis(1.0f, 1.0f, b);
    for (int i = 0; i < 8; ++i) expect(std::abs(b[i]) < 1e-7f);

    // At center, terms should be non-zero
    BasisEvaluator::deform_basis(0.5f, 0.5f, b);
    expect(std::abs(b[0] - 0.125f) < 1e-6f); // u(1-u)(1-v) = 0.5*0.5*0.5 = 0.125
    expect(std::abs(b[1] - 0.125f) < 1e-6f);
    expect(std::abs(b[2] - 0.125f) < 1e-6f);
    expect(std::abs(b[3] - 0.125f) < 1e-6f);
    expect(std::abs(b[4]) < 1e-7f);  // zero at center (2u-1 = 0)
    expect(std::abs(b[5]) < 1e-7f);
    expect(std::abs(b[6]) < 1e-7f);
    expect(std::abs(b[7]) < 1e-7f);
}

/* ── BodyTopology ────────────────────────────────────────────────── */
static void test_topology() {
    BodyTopology topo;

    AnchorId a0 = topo.add_anchor();
    AnchorId a1 = topo.add_anchor();
    AnchorId a2 = topo.add_anchor();
    AnchorId a3 = topo.add_anchor();
    expect(a0 == 0 && a1 == 1 && a2 == 2 && a3 == 3);

    PatchId p = topo.add_patch({a0, a1, a2, a3}, 0);
    expect(p == 0);
    expect(topo.patch_count() == 1);
    expect(topo.anchor_count() == 4);

    expect(!topo.finalized());
    expect(topo.finalize());
    expect(topo.finalized());
    expect(topo.topology_generation() == 1);

    // Empty topology should fail finalize
    BodyTopology empty;
    expect(!empty.finalize());
}

/* ── DirtySet ────────────────────────────────────────────────────── */
static void test_dirty_set() {
    DirtySet ds(10);
    expect(!ds.marked(0));

    ds.mark(3);
    ds.mark(7);
    expect(ds.marked(3));
    expect(ds.marked(7));
    expect(!ds.marked(5));

    auto idx = ds.indices();
    expect(idx.size() == 2);
    expect(idx[0] == 3 && idx[1] == 7);

    ds.clear();
    expect(!ds.marked(3));
    expect(!ds.marked(7));
}

/* ── MaterialRegistry ────────────────────────────────────────────── */
static void test_materials() {
    MaterialRegistry reg;
    MaterialProperties mat;
    mat.youngs_modulus = 2.0e6f;
    mat.r = 255; mat.g = 128; mat.b = 64;

    reg.register_material(42, mat);
    expect(reg.has(42));
    expect(!reg.has(99));

    const auto& got = reg.get(42);
    expect(got.youngs_modulus == 2.0e6f);
    expect(got.r == 255 && got.g == 128 && got.b == 64);
}

/* ── BodyRuntime: create, init, step ─────────────────────────────── */
static void test_body_runtime() {
    // Build a simple quad
    BodyTopology topo;
    AnchorId a[4];
    for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
    topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
    topo.finalize();

    BodyRuntime body(std::move(topo), 0);

    // Register a material
    MaterialRegistry reg;
    reg.register_material(0, MaterialProperties{});
    body.set_materials(std::move(reg));

    // Initialize with rest positions (unit quad in XY plane)
    std::vector<Vec3> rest = {
        {-0.5f, -0.5f, 0},
        { 0.5f, -0.5f, 0},
        {-0.5f,  0.5f, 0},
        { 0.5f,  0.5f, 0},
    };
    expect(body.initialize(rest));

    // Evaluate surface at center
    SurfacePoint sp = body.evaluate_surface(0, 0.5f, 0.5f);
    expect(std::abs(sp.position.x) < 1e-5f);
    expect(std::abs(sp.position.y) < 1e-5f);
    expect(std::abs(sp.position.z) < 1e-5f);

    // Bounding box
    Vec3 center = body.bounding_center();
    expect(nearly_equal(center, {0, 0, 0}, 1e-5f));
    expect(body.bounding_radius() > 0.5f);
}

/* ── BodyRuntime: physics step with gravity ───────────────────────── */
static void test_body_physics() {
    BodyTopology topo;
    AnchorId a[4];
    for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
    topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
    topo.finalize();

    BodyRuntime body(std::move(topo), 0);
    MaterialRegistry reg;
    reg.register_material(0, MaterialProperties{});
    body.set_materials(std::move(reg));

    std::vector<Vec3> rest = {
        {-0.5f, 1.0f, 0}, {0.5f, 1.0f, 0},
        {-0.5f, 1.0f, 0}, {0.5f, 1.0f, 0},
    };
    body.initialize(rest);

    // Apply gravity
    body.set_external_forces({{0, -9.81f, 0}, {}, 0.0f});

    // Step 10 frames at 1/60s
    for (int i = 0; i < 10; ++i) {
        body.step(1.0f / 60.0f);
    }

    // Y should have decreased (gravity pulls down)
    const auto& state = body.read_state();
    expect(state.anchors[0].position.y < 1.0f);
}

/* ── Modal system ────────────────────────────────────────────────── */
static void test_modal_system() {
    ModalSystem ms;
    ModalMode mode;
    mode.type = ModalMode::Type::Breathing;
    mode.frequency = 1.0f;
    mode.amplitude = 0.1f; // damping
    ms.add_mode(mode);

    std::vector<float> amps = {0.1f};
    std::vector<float> vel = {0.0f};

    ms.integrate(amps, vel, 0.016f);

    // Amplitude should change after integration
    expect(std::abs(amps[0] - 0.1f) > 1e-8f || std::abs(vel[0]) > 1e-8f);
}

/* ── Scene ───────────────────────────────────────────────────────── */
static void test_scene() {
    Scene scene;

    BodyTopology topo;
    AnchorId a[4];
    for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
    topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
    topo.finalize();

    static BodyRuntime body(std::move(topo), 0);
    MaterialRegistry reg;
    reg.register_material(0, MaterialProperties{});
    body.set_materials(std::move(reg));
    body.initialize({{-0.5f,-0.5f,0},{0.5f,-0.5f,0},{-0.5f,0.5f,0},{0.5f,0.5f,0}});

    scene.add_body(&body);
    expect(scene.bodies().size() == 1);
    expect(scene.get_body(0) == &body);

    // Step scene
    expect(scene.step(1.0f / 60.0f));
}

/* ── SurfelCache ─────────────────────────────────────────────────── */
static void test_surfel_cache() {
    BodyTopology topo;
    AnchorId a[4];
    for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
    topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
    topo.finalize();

    BodyRuntime body(std::move(topo), 0);
    MaterialRegistry reg;
    MaterialProperties mat;
    mat.r = 200; mat.g = 100; mat.b = 50;
    reg.register_material(0, mat);
    body.set_materials(std::move(reg));
    body.initialize({{-0.5f,-0.5f,0},{0.5f,-0.5f,0},{-0.5f,0.5f,0},{0.5f,0.5f,0}});

    SurfelCache cache;
    expect(body.rebuild_cache(cache));
    expect(cache.count() > 0);

    // Surfels should have the material colour
    expect(cache.surfels()[0].r == 200);
    expect(cache.surfels()[0].g == 100);
    expect(cache.surfels()[0].b == 50);
}

/* ── 64-byte record round-trip ───────────────────────────────────── */
static void test_record_roundtrip() {
    // Verify AnchorRecord is exactly 64 bytes and packable
    AnchorRecord rec{};
    rec.bone_index = 7;
    rec.pos[0] = 1000;
    rec.pos[1] = 2000;
    rec.pos[2] = 3000;
    rec.normal[0] = 128;
    rec.normal[1] = 255;
    rec.normal[2] = 0;

    // Simulate memcpy round-trip
    std::uint8_t buf[64];
    std::memcpy(buf, &rec, 64);

    AnchorRecord rec2;
    std::memcpy(&rec2, buf, 64);

    expect(rec2.bone_index == 7);
    expect(rec2.pos[0] == 1000);
    expect(rec2.normal[1] == 255);
}

/* ── SurfaceRecord layout ────────────────────────────────────────── */
static void test_surface_record() {
    SurfaceRecord rec{};
    rec.anchor_idx[0] = 0;
    rec.anchor_idx[1] = 1;
    rec.anchor_idx[2] = 2;
    rec.anchor_idx[3] = 3;
    rec.surface_def[0] = static_cast<std::uint16_t>(BasisType::Bilinear);
    rec.coef[0] = 0.5f;
    rec.coef[1] = -0.25f;

    std::uint8_t buf[64];
    std::memcpy(buf, &rec, 64);

    SurfaceRecord rec2;
    std::memcpy(&rec2, buf, 64);

    expect(rec2.anchor_idx[2] == 2);
    expect(std::abs(rec2.coef[0] - 0.5f) < 1e-6f);
    expect(std::abs(rec2.coef[1] - (-0.25f)) < 1e-6f);
}

/* ── Engine ──────────────────────────────────────────────────────── */
static void test_engine() {
    Engine engine;
    engine.set_physics_dt(1.0f / 120.0f);
    engine.set_viewport(640, 480);
    engine.set_camera({0, 0, 5}, {0, 0, 0}, {0, 1, 0});

    expect(engine.viewport_width() == 640);
    expect(engine.viewport_height() == 480);
    expect(engine.physics_dt() == 1.0f / 120.0f);

    // Update with no scene should succeed
    expect(engine.update(1.0f / 60.0f));
}

/* ── Engine: CPU render ──────────────────────────────────────────── */
static void test_cpu_render() {
    Engine engine;
    engine.set_viewport(256, 256);
    engine.set_camera({0, 0, 5}, {0, 0, 0}, {0, 1, 0});

    // Create a body and populate surfel cache
    BodyTopology topo;
    AnchorId a[4];
    for (int i = 0; i < 4; ++i) a[i] = topo.add_anchor();
    topo.add_patch({a[0], a[1], a[2], a[3]}, 0);
    topo.finalize();

    BodyRuntime body(std::move(topo), 0);
    MaterialRegistry reg;
    MaterialProperties mat;
    mat.r = 255; mat.g = 128; mat.b = 64;
    reg.register_material(0, mat);
    body.set_materials(std::move(reg));
    body.initialize({{-0.5f,-0.5f,0},{0.5f,-0.5f,0},{-0.5f,0.5f,0},{0.5f,0.5f,0}});

    SurfelCache cache;
    body.rebuild_cache(cache);

    std::vector<std::uint32_t> pixels;
    engine.render_cpu(cache, pixels);

    expect(pixels.size() == 256 * 256);
    // Should have rendered something (not all black)
    bool has_nonzero = false;
    for (auto p : pixels) {
        if (p != 0xFF000000u) { has_nonzero = true; break; }
    }
    expect(has_nonzero);
}

/* ── Main ────────────────────────────────────────────────────────── */
int main() {
    test_record_sizes();
    test_vec3();
    test_deform_basis();
    test_topology();
    test_dirty_set();
    test_materials();
    test_body_runtime();
    test_body_physics();
    test_modal_system();
    test_scene();
    test_surfel_cache();
    test_record_roundtrip();
    test_surface_record();
    test_engine();
    test_cpu_render();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}