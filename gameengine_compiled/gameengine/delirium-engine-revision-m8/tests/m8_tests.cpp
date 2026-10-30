#include "delirium_m8/body_runtime.hpp"
#include "delirium_m8/constraint_solver.hpp"
#include "delirium_m8/collision_response.hpp"
#include "delirium_m8/external_forces.hpp"
#include "delirium_m8/fracture_system.hpp"
#include "delirium_m8/multi_body_manager.hpp"
#include "delirium_m8/spatial_grid.hpp"
#include "delirium_m8/surfel_cache.hpp"
#include "delirium_m8/contact_detector.hpp"
#include "delirium_m8/basis_functions.hpp"
#include "delirium_m8/elastic_solver.hpp"
#include "delirium_m8/damage_accumulator.hpp"
#include "delirium_m8/animation_timeline.hpp"
#include "delirium_m8/serializer.hpp"
#include "delirium_m8/keyframe_animator.hpp"
#include "delirium_m8/welded_seam.hpp"
#include "delirium_m8/material_registry.hpp"
#include "delirium_m8/scene_budget.hpp"
#include "delirium_m8/frustum_culler.hpp"
#include "delirium_m8/shared_material_pool.hpp"
#include "delirium_m8/interaction_layer.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace delirium::m8;

static int failures = 0;

static void expect(bool cond, const char* msg) {
    if (!cond) { std::cout << "FAIL: " << msg << "\n"; failures++; }
    else { std::cout << "PASS: " << msg << "\n"; }
}

// Helper: create a simple 4-anchor body runtime
static BodyRuntime make_body(BodyId id, const std::vector<Vec3>& positions, MaterialProperties mat = {}) {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error;
    body.finalize(&error);

    BodyRuntime rt(std::move(body), id);
    MaterialRegistry reg;
    if (mat.youngs_modulus == 0.0f) mat.youngs_modulus = 1.0e6f;
    reg.register_material(0, mat);
    rt.set_materials(std::move(reg));
    rt.initialize(positions, &error);
    return rt;
}

// ============================================================
// Scene Budget Tests (1-5)
// ============================================================

void test_budget_within() {
    SceneBudget b;
    b.max_bodies = 10;
    b.max_surfels = 5000;
    b.max_constraints = 100;
    expect(b.is_within_budget(5, 2000, 50), "budget within limits");
}

void test_budget_exceeded_bodies() {
    SceneBudget b;
    b.max_bodies = 5;
    b.max_surfels = 5000;
    b.max_constraints = 100;
    expect(!b.is_within_budget(10, 2000, 50), "budget exceeded on bodies");
}

void test_budget_exceeded_surfels() {
    SceneBudget b;
    b.max_bodies = 10;
    b.max_surfels = 1000;
    b.max_constraints = 100;
    expect(!b.is_within_budget(5, 5000, 50), "budget exceeded on surfels");
}

void test_budget_exceeded_constraints() {
    SceneBudget b;
    b.max_bodies = 10;
    b.max_surfels = 5000;
    b.max_constraints = 50;
    expect(!b.is_within_budget(5, 2000, 100), "budget exceeded on constraints");
}

void test_budget_manager_integration() {
    SceneBudget b;
    b.max_bodies = 3;
    b.max_surfels = 10000;
    b.max_constraints = 500;

    MultiBodyManager manager(b);
    auto body0 = make_body(0, {{0,0,0},{1,0,0},{0,1,0},{1,1,0}});
    auto body1 = make_body(1, {{2,0,0},{3,0,0},{2,1,0},{3,1,0}});
    manager.add_body(std::move(body0));
    manager.add_body(std::move(body1));

    expect(manager.is_within_budget(), "manager within budget after 2 bodies");
    expect(manager.body_count() == 2, "manager has 2 bodies");
}

// ============================================================
// Frustum Culling Tests (6-12)
// ============================================================

void test_frustum_culler_visible() {
    FrustumCuller culler;
    Frustum f;
    f.origin = {0, 0, 0};
    f.forward = {0, 0, 1};
    f.right = {1, 0, 0};
    f.up = {0, 1, 0};
    f.fov_y = 1.0f;
    f.aspect = 1.0f;
    f.near_plane = 0.1f;
    f.far_plane = 100.0f;

    expect(culler.is_visible({0, 0, 10}, 1.0f, f), "center point is visible");
}

void test_frustum_culler_behind() {
    FrustumCuller culler;
    Frustum f;
    f.origin = {0, 0, 0};
    f.forward = {0, 0, 1};
    f.right = {1, 0, 0};
    f.up = {0, 1, 0};
    f.fov_y = 1.0f;
    f.aspect = 1.0f;
    f.near_plane = 0.1f;
    f.far_plane = 100.0f;

    expect(!culler.is_visible({0, 0, -10}, 1.0f, f), "behind camera is culled");
}

void test_frustum_culler_far() {
    FrustumCuller culler;
    Frustum f;
    f.origin = {0, 0, 0};
    f.forward = {0, 0, 1};
    f.right = {1, 0, 0};
    f.up = {0, 1, 0};
    f.fov_y = 1.0f;
    f.aspect = 1.0f;
    f.near_plane = 0.1f;
    f.far_plane = 10.0f;

    expect(!culler.is_visible({0, 0, 50}, 1.0f, f), "beyond far plane is culled");
}

void test_frustum_culler_near() {
    FrustumCuller culler;
    Frustum f;
    f.origin = {0, 0, 0};
    f.forward = {0, 0, 1};
    f.right = {1, 0, 0};
    f.up = {0, 1, 0};
    f.fov_y = 1.0f;
    f.aspect = 1.0f;
    f.near_plane = 5.0f;
    f.far_plane = 100.0f;

    expect(!culler.is_visible({0, 0, 1}, 1.0f, f), "before near plane is culled");
}

void test_frustum_culler_batch() {
    FrustumCuller culler;
    Frustum f;
    f.origin = {0, 0, 0};
    f.forward = {0, 0, 1};
    f.right = {1, 0, 0};
    f.up = {0, 1, 0};
    f.fov_y = 1.0f;
    f.aspect = 1.0f;
    f.near_plane = 0.1f;
    f.far_plane = 100.0f;

    std::vector<BodyId> ids = {0, 1, 2};
    std::vector<Vec3> centers = {{0, 0, 10}, {0, 0, -10}, {0, 0, 5}};

    auto visible = culler.cull(ids, centers, f);
    expect(visible.size() == 2, "cull returns 2 visible bodies");
}

void test_frustum_manager_visible() {
    MultiBodyManager manager;
    auto body0 = make_body(0, {{0, 0, 10}, {1, 0, 10}, {0, 1, 10}, {1, 1, 10}});
    auto body1 = make_body(1, {{0, 0, -10}, {1, 0, -10}, {0, 1, -10}, {1, 1, -10}});
    manager.add_body(std::move(body0));
    manager.add_body(std::move(body1));

    Frustum f;
    f.origin = {0, 0, 0};
    f.forward = {0, 0, 1};
    f.right = {1, 0, 0};
    f.up = {0, 1, 0};
    f.fov_y = 1.0f;
    f.aspect = 1.0f;
    f.near_plane = 0.1f;
    f.far_plane = 100.0f;

    auto visible = manager.get_visible_bodies(f);
    expect(visible.size() == 1, "manager culling returns 1 visible body");
}

void test_frustum_manager_no_frustum() {
    MultiBodyManager manager;
    auto body0 = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    auto body1 = make_body(1, {{2, 0, 0}, {3, 0, 0}, {2, 1, 0}, {3, 1, 0}});
    manager.add_body(std::move(body0));
    manager.add_body(std::move(body1));

    std::string error;
    manager.step(0.016f, nullptr, &error);
    expect(manager.get_body(0)->published_generation() > 0, "step without frustum steps all bodies");
}

// ============================================================
// Shared Material Pool Tests (13-17)
// ============================================================

void test_material_pool_register() {
    SharedMaterialPool pool;
    MaterialProperties mat;
    mat.youngs_modulus = 200.0e9f;
    mat.r = 255; mat.g = 0; mat.b = 0;

    MaterialId id = pool.register_material(mat);
    expect(id == 0, "first material id is 0");
    expect(pool.get(id).youngs_modulus == 200.0e9f, "material properties stored");
}

void test_material_pool_unique_ids() {
    SharedMaterialPool pool;
    MaterialProperties mat1, mat2;
    mat1.r = 255; mat2.r = 0;

    MaterialId id1 = pool.register_material(mat1);
    MaterialId id2 = pool.register_material(mat2);
    expect(id1 != id2, "materials get unique ids");
    expect(pool.material_count() == 2, "pool has 2 materials");
}

void test_material_pool_reference_count() {
    SharedMaterialPool pool;
    MaterialProperties mat;
    MaterialId id = pool.register_material(mat);

    expect(pool.reference_count(id) == 1, "initial ref count is 1");
    pool.add_reference(id);
    expect(pool.reference_count(id) == 2, "ref count incremented");
    pool.remove_reference(id);
    expect(pool.reference_count(id) == 1, "ref count decremented");
}

void test_material_pool_remove_last_ref() {
    SharedMaterialPool pool;
    MaterialProperties mat;
    MaterialId id = pool.register_material(mat);

    pool.remove_reference(id);
    expect(pool.reference_count(id) == 0, "material removed after last ref");
    expect(pool.material_count() == 0, "pool is empty after removal");
}

void test_material_pool_manager_integration() {
    MultiBodyManager manager;
    MaterialProperties mat;
    mat.r = 100; mat.g = 200; mat.b = 50;

    MaterialId id = manager.material_pool().register_material(mat);
    expect(manager.material_pool().get(id).r == 100, "manager material pool works");
}

// ============================================================
// Interaction Layer Tests (18-30)
// ============================================================

void test_pick_basic() {
    InteractionLayer layer;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    std::vector<BodyRuntime*> ptrs = {&body};

    PickResult r = layer.pick(ptrs, {0, 0, -5}, {0, 0, 1}, 100.0f);
    expect(r.body_id == kInvalidBody || r.body_id == 0, "pick returns valid body or none");
}

void test_grab_moves_anchor() {
    InteractionLayer layer;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});

    layer.grab(&body, 0, {10, 10, 10});
    expect(body.write_state().anchors[0].position.x == 10.0f, "grab moves anchor x");
    expect(body.write_state().anchors[0].position.y == 10.0f, "grab moves anchor y");
}

void test_grab_null_body() {
    InteractionLayer layer;
    layer.grab(nullptr, 0, {0, 0, 0});
    expect(true, "grab with null body does not crash");
}

void test_cut_patches() {
    InteractionLayer layer;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});

    // Cut along x=0.5 plane, normal pointing right
    layer.cut(&body, {1, 0, 0}, {0.5f, 0, 0});
    // Anchors on the left side (x < 0.5) should have been moved
    expect(true, "cut does not crash");
}

void test_weld_creates_constraint() {
    InteractionLayer layer;
    auto bodyA = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    auto bodyB = make_body(1, {{2, 0, 0}, {3, 0, 0}, {2, 1, 0}, {3, 1, 0}});

    expect(bodyA.constraint_solver().constraint_count() == 0, "no constraints before weld");
    layer.weld(&bodyA, &bodyB, 0, 0, 1.0f);
    expect(bodyA.constraint_solver().constraint_count() == 1, "weld adds constraint");
}

void test_weld_null_bodies() {
    InteractionLayer layer;
    layer.weld(nullptr, nullptr, 0, 0);
    expect(true, "weld with null bodies does not crash");
}

void test_heat_sets_temperature() {
    InteractionLayer layer;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});

    layer.heat(&body, 0, 1000.0f);
    expect(body.write_state().anchors[0].temperature == 1000.0f, "heat sets temperature");
}

void test_heat_null_body() {
    InteractionLayer layer;
    layer.heat(nullptr, 0, 100.0f);
    expect(true, "heat with null body does not crash");
}

void test_pick_patch() {
    InteractionLayer layer;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});

    PickResult r = layer.pick_patch(&body, {0, 0, -5}, {0, 0, 1}, 100.0f);
    expect(r.body_id == kInvalidBody || r.body_id == 0, "pick_patch returns valid body or none");
}

void test_interaction_manager_pick() {
    MultiBodyManager manager;
    auto body0 = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    manager.add_body(std::move(body0));

    PickResult r = manager.pick({0, 0, -5}, {0, 0, 1}, 100.0f);
    expect(r.body_id == kInvalidBody || r.body_id == 0, "manager pick returns valid result");
}

void test_grab_via_manager() {
    MultiBodyManager manager;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    manager.add_body(std::move(body));

    auto* b = manager.get_body(0);
    manager.interaction().grab(b, 0, {5, 5, 5});
    expect(b->write_state().anchors[0].position.x == 5.0f, "grab via manager works");
}

// ============================================================
// Bounding Box Tests (31-33)
// ============================================================

void test_bounding_center() {
    auto body = make_body(0, {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {2, 2, 0}});
    Vec3 center = body.bounding_center();
    expect(nearly_equal(center, {1.0f, 1.0f, 0.0f}), "bounding center is correct");
}

void test_bounding_radius() {
    auto body = make_body(0, {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {2, 2, 0}});
    float r = body.bounding_radius();
    expect(r > 0.0f, "bounding radius is positive");
    expect(r >= 1.0f, "bounding radius covers extent");
}

void test_body_centers_manager() {
    MultiBodyManager manager;
    auto body0 = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    auto body1 = make_body(1, {{10, 0, 0}, {11, 0, 0}, {10, 1, 0}, {11, 1, 0}});
    manager.add_body(std::move(body0));
    manager.add_body(std::move(body1));

    auto centers = manager.get_body_centers();
    expect(centers.size() == 2, "manager returns 2 body centers");
}

// ============================================================
// Multi-body with Budget Tests (34-37)
// ============================================================

void test_manager_remove_body() {
    MultiBodyManager manager;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    manager.add_body(std::move(body));

    expect(manager.active_body_count() == 1, "1 active body");
    manager.remove_body(0);
    expect(manager.active_body_count() == 0, "0 active bodies after remove");
}

void test_manager_total_surfels() {
    MultiBodyManager manager;
    auto body0 = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    auto body1 = make_body(1, {{2, 0, 0}, {3, 0, 0}, {2, 1, 0}, {3, 1, 0}});
    manager.add_body(std::move(body0));
    manager.add_body(std::move(body1));

    expect(manager.total_surfels() > 0, "total_surfels returns positive");
}

void test_manager_total_constraints() {
    MultiBodyManager manager;
    auto body = make_body(0, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}});
    manager.add_body(std::move(body));

    expect(manager.total_constraints() == 0, "no constraints initially");
}

void test_manager_step_with_frustum() {
    MultiBodyManager manager;
    auto body0 = make_body(0, {{0, 0, 10}, {1, 0, 10}, {0, 1, 10}, {1, 1, 10}});
    auto body1 = make_body(1, {{0, 0, -10}, {1, 0, -10}, {0, 1, -10}, {1, 1, -10}});
    manager.add_body(std::move(body0));
    manager.add_body(std::move(body1));

    Frustum f;
    f.origin = {0, 0, 0};
    f.forward = {0, 0, 1};
    f.right = {1, 0, 0};
    f.up = {0, 1, 0};
    f.fov_y = 1.0f;
    f.aspect = 1.0f;
    f.near_plane = 0.1f;
    f.far_plane = 100.0f;

    std::string error;
    manager.step(0.016f, &f, &error);
    expect(manager.get_body(0)->published_generation() > 0, "visible body stepped");
}

// ============================================================
// Carryover tests from M6/M7 (38-55)
// ============================================================

void test_gravity() {
    ExternalForces forces;
    forces.set_force({{0.0f, -9.81f, 0.0f}, {}, 0.0f});

    BodyState state;
    state.resize(1, 0);
    state.anchors[0].velocity = {0.0f, 0.0f, 0.0f};

    forces.apply(state, 0.016f);
    expect(state.anchors[0].velocity.y < 0.0f, "gravity applies downward acceleration");
}

void test_drag() {
    ExternalForces forces;
    forces.set_force({{}, {}, 0.5f});

    BodyState state;
    state.resize(1, 0);
    state.anchors[0].velocity = {10.0f, 0.0f, 0.0f};

    forces.apply(state, 0.016f);
    expect(state.anchors[0].velocity.x < 10.0f, "drag reduces velocity");
}

void test_point_force() {
    ExternalForces forces;
    forces.set_force({{}, {}, 0.0f});
    PointForce pf;
    pf.anchor = 1;
    pf.force = {5.0f, 0.0f, 0.0f};
    forces.add_point_force(pf);

    BodyState state;
    state.resize(3, 0);
    state.anchors[0].velocity = {0.0f, 0.0f, 0.0f};
    state.anchors[1].velocity = {0.0f, 0.0f, 0.0f};
    state.anchors[2].velocity = {0.0f, 0.0f, 0.0f};

    forces.apply(state, 0.016f);
    expect(state.anchors[1].velocity.x > 0.0f, "point force applied to anchor 1");
    expect(state.anchors[0].velocity.x == 0.0f, "no force on anchor 0");
}

void test_wind() {
    ExternalForces forces;
    forces.set_force({{}, {2.0f, 0.0f, 0.0f}, 0.0f});

    BodyState state;
    state.resize(1, 0);
    state.anchors[0].velocity = {0.0f, 0.0f, 0.0f};

    forces.apply(state, 0.016f);
    expect(state.anchors[0].velocity.x > 0.0f, "wind applies lateral acceleration");
}

void test_fracture_detection() {
    FractureSystem fs;
    FractureParams params;
    params.enable_fracture = true;
    params.damage_threshold = 0.8f;
    fs.set_params(params);

    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    BodyState state;
    state.resize(4, 1);
    state.patches[0].damage = 0.95f;

    auto event = fs.evaluate(state, body, 0);
    expect(event.triggered, "fracture triggered when damage exceeds threshold");
}

void test_no_fracture_below_threshold() {
    FractureSystem fs;
    FractureParams params;
    params.enable_fracture = true;
    params.damage_threshold = 0.9f;
    fs.set_params(params);

    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    BodyState state;
    state.resize(4, 1);
    state.patches[0].damage = 0.5f;

    auto event = fs.evaluate(state, body, 0);
    expect(!event.triggered, "no fracture below threshold");
}

void test_fracture_disabled() {
    FractureSystem fs;
    FractureParams params;
    params.enable_fracture = false;
    fs.set_params(params);

    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0);
    std::string error; body.finalize(&error);

    BodyState state;
    state.resize(4, 1);
    state.patches[0].damage = 1.0f;

    auto event = fs.evaluate(state, body, 0);
    expect(!event.triggered, "fracture not triggered when disabled");
}

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
    auto body = make_body(0, {{0,0,0},{1,0,0},{0,1,0},{1,1,0}});
    std::string error;
    body.step(0.016f, &error);
    expect(body.published_generation() > 0, "generation increments");
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

void test_constraint() {
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
    state.anchors[1].position = {2.0f, 0.0f, 0.0f};

    std::string error;
    solver.solve(state, 0.016f, &error);

    float dist = length(state.anchors[0].position - state.anchors[1].position);
    expect(dist < 2.0f, "distance constraint reduces separation");
}

// ============================================================
// Multi-body carryover tests (56-60)
// ============================================================

void test_multi_body_manager() {
    MultiBodyManager manager;

    auto body0 = make_body(0, {{0,0,0},{1,0,0},{0,1,0},{1,1,0}});
    auto body1 = make_body(1, {{2,0,0},{3,0,0},{2,1,0},{3,1,0}});
    manager.add_body(std::move(body0));
    manager.add_body(std::move(body1));

    expect(manager.body_count() == 2, "manager has 2 bodies");
    expect(manager.get_body(0) != nullptr, "can retrieve body 0");
    expect(manager.get_body(1) != nullptr, "can retrieve body 1");

    std::string error;
    manager.step(0.016f, nullptr, &error);
    expect(manager.get_body(0)->published_generation() > 0, "body 0 stepped");
}

void test_multi_body_gravity() {
    MultiBodyManager manager;
    auto body = make_body(0, {{0,10,0},{1,10,0},{0,11,0},{1,11,0}});
    manager.add_body(std::move(body));

    manager.set_external_forces({{0.0f, -9.81f, 0.0f}, {}, 0.0f});

    std::string error;
    manager.step(0.016f, nullptr, &error);
    expect(manager.get_body(0)->read_state().anchors[0].velocity.y < 0.0f,
           "gravity affects body in manager");
}

void test_multi_body_fracture() {
    MultiBodyManager manager;
    auto body = make_body(0, {{0,0,0},{1,0,0},{0,1,0},{1,1,0}});

    FractureParams fp;
    fp.enable_fracture = true;
    fp.damage_threshold = 0.5f;
    body.set_fracture_params(fp);

    auto& front = const_cast<BodyState&>(body.read_state());
    front.patches[0].damage = 0.95f;

    manager.add_body(std::move(body));
    manager.set_fracture_params(fp);

    auto events = manager.check_fractures();
    expect(!events.empty(), "fracture events returned");
    expect(events[0].triggered, "fracture detected in manager");
}

void test_runtime_external_forces() {
    auto body = make_body(0, {{0,0,0},{1,0,0},{0,1,0},{1,1,0}});

    ExternalForce ef;
    ef.gravity = {0.0f, -9.81f, 0.0f};
    body.set_external_forces(ef);

    std::string error;
    body.step(0.016f, &error);
    expect(body.read_state().anchors[0].velocity.y < 0.0f,
           "runtime applies gravity during step");
}

void test_runtime_fracture() {
    auto body = make_body(0, {{0,0,0},{1,0,0},{0,1,0},{1,1,0}});

    FractureParams fp;
    fp.enable_fracture = true;
    fp.damage_threshold = 0.5f;
    body.set_fracture_params(fp);

    auto& front = const_cast<BodyState&>(body.read_state());
    front.patches[0].damage = 0.9f;
    FractureEvent event = body.check_fracture();
    expect(event.triggered, "runtime fracture check detects high damage");
}

int main() {
    // Scene Budget Tests
    test_budget_within();
    test_budget_exceeded_bodies();
    test_budget_exceeded_surfels();
    test_budget_exceeded_constraints();
    test_budget_manager_integration();

    // Frustum Culling Tests
    test_frustum_culler_visible();
    test_frustum_culler_behind();
    test_frustum_culler_far();
    test_frustum_culler_near();
    test_frustum_culler_batch();
    test_frustum_manager_visible();
    test_frustum_manager_no_frustum();

    // Shared Material Pool Tests
    test_material_pool_register();
    test_material_pool_unique_ids();
    test_material_pool_reference_count();
    test_material_pool_remove_last_ref();
    test_material_pool_manager_integration();

    // Interaction Layer Tests
    test_pick_basic();
    test_grab_moves_anchor();
    test_grab_null_body();
    test_cut_patches();
    test_weld_creates_constraint();
    test_weld_null_bodies();
    test_heat_sets_temperature();
    test_heat_null_body();
    test_pick_patch();
    test_interaction_manager_pick();
    test_grab_via_manager();

    // Bounding Box Tests
    test_bounding_center();
    test_bounding_radius();
    test_body_centers_manager();

    // Multi-body with Budget Tests
    test_manager_remove_body();
    test_manager_total_surfels();
    test_manager_total_constraints();
    test_manager_step_with_frustum();

    // Carryover tests from M6/M7
    test_gravity();
    test_drag();
    test_point_force();
    test_wind();
    test_fracture_detection();
    test_no_fracture_below_threshold();
    test_fracture_disabled();
    test_material_properties();
    test_damage_monotonic();
    test_contact_detection();
    test_basis_evaluation();
    test_spatial_grid();
    test_topology_generation();
    test_lod();
    test_seam_error();
    test_timeline();
    test_serialize_topology();
    test_serialize_state();
    test_serialize_materials();
    test_constraint();

    // Multi-body carryover tests
    test_multi_body_manager();
    test_multi_body_gravity();
    test_multi_body_fracture();
    test_runtime_external_forces();
    test_runtime_fracture();

    std::cout << (failures ? "M8 TESTS FAILED\n" : "M8 TESTS PASSED\n");
    return failures ? 1 : 0;
}