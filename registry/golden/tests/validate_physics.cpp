// =============================================================================
// validate_physics.cpp - Milestone 8 gate. Full physics loop:
//  A) VOLUME: analytic divergence-theorem volume of the encoded torus
//     vs closed form 2*pi^2*R*r^2 (R=1, r=0.35) within 2% (the encode
//     itself carries 0.167% RMS, so this also cross-checks derivatives).
//  B) DROP TEST: torus falls onto plane z=0 with restitution 0.4:
//     B1 penetration at every step end < 0.025 world units
//     B2 bounce apexes strictly decrease; object settles (|v| < 0.05,
//        height stable) within 6 simulated seconds
//     B3 the first impact injects modal energy (deformation on impact)
//  C) REST HEIGHT: settled height ~= 0.35 (tube radius) within 0.02.
// =============================================================================
#include "delirium/core/GameObject.h"
#include "delirium/physics/PhysicsWorld.h"
#include <cstdio>
#include <cmath>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* psf = argc > 1 ? argv[1] : "assets/torus.psf";
    const char* psm = argc > 2 ? argv[2] : "assets/torus.modes";
    int fails = 0;

    FixedReconstructor maker;
    ReconstructionRequest req;
    req.semantic = std::string("file:") + psf;
    GameObject torus("torus", maker, req);
    if (!&torus.Body()) { std::fprintf(stderr, "load failed\n"); return 1; }
    const bool has_modes = torus.LoadModes(psm);
    std::printf("modes loaded: %s (%d modes)\n",
                has_modes ? "yes" : "NO", torus.Dynamics().ModeCount());

    // A) volume
    const float V = torus.Body().Volume();
    const float Va = 2.0f * 9.8696044f * 1.0f * 0.35f * 0.35f; // 2 pi^2 R r^2
    const float verr = std::fabs(V - Va) / Va;
    std::printf("A) volume: analytic %.4f vs closed-form %.4f (err %.2f%%) %s\n",
                V, Va, verr*100, verr < 0.02f ? "OK" : "FAIL");
    fails += !(verr < 0.02f);

    // B) drop test
    PhysicsWorld world;
    world.AddPlane({0,0,1}, 0.0f);
    world.SetGravity({0,0,-9.81f});
    world.Register(&torus);
    torus.GetTransform().position = {0, 0, 1.2f};
    torus.Rigid().restitution = 0.4f;
    torus.Rigid().mass = 1; torus.Rigid().inv_mass = 1;
    torus.Rigid().inertia = 0.4f; torus.Rigid().inv_inertia = 2.5f;

    const float dt = 1.0f / 240.0f;
    const float rest_h = 0.35f;              // tube radius = rest height
    const float fly_th = rest_h + 0.02f;     // "airborne" above this
    float max_pen = 0, apex1 = -1, apex2 = -1, cur_apex = -1;
    bool airborne = false, modal_kicked = false;
    float settle_t = -1;
    for (float t = 0; t < 6.0f; t += dt) {
        world.Step(dt);
        torus.FixedUpdate(dt);
        std::vector<Contact> cs;
        CollisionAnalytic::BodyVsPlane(torus.Body(), torus.GetTransform(),
                                       {0,0,1}, 0, cs);
        for (const Contact& c : cs) max_pen = std::fmax(max_pen, c.depth);
        if (has_modes && torus.Dynamics().TotalEnergy() > 1e-9f)
            modal_kicked = true;
        // bounce = a flight arc that clears fly_th, then comes back down.
        // record the PEAK of each distinct arc (height hysteresis, not
        // velocity sign -> immune to modal micro-oscillation).
        const float z = torus.GetTransform().position.z;
        if (z > fly_th) { airborne = true; cur_apex = std::fmax(cur_apex, z); }
        else if (airborne) {                 // landed: close this arc
            if (apex1 < 0)      apex1 = cur_apex;
            else if (apex2 < 0) apex2 = cur_apex;
            airborne = false; cur_apex = -1;
        }
        const float vz = torus.Rigid().velocity.z;
        const float sp = std::sqrt(torus.Rigid().velocity.x*torus.Rigid().velocity.x
                       + torus.Rigid().velocity.y*torus.Rigid().velocity.y + vz*vz);
        if (settle_t < 0 && t > 0.8f && sp < 0.05f) settle_t = t;
        if (sp >= 0.05f) settle_t = -1;
    }
    std::printf("B1) penetration: max %.4f (tol 0.025) %s\n",
                max_pen, max_pen < 0.025f ? "OK" : "FAIL");
    fails += !(max_pen < 0.025f);
    const bool apex_ok = apex1 > 0 && (apex2 < 0 || apex2 < apex1)
                      && apex1 < 1.2f;
    std::printf("B2) bounces: apex1 %.3f apex2 %.3f, settled at t=%.2fs %s\n",
                apex1, apex2, settle_t,
                (apex_ok && settle_t > 0) ? "OK" : "FAIL");
    fails += !(apex_ok && settle_t > 0);
    std::printf("B3) impact->modal energy: %s %s\n",
                modal_kicked ? "injected" : "none",
                (has_modes && modal_kicked) ? "OK" : "FAIL");
    fails += !(has_modes && modal_kicked);   // STRICT: modes must load AND fire

    const float zrest = torus.GetTransform().position.z;
    std::printf("C) rest height: %.4f (expect ~0.35) %s\n", zrest,
                std::fabs(zrest - 0.35f) < 0.02f ? "OK" : "FAIL");
    fails += !(std::fabs(zrest - 0.35f) < 0.02f);

    std::printf(fails ? "PHYSICS VALIDATION FAILED (%d)\n"
                      : "PHYSICS VALIDATION PASSED\n", fails);
    return fails;
}
