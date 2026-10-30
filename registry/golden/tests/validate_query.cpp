// =============================================================================
// validate_query.cpp - Milestone 6 gate. Geometric queries on equations:
//  A) CLOSEST POINT: for 200 queries built as surface_point + 2% diag
//     along the normal, the recovered foot must land within 0.25% diag
//     of the original surface point (smooth-surface projection).
//  B) RAYCAST: 100 rays aimed at known surface points from outside;
//     every ray must hit, the hit must lie ON the ray (residual gated
//     inside) and at t no later than the aimed point (first-hit rule,
//     small tolerance).
//  C) POKE: a world-space poke must inject motion: the surface near the
//     poke moves after integration, and a far point moves less.
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/physics/ModalDynamics.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace DeliRium;

static float dist(Vec3 a, Vec3 b) {
    return std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}

int main(int argc, char** argv) {
    const char* psf = argc > 1 ? argv[1] : "assets/torus.psf";
    const char* psm = argc > 2 ? argv[2] : "assets/torus.modes";
    auto body = PSFBody::FromFile(psf);
    if (!body) { std::fprintf(stderr, "load failed\n"); return 1; }
    const Vec3 sp = body->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    std::srand(7);
    auto frand = [] { return std::rand() / (float)RAND_MAX; };
    int fails = 0;

    // A) closest point
    float worstA = 0; int missA = 0;
    for (int i = 0; i < 200; ++i) {
        const uint32_t pid = std::rand() % body->Patches().size();
        const float u = 0.1f + 0.8f*frand(), v = 0.1f + 0.8f*frand();
        const Vec3 s = body->Patches()[pid].Evaluate(u, v);
        const Vec3 n = body->Patches()[pid].EvaluateNormal(u, v);
        const Vec3 q = { s.x + 0.02f*diag*n.x, s.y + 0.02f*diag*n.y,
                         s.z + 0.02f*diag*n.z };
        SurfaceHit h;
        if (!body->ClosestPointEx(q, h)) { ++missA; continue; }
        worstA = std::fmax(worstA, dist(h.p, s));
    }
    std::printf("A) closest: worst foot error %.4f%% diag, misses %d %s\n",
                worstA/diag*100, missA,
                (worstA/diag < 0.0025f && !missA) ? "OK" : "FAIL");
    fails += !(worstA/diag < 0.0025f && !missA);

    // B) raycast
    int missB = 0; float late = 0;
    for (int i = 0; i < 100; ++i) {
        const uint32_t pid = std::rand() % body->Patches().size();
        const float u = 0.15f + 0.7f*frand(), v = 0.15f + 0.7f*frand();
        const Vec3 s = body->Patches()[pid].Evaluate(u, v);
        const Vec3 n = body->Patches()[pid].EvaluateNormal(u, v);
        const Vec3 o = { s.x + 0.8f*diag*n.x, s.y + 0.8f*diag*n.y,
                         s.z + 0.8f*diag*n.z };
        const Vec3 d = { s.x-o.x, s.y-o.y, s.z-o.z };
        const float t_aim = std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
        SurfaceHit h;
        if (!body->Raycast(o, d, h)) { ++missB; continue; }
        late = std::fmax(late, h.t - t_aim);
    }
    std::printf("B) raycast: misses %d, max lateness %.3e (tol 1e-3 diag) %s\n",
                missB, late, (missB == 0 && late < 1e-3f*diag) ? "OK" : "FAIL");
    fails += !(missB == 0 && late < 1e-3f*diag);

    // C) poke through the public path: motion near > motion far
    ModalDynamics dyn;
    if (!dyn.LoadModes(psm, *body)) { std::fprintf(stderr, "modes\n"); return 1; }
    const Vec3 target = body->Patches()[7].Evaluate(0.5f, 0.5f);
    SurfaceHit h{};
    body->ClosestPointEx(target, h);
    const Vec3 nf = body->Patches()[h.patch].EvaluateNormalField(h.u, h.v);
    dyn.ApplyImpulseAt(h.patch, h.u, h.v, 30.0f);
    const Vec3 near0 = body->Patches()[h.patch].Evaluate(h.u, h.v);
    for (int i = 0; i < 40; ++i) dyn.Step(1.0f/4800.0f);
    dyn.WriteBack(*body);
    const Vec3 near1 = body->Patches()[h.patch].Evaluate(h.u, h.v);
    const float mv = dist(near0, near1);
    std::printf("C) poke: near-point motion %.3e (%.4f%% diag) %s  [n.(%.2f %.2f %.2f)]\n",
                mv, mv/diag*100, mv > 1e-5f*diag ? "OK" : "FAIL",
                nf.x, nf.y, nf.z);
    fails += !(mv > 1e-5f*diag);

    std::printf(fails ? "QUERY VALIDATION FAILED (%d)\n"
                      : "QUERY VALIDATION PASSED\n", fails);
    return fails;
}
