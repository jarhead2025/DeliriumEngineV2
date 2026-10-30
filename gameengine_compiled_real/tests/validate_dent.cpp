// =============================================================================
// validate_dent.cpp - Milestone 5 gate. Live local deformation:
//  A) DEPTH: the surface point nearest the dent center must displace by
//     the requested depth within 15% (LSQ fit + anchor quantization).
//  B) LOCALITY: patches beyond 2.5x the radius must not move (> 1e-5
//     of the diagonal counts as motion).
//  C) SEAMS: max gap after refine+dent+restitch stays under the 0.05%
//     subpixel budget.
//  D) REFINEMENT: patch count grew (RefineWhere actually split) and the
//     stitch blob stayed consistent (8 B per surface).
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* psf = argc > 1 ? argv[1] : "assets/torus.psf";
    auto body = PSFBody::FromFile(psf);
    if (!body) { std::fprintf(stderr, "load failed\n"); return 1; }
    const Vec3 sp = body->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    int fails = 0;

    // dent target: a point on the surface (patch 42, center) + its normal
    const Vec3 c0 = body->Patches()[42].Evaluate(0.5f, 0.5f);
    const Vec3 n0 = body->Patches()[42].EvaluateNormalField(0.5f, 0.5f);
    const float radius = 0.06f * diag, depth = 0.012f * diag;

    // far-field snapshot BEFORE the dent
    struct Far { uint32_t pid; float u, v; Vec3 p; };
    std::vector<Far> far;
    for (uint32_t pid = 0; pid < body->Patches().size(); pid += 7) {
        const Vec3 pc = body->Patches()[pid].Evaluate(0.5f, 0.5f);
        const float d = std::sqrt((pc.x-c0.x)*(pc.x-c0.x)
                      + (pc.y-c0.y)*(pc.y-c0.y) + (pc.z-c0.z)*(pc.z-c0.z));
        if (d > 2.5f * radius) far.push_back({pid, 0.5f, 0.5f, pc});
    }

    const size_t ns0 = body->Patches().size();
    body->RefineWhere(c0, radius, radius * 0.8f);
    const size_t ns1 = body->Patches().size();
    body->ApplyRadialDent(c0, radius, depth);

    // A) depth at center: find nearest surface sample to c0's normal line
    float best_disp = 0, best_d = 1e9f;
    for (uint32_t pid = 0; pid < body->Patches().size(); ++pid) {
        for (int i = 0; i <= 4; ++i)
            for (int j = 0; j <= 4; ++j) {
                const float u = i/4.0f, v = j/4.0f;
                const Vec3 p = body->Patches()[pid].Evaluate(u, v);
                // distance from ORIGINAL center point, displaced inward
                const Vec3 q = { p.x + depth*n0.x*0, p.y, p.z }; (void)q;
                const float d = std::sqrt((p.x-(c0.x-depth*n0.x))*(p.x-(c0.x-depth*n0.x))
                              + (p.y-(c0.y-depth*n0.y))*(p.y-(c0.y-depth*n0.y))
                              + (p.z-(c0.z-depth*n0.z))*(p.z-(c0.z-depth*n0.z)));
                if (d < best_d) {
                    best_d = d;
                    const float dx = p.x-c0.x, dy = p.y-c0.y, dz = p.z-c0.z;
                    best_disp = -(dx*n0.x + dy*n0.y + dz*n0.z); // inward +
                }
            }
    }
    const float depth_err = std::fabs(best_disp - depth) / depth;
    std::printf("A) depth: requested %.4f, measured %.4f (err %.1f%%) %s\n",
                depth, best_disp, depth_err*100,
                depth_err < 0.15f ? "OK" : "FAIL");
    fails += !(depth_err < 0.15f);

    // B) locality (note: RefineWhere replaced/added ids only NEAR the
    // dent; far snapshot pids taken before are still the same records
    // because splits replace in place / append at the end -- but a far
    // pid could have been a split TARGET only if near, so it's stable)
    float far_move = 0;
    for (const Far& f : far) {
        const Vec3 p = body->Patches()[f.pid].Evaluate(f.u, f.v);
        const float d = std::sqrt((p.x-f.p.x)*(p.x-f.p.x)
                      + (p.y-f.p.y)*(p.y-f.p.y) + (p.z-f.p.z)*(p.z-f.p.z));
        if (d > far_move) far_move = d;
    }
    std::printf("B) locality: max far-field motion %.2e (%.4f%% diag) %s\n",
                far_move, far_move/diag*100,
                far_move < 1e-5f*diag ? "OK" : "FAIL");
    fails += !(far_move < 1e-5f*diag);

    // C) seams
    const float g = body->Seams().MaxGap(*body);
    std::printf("C) seams: max gap %.4f%% diag (budget 0.05%%) %s\n",
                g/diag*100, g < 0.0005f*diag ? "OK" : "FAIL");
    fails += !(g < 0.0005f*diag);

    // D) refinement + bookkeeping
    const bool d_ok = ns1 > ns0
        && body->StitchBlob().size() == body->Patches().size()*8;
    std::printf("D) refine: %zu -> %zu patches, stitch blob %s\n",
                ns0, ns1, d_ok ? "OK" : "FAIL");
    fails += !d_ok;

    std::printf(fails ? "DENT VALIDATION FAILED (%d)\n"
                      : "DENT VALIDATION PASSED\n", fails);
    return fails;
}
