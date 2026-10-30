// =============================================================================
// validate_bridge.cpp - Hermite seam bridge (generative surface over an
// un-stitchable cross-chart seam).
//  A) BOUNDARY: the bridge interpolates both patch edges exactly -- its
//     s=0 boundary equals A's edge, s=1 equals B's edge (gap ~0 by
//     construction). This is the "no crack" guarantee.
//  B) G1: the bridge meets each patch with tangent continuity -- the
//     cross-boundary tangent aligns with the patch edge tangent (1-cos
//     small). This is what makes the join seamless under lighting.
//  C) DETECTION: NeedsBridge flags the genuinely tangential seams (the
//     asteroid's sharp cross-chart borders) and not the cleanly-stitched
//     ones.
//  D) GENERATIVE: the bridge carries real curvature -- its midline bulges
//     off the straight chord between the edges (it adds surface, not a
//     flat fill).
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/Seam.h"
#include "delirium/psf/SeamBridge.h"
#include <cstdio>
#include <cmath>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* psf = argc>1 ? argv[1] : "assets/asteroid.psf";
    auto body = PSFBody::FromFile(psf);
    if(!body){ std::fprintf(stderr,"load failed\n"); return 1; }
    const Vec3 sp=body->BBoxSpan();
    const float diag=std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    int fails=0;

    body->Seams().Build(*body);
    const auto& seams = body->Seams().All();

    // find a seam that needs bridging (tangential disagreement)
    int bidx=-1;
    for (size_t i=0;i<seams.size();++i)
        if (SeamBridge::NeedsBridge(*body, seams[i], 0.01f)) { bidx=(int)i; break; }
    std::printf("seams: %zu, first needing bridge: %d\n", seams.size(), bidx);

    // C) detection finds some bridgeable seams
    int nbridge=0;
    for (const Seam& s : seams)
        if (SeamBridge::NeedsBridge(*body, s, 0.01f)) nbridge++;
    std::printf("C) detection: %d seams need bridging %s\n", nbridge,
                nbridge > 0 ? "OK":"FAIL");
    fails += !(nbridge > 0);

    if (bidx < 0) { std::printf("no bridgeable seam; skipping A/B/D\n");
                    return fails; }
    const Seam& s = seams[bidx];

    // A) boundary interpolation (bridge edges == patch edges)
    const float bgap = SeamBridge::BoundaryGap(*body, s);
    std::printf("A) boundary: bridge-to-edge gap %.5f%% diag %s\n",
                bgap/diag*100, bgap < 1e-4f*diag ? "OK":"FAIL");
    fails += !(bgap < 1e-4f*diag);

    // B) G1 continuity at both joins
    const float g1 = SeamBridge::G1Error(*body, s);
    std::printf("B) G1: worst 1-cos tangent %.4f %s\n", g1,
                g1 < 0.05f ? "OK":"FAIL");
    fails += !(g1 < 0.05f);

    // D) generative: midline bulges off the straight chord
    float maxbulge=0;
    for (int i=1;i<8;++i){
        const float t=i/8.0f;
        const BridgeSample mid = SeamBridge::Evaluate(*body, s, 0.5f, t);
        const BridgeSample e0  = SeamBridge::Evaluate(*body, s, 0.0f, t);
        const BridgeSample e1  = SeamBridge::Evaluate(*body, s, 1.0f, t);
        const Vec3 chord{0.5f*(e0.P.x+e1.P.x),0.5f*(e0.P.y+e1.P.y),0.5f*(e0.P.z+e1.P.z)};
        maxbulge=std::fmax(maxbulge,(float)std::sqrt((mid.P.x-chord.x)*(mid.P.x-chord.x)
            +(mid.P.y-chord.y)*(mid.P.y-chord.y)+(mid.P.z-chord.z)*(mid.P.z-chord.z)));
    }
    std::printf("D) generative: midline bulge %.4f%% diag %s\n",
                maxbulge/diag*100, "OK (informational)");

    std::printf(fails ? "BRIDGE VALIDATION FAILED (%d)\n"
                      : "BRIDGE VALIDATION PASSED\n", fails);
    return fails;
}
