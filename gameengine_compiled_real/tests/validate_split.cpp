// =============================================================================
// validate_split.cpp - Milestone 4 gate. Live subdivision checks:
//  A) FIDELITY: each child must reproduce its parent's surface; max
//     deviation sampled at 17x17 over the parent domain < 0.15% of diag.
//  B) SEAMS: new T-junction seams carry an irreducible TANGENTIAL
//     residual (child normal field = lerp over the half-edge vs parent
//     field = normalized lerp over the full edge; a scalar stitch cannot
//     correct perpendicular to itself). Measured floor ~0.027% of the
//     diagonal = ~0.14 px on a 400 px object. Gate: absolute subpixel
//     budget, max gap < 0.05% diag after splits+restitch.
//  C) BOOKKEEPING: +3 surfaces per split, anchors grow <= 5 per split,
//     stitch blob stays 8 bytes per surface.
//  D) RECURSION: splitting a child again keeps A and C holding.
// Known limitation (documented): >1 level difference between neighbors
// is not matched by the seam graph yet (2:1 balance rule pending), so
// recursive splits here use interior children only.
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace DeliRium;

struct ParentSnapshot {
    std::vector<Vec3> pts;          // 17x17 over parent domain
    static const int G = 17;
    void Take(const Patch& p) {
        pts.clear();
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j)
                pts.push_back(p.Evaluate(i/float(G-1), j/float(G-1)));
    }
    // children_out[4] in quadrant order (u-,v-),(u-,v+),(u+,v-),(u+,v+)
    float MaxDev(const PSFBody& body, const uint32_t c[4]) const {
        float worst = 0;
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float pu = i/float(G-1), pv = j/float(G-1);
                const int qu = pu > 0.5f, qv = pv > 0.5f;
                const uint32_t cid = c[qu*2 + qv];
                const float cu = pu*2 - qu, cv = pv*2 - qv;
                const Vec3 a = body.Patches()[cid].Evaluate(cu, cv);
                const Vec3 b = pts[i*G + j];
                const float d = std::sqrt((a.x-b.x)*(a.x-b.x)
                              + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
                if (d > worst) worst = d;
            }
        return worst;
    }
};

int main(int argc, char** argv) {
    const char* psf = argc > 1 ? argv[1] : "assets/torus.psf";
    auto body = PSFBody::FromFile(psf);
    if (!body) { std::fprintf(stderr, "load failed\n"); return 1; }
    const Vec3 sp = body->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);

    const size_t ns0 = body->Patches().size();
    const size_t na0 = body->Anchors().Count();
    const float  g0  = body->Seams().MaxGap(*body);
    std::printf("loaded: %zu patches, %zu anchors, %zu seams, "
                "max gap %.4f%% diag\n", ns0, na0,
                body->Seams().SeamCount(), g0/diag*100);
    int fails = 0;

    // A+B+C: split 5 mutually non-adjacent patches
    const uint32_t targets[5] = {0, 97, 211, 350, 489};
    float worst_dev = 0;
    size_t splits = 0;
    for (uint32_t pid : targets) {
        ParentSnapshot snap; snap.Take(body->Patches()[pid]);
        uint32_t kids[4];
        if (!body->SplitPatch(pid, kids)) { ++fails; continue; }
        ++splits;
        worst_dev = std::fmax(worst_dev, snap.MaxDev(*body, kids));
    }
    std::printf("A) fidelity: worst child-vs-parent deviation %.4f%% diag %s\n",
                worst_dev/diag*100, worst_dev/diag < 0.0015f ? "OK" : "FAIL");
    fails += !(worst_dev/diag < 0.0015f);

    const float g1 = body->Seams().MaxGap(*body);
    const bool seams_ok = g1 < 0.0005f * diag;     // subpixel budget
    std::printf("B) seams: max gap %.4f%% -> %.4f%% diag (budget 0.05%%) %s\n",
                g0/diag*100, g1/diag*100, seams_ok ? "OK" : "FAIL");
    fails += !seams_ok;

    const size_t ns1 = body->Patches().size();
    const size_t na1 = body->Anchors().Count();
    const bool counts_ok = (ns1 == ns0 + 3*splits)
                        && (na1 <= na0 + 5*splits)
                        && (body->StitchBlob().size() == ns1*8);
    std::printf("C) bookkeeping: %zu->%zu patches, %zu->%zu anchors %s\n",
                ns0, ns1, na0, na1, counts_ok ? "OK" : "FAIL");
    fails += !counts_ok;

    // D: recursive split of an interior child of the first split
    {
        ParentSnapshot snap; snap.Take(body->Patches()[0]);
        uint32_t kids[4];
        const bool ok = body->SplitPatch(0, kids);
        const float dev = ok ? snap.MaxDev(*body, kids) : 1e9f;
        std::printf("D) recursion: nested split dev %.4f%% diag %s\n",
                    dev/diag*100, ok && dev/diag < 0.0015f ? "OK" : "FAIL");
        fails += !(ok && dev/diag < 0.0015f);
    }

    std::printf(fails ? "SPLIT VALIDATION FAILED (%d)\n"
                      : "SPLIT VALIDATION PASSED\n", fails);
    return fails;
}
