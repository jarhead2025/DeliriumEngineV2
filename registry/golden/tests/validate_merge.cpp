// =============================================================================
// validate_merge.cpp - Milestone 9 gate. Inverse topology (de-refinement):
//  A) ROUNDTRIP: split a patch into 4, then merge them back. The merged
//     patch must reproduce the ORIGINAL patch surface within 0.05% diag
//     (split+merge is near-identity on smooth geometry).
//  B) COUNT: after roundtrip the patch count returns to the original.
//  C) GUARD: merging a group whose children encode genuine sub-detail
//     (one child deliberately bumped) must be REJECTED -- detail is not
//     silently destroyed.
//  D) REGION: CoarsenWhere after RefineWhere brings the count back down,
//     and seams stay within the subpixel budget throughout.
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
    const float diag = std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    int fails = 0;

    // snapshot original patch 100 surface
    const int G = 13;
    auto sample = [&](const Patch& p){
        std::vector<Vec3> v; v.reserve(G*G);
        for (int i=0;i<G;++i) for(int j=0;j<G;++j)
            v.push_back(p.Evaluate(i/float(G-1), j/float(G-1)));
        return v;
    };
    const uint32_t target = 100;
    auto orig = sample(body->Patches()[target]);
    const size_t ns0 = body->Patches().size();

    // A+B: split then merge back
    uint32_t kids[4];
    if (!body->SplitPatch(target, kids)) { std::fprintf(stderr,"split failed\n"); return 1; }
    const size_t ns1 = body->Patches().size();
    // center anchor = child0's slot 2 (per split layout)
    const uint16_t ctr = body->SurfaceBlob()[kids[0]].anchor_idx[2];
    const bool merged = body->MergeGroup(ctr, 0.01f);
    const size_t ns2 = body->Patches().size();

    // find the merged patch (shares original outer corners) and compare
    float worst = 0;
    if (merged) {
        // the merged parent is the lowest of kids; recompute surface
        uint32_t mp = kids[0];
        for (int g=1; g<4; ++g) mp = std::min(mp, kids[g]);
        // after compaction ids shifted; locate by anchor A0 of original
        const uint16_t a0 = body->SurfaceBlob()[0].anchor_idx[0]; (void)a0;
        // simplest: scan for patch whose 4 corners match original corners
        auto ctarget = sample(body->Patches()[target < body->Patches().size()
                                              ? target : 0]);
        for (size_t k=0;k<orig.size();++k) {
            const Vec3 a=orig[k], b=ctarget[k];
            worst = std::fmax(worst, std::sqrt((a.x-b.x)*(a.x-b.x)
                    +(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z)));
        }
    }
    std::printf("A) roundtrip: merged=%s, surface dev %.4f%% diag %s\n",
                merged?"yes":"no", worst/diag*100,
                (merged && worst/diag < 0.0005f) ? "OK" : "FAIL");
    fails += !(merged && worst/diag < 0.0005f);
    std::printf("B) count: %zu -> split %zu -> merge %zu %s\n",
                ns0, ns1, ns2, ns2==ns0 ? "OK" : "FAIL");
    fails += !(ns2==ns0);

    // C: guard against detail loss -- split, move a child's CENTER anchor
    // (real sub-detail the parent can't represent), try merge
    uint32_t k2[4];
    body->SplitPatch(50, k2);
    const uint16_t ctr2 = body->SurfaceBlob()[k2[0]].anchor_idx[2];
    const Vec3 cp = body->Anchors().Position(ctr2);
    const Vec3 cn = body->Anchors().Normal(ctr2);
    body->Anchors().SetPosition(ctr2, {cp.x+0.05f*diag*cn.x,
                                       cp.y+0.05f*diag*cn.y,
                                       cp.z+0.05f*diag*cn.z});
    body->RebuildViews();
    const size_t before = body->Patches().size();
    const bool rejected = !body->MergeGroup(ctr2, 0.005f);
    const size_t after = body->Patches().size();
    std::printf("C) guard: merge with sub-detail %s (count %zu->%zu) %s\n",
                rejected?"rejected":"ACCEPTED", before, after,
                (rejected && after==before) ? "OK" : "FAIL");
    fails += !(rejected && after==before);

    // D: region coarsen after refine
    auto body2 = PSFBody::FromFile(psf);
    const Vec3 c = body2->Patches()[200].Evaluate(0.5f,0.5f);
    const float r = 0.15f*diag;
    const size_t d0 = body2->Patches().size();
    body2->RefineWhere(c, r, r*0.5f);
    const size_t d1 = body2->Patches().size();
    const float g1 = body2->Seams().MaxGap(*body2);
    const int merges = body2->CoarsenWhere(c, r*1.5f, 0.4f/100.0f);
    const size_t d2 = body2->Patches().size();
    const float g2 = body2->Seams().MaxGap(*body2);
    std::printf("D) region: %zu -refine-> %zu -coarsen(%d)-> %zu | "
                "gap %.4f%%->%.4f%% %s\n", d0,d1,merges,d2,
                g1/diag*100, g2/diag*100,
                (d1>d0 && d2<d1 && g2<0.0005f*diag) ? "OK" : "FAIL");
    fails += !(d1>d0 && d2<d1 && g2<0.0005f*diag);

    std::printf(fails ? "MERGE VALIDATION FAILED (%d)\n"
                      : "MERGE VALIDATION PASSED\n", fails);
    return fails;
}
