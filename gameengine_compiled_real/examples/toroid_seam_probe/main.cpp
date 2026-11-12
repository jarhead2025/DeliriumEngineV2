// Seam statistics on assets/torus.psf.
//
// THE QUESTION (user): with a dual patch system where each set covers the
// other's empty areas, we should not need to match edges that closely --
// "more breathing space" -- and a deliberately made "empty path" could be
// mapped to generate compensating anchors for the second, mathematically
// driven patch set.
//
// THE MEASUREMENT: how do the toroid's patch edges actually resolve?
//   index-matched : two patches share the EXACT same anchor pair (tight)
//   cross-chart   : two edges coincide in POSITION (breathing space)
//   free          : no partner -- the "empty areas"
// plus what the engine's own SeamGraph finds, and how many seams are
// UN-STITCHABLE (needing a bridge -- the Hermite fillet).
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/Seam.h"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>
#include <set>
#include <memory>

using namespace DeliRium;

static const int EP[4][2] = {{0,3},{1,2},{0,1},{3,2}};

static double dist(const Vec3& a, const Vec3& b) {
    const double dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "assets/torus.psf";
    auto body_ptr = PSFBody::FromFile(path);
    if (!body_ptr) {
        std::fprintf(stderr, "FAILED to load %s\n", path);
        return 1;
    }
    // DO NOT MOVE the body out of its unique_ptr. FromRecords binds the
    // AnchorPool to &body->anchor_blob -- an address INSIDE the heap
    // allocation. Moving the body to the stack relocates anchor_blob, so
    // the pool's bound pointer dangles into the freed heap block, and the
    // first pool.Position() reads freed memory (a segfault, not a bad
    // value -- the read "succeeds"). Keep the body where it was born.
    PSFBody& body = *body_ptr;
    const auto& S = body.SurfaceBlob();
    const auto& pool = body.AnchorsConst();
    const Vec3 sp = body.BBoxSpan();
    const double diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    std::printf("torus: %zu patches, %zu anchors, diag=%.4f\n",
                S.size(), body.AnchorBlob().size(), diag);

    // --- classify every edge by INDEX match ---
    std::map<std::pair<uint16_t,uint16_t>, std::vector<int>> by_pair;
    struct E { uint32_t s; uint8_t e; uint16_t a0, a1; };
    std::vector<E> edges;
    for (uint32_t s = 0; s < S.size(); ++s) {
        for (uint8_t e = 0; e < 4; ++e) {
            E r{ s, e, S[s].anchor_idx[EP[e][0]], S[s].anchor_idx[EP[e][1]] };
            by_pair[{std::min(r.a0,r.a1), std::max(r.a0,r.a1)}].push_back((int)edges.size());
            edges.push_back(r);
        }
    }
    int index_pairs = 0, index_edges = 0;
    std::vector<int> free_edges;
    for (auto& kv : by_pair) {
        if (kv.second.size() == 2) { index_pairs++; index_edges += 2; }
        else free_edges.insert(free_edges.end(), kv.second.begin(), kv.second.end());
    }
    std::printf("\nedge classification (by INDEX):\n");
    std::printf("  index-matched seam pairs : %d  (%d edges)\n", index_pairs, index_edges);
    std::printf("  unmatched-by-index edges  : %zu\n", free_edges.size());

    // --- of the free edges, how many have a POSITION match (cross-chart)? ---
    auto mid_of = [&](int ei) {
        const E& r = edges[ei];
        const Vec3 p0 = pool.Position(r.a0), p1 = pool.Position(r.a1);
        return Vec3{0.5f*(p0.x+p1.x), 0.5f*(p0.y+p1.y), 0.5f*(p0.z+p1.z)};
    };
    std::printf("\nposition matching on the FREE edges (the 'breathing space'):\n");
    for (double frac : {0.004, 0.008, 0.016, 0.032, 0.064}) {
        const double cell = frac * diag, wtol = frac * diag;
        std::map<std::tuple<long long,long long,long long>, std::vector<int>> buckets;
        for (int ei : free_edges) {
            const Vec3 m = mid_of(ei);
            auto qc=[&](float x){ return (long long)std::llround(x/cell); };
            buckets[{qc(m.x),qc(m.y),qc(m.z)}].push_back(ei);
        }
        int pairs = 0;
        std::set<int> used;
        for (auto& kv : buckets) {
            const auto& lst = kv.second;
            for (size_t i=0;i<lst.size();++i) {
                if (used.count(lst[i])) continue;
                const E& A = edges[lst[i]];
                for (size_t j=i+1;j<lst.size();++j) {
                    if (used.count(lst[j])) continue;
                    const E& B = edges[lst[j]];
                    if (A.s == B.s) continue;
                    const Vec3 a0=pool.Position(A.a0),a1=pool.Position(A.a1);
                    const Vec3 b0=pool.Position(B.a0),b1=pool.Position(B.a1);
                    const bool same = dist(a0,b0)<wtol && dist(a1,b1)<wtol;
                    const bool flip = dist(a0,b1)<wtol && dist(a1,b0)<wtol;
                    if (same||flip) { used.insert(lst[i]); used.insert(lst[j]); pairs++; }
                }
            }
        }
        std::printf("  tol=%5.3f*diag -> cross-chart pairs=%4d  covers %zu/%zu free edges\n",
                    frac, pairs, used.size(), free_edges.size());
    }

    // --- what the engine's SeamGraph actually finds ---
    SeamGraph sg;
    sg.Build(body);
    const auto& seams = sg.All();
    std::printf("\nSeamGraph::Build -> %zu seams\n", seams.size());
    const double gap = sg.MaxGap(body);
    std::printf("  MaxGap = %.6f  (%.4f%% of the diagonal)\n", gap, 100.0*gap/diag);

    // how many patches have a partner?
    std::set<uint32_t> partnered;
    for (const auto& s : seams) { partnered.insert(s.patch_a); partnered.insert(s.patch_b); }
    std::printf("  patches with a seam partner : %zu / %zu  (%.1f%%)\n",
                partnered.size(), S.size(), 100.0*partnered.size()/std::max<size_t>(1,S.size()));
    std::printf("  patches with NO partner      : %zu  (the 'empty areas')\n",
                S.size()-partnered.size());

    // --- UN-STITCHABLE seams: the ones that need a bridge (Hermite fillet) ---
    const std::vector<uint32_t> bridge = sg.BridgeSeams(body, 0.01f);
    std::printf("\nBridgeSeams(tol=0.01) -> %zu seams need a Hermite BRIDGE\n",
                bridge.size());
    std::printf("  (a bridge is a DERIVED fillet: tangent-continuous with both sides,\n"
                "   zero stored bytes -- the 'compensating surface' mechanism)\n");

    // --- and the weld, which is the global 'average them' relaxation ---
    // A COPY is hazardous: PSFBody's copy constructor copies the AnchorPool,
    // whose `blob` pointer still points at the ORIGINAL body's blob, so the
    // copy's pool reads the wrong (or freed) memory. Rebuild the copy
    // through FromRecords, which re-binds the pool to its OWN blob.
    auto copy_ptr = PSFBody::FromRecords(body.AnchorBlob(), body.SurfaceBlob(),
                                         body.StitchBlob(), body.BBoxMin(),
                                         body.BBoxSpan(), body.StitchScale());
    if (!copy_ptr) { std::fprintf(stderr, "copy failed\n"); return 1; }
    PSFBody& copy = *copy_ptr;
    SeamGraph sg2;
    sg2.Build(copy);
    const double before = sg2.MaxGap(copy);
    const double after = sg2.HarmonicWeld(copy, 24, 0.2f);
    std::printf("\nHarmonicWeld(24 iters, smooth=0.2):\n");
    std::printf("  max gap %.6f -> %.6f  (%.4f%% -> %.4f%% of diag)\n",
                before, after, 100.0*before/diag, 100.0*after/diag);
    return 0;
}
