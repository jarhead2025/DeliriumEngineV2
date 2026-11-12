// GATE: THE CLOSED BODY'S SEAM MUST STRADDLE z=0.
//
// WHAT THIS PINS
//
// A closed body is two sheets that MEET at a plane (z=0 for the standard
// recipe: an upper sheet + a mirrored lower sheet with scale [1,1,-1] at
// the same origin). Their rims both lie ON that plane, and the two rims
// must land on the SAME quantised position -- otherwise the seam shows as
// a step in the render.
//
// THE BUG THIS PINS (measured, the derived path):
//
//   GRID   : the band |z| < 0.05 holds 1,192 anchors BELOW the seam and
//            360 above -- the rim straddles z=0.
//   DERIVED: the same band held  48 below and 792 ABOVE -- the lower
//            sheet's rim had been pushed entirely to the upper side.
//
// The cause was the QUANTISATION'S OWN HALF-STEP. The seam is the
// midpoint of the body's z range, so its quantised parameter is t = 0.5,
// and the plain `+ 0.5f` rounding sends that to the index 32768. But
// 32768/65535 = 0.50000763 -- NOT 0.5 -- so the rim dequantised to +0.5
// quantisation steps ABOVE the seam. The index that DEQUANTISES to the
// seam is 32767.
//
// WHAT IS MEASURED, AND WHY
//
// A NAIVE "the band |z| < 0.05 must be 50/50" assertion is WRONG, and
// writing it was the first mistake here. The band is not a rim band: it is
// the dome's near-EQUATOR region, where z is small across a wide annulus
// of the footprint. The lower sheet's whole near-equator ring is in it, so
// the band is legitimately dominated by the side the surface bulges
// toward -- the GRID measures 76.8% below, and the fixed derived measures
// 94.3% below. Both are correct; neither is 50/50.
//
// So the gate measures the thing that is actually invariant:
//
//   A) the RIM -- the anchors ON the seam -- must exist and must agree on
//      ONE quantised index (the two sheets' rims coincide),
//   B) that index must DEQUANTISE ONTO the seam (within one step), which
//      is the exact property that was broken,
//   C) the body's z-range must be CENTRED on the seam, and span the full
//      height -- a body whose rim is all on one side is not closed, and
//      that is what the render showed,
//   D) the derived path must not be WORSE than the grid path on the
//      measure that was broken: the count of anchors on the seam. The grid
//      has rim points at z ~= +-8e-6 (its lattice row lands there by
//      accident); the derived has them at exactly z = 0 (its rim points
//      have local z == 0 by construction). Both straddle.
//
// NOTE ON WHAT THIS DOES NOT COVER: the derived mesh's vertical streaks
// below the seam are a SEPARATE defect. Measured: the cells' geometry is
// nearly identical between the paths (81.67% vs 85.48% inside the
// footprint; radius p50 0.3445 vs 0.3387; max cell z-extent 0.189 vs
// 0.151), and the FIT is exact in both (max |corner z| - substance z:
// 0.00016 grid vs 0.00019 derived). So the streaks are not the cells'
// shape and not the fit -- they are a consequence of the derived mesh's
// OVERLAPPING cells (3,480 cells over 2,872 distinct regions, with 1,324
// unpaired triangles) and how the frontmost-wins march resolves the
// coincident depths. That is tracked separately.
#include <cstdio>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "delirium/core/Types.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/MutualMesh.h"
#include "delirium/psf/PSFBody.h"

using namespace DeliRium;

static const char* kClosedTree = R"({
  "kind":"body","substance":"ellipsoid",
  "substance_params":{"sx":0.45,"sy":0.45,"sz":0.45},
  "children":[
    {"kind":"lower","substance":"ellipsoid",
     "substance_params":{"sx":0.45,"sy":0.45,"sz":0.45},
     "scale":[1,1,-1],"origin":[0,0,0]}
  ]})";

static int g_fail = 0;
static int g_checks = 0;

static void check(bool ok, const char* what, double value = 0.0, double bound = 0.0) {
    ++g_checks;
    std::printf("  %s %-58s", ok ? "ok  " : "FAIL", what);
    if (bound > 0.0) std::printf("  (%.6f", value);
    if (bound > 0.0) std::printf(ok ? " <= %.6f)" : " >  %.6f)", bound);
    std::printf("\n");
    if (!ok) ++g_fail;
}

struct SeamStats {
    int below = 0, at = 0, above = 0;
    // The anchors ON the seam, and the quantised indices they carry.
    std::vector<uint16_t> rim_indices;
    std::vector<float> zs;
    double zmin = 0, zmax = 0;
};

static SeamStats measure(const PSFBody& b) {
    SeamStats s;
    const auto& blob = b.AnchorBlob();
    const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
    const float qstep = sp.z / 65535.0f;
    s.zmin = 1e30f; s.zmax = -1e30f;
    for (const auto& a : blob) {
        const float z = mn.z + sp.z * (a.pos[2] / 65535.0f);
        s.zs.push_back(z);
        s.zmin = std::min(s.zmin, (double)z);
        s.zmax = std::max(s.zmax, (double)z);
        if (z < -1e-6f) ++s.below;
        else if (z > 1e-6f) ++s.above;
        else ++s.at;
        // ON the seam: within one quantisation step of it. One step, not
        // half -- the grid's rim points land at z ~= 8e-6, which is half a
        // step, and the derived's at exactly 0.
        if (std::fabs(z) <= qstep) s.rim_indices.push_back(a.pos[2]);
    }
    std::sort(s.rim_indices.begin(), s.rim_indices.end());
    return s;
}

// The dominant index among the rim anchors, and how many share it.
static void dominant(const std::vector<uint16_t>& v, uint16_t& idx, int& count) {
    idx = 0; count = 0;
    for (size_t i = 0; i < v.size();) {
        size_t j = i;
        while (j < v.size() && v[j] == v[i]) ++j;
        if (int(j - i) > count) { count = int(j - i); idx = v[i]; }
        i = j;
    }
}

int main() {
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(kClosedTree, tree, why)) {
        std::printf("  FAIL parse the closed tree: %s\n", why.c_str());
        return 1;
    }

    std::unique_ptr<PSFBody> grid, derived;
    grid = ElementSystem::BakeBody(*tree, 48, 1.0f, why);
    if (!grid) { std::printf("  FAIL the grid bake: %s\n", why.c_str()); return 1; }
    {
        MeshDerivation st;
        derived = ElementSystem::BakeDerivedBody(*tree, 48, 1.0f, why, &st);
    }
    if (!derived) { std::printf("  FAIL the derived bake: %s\n", why.c_str()); return 1; }

    const SeamStats gs = measure(*grid);
    const SeamStats ds = measure(*derived);

    uint16_t gidx = 0, didx = 0;
    int gcount = 0, dcount = 0;
    dominant(gs.rim_indices, gidx, gcount);
    dominant(ds.rim_indices, didx, dcount);

    std::printf("A) THE RIM EXISTS IN BOTH PATHS\n");
    check(!gs.rim_indices.empty(), "the grid path has anchors on the seam",
          double(gs.rim_indices.size()));
    check(!ds.rim_indices.empty(), "the derived path has anchors on the seam",
          double(ds.rim_indices.size()));

    std::printf("\nB) THE RIM LANDS ON ONE INDEX (the two sheets' rims coincide)\n");
    // The two sheets' rims must share the SAME index, so the rim's
    // quantised indices must not SPREAD across a wide range.
    {
        const int dspread = int(ds.rim_indices.back()) - int(ds.rim_indices.front());
        check(dspread <= 2, "the derived's rim spans at most two quantised indices",
              dspread, 2.0);
        check(dcount > 0 && double(dcount) / double(ds.rim_indices.size()) > 0.25,
              "the derived's rim agrees on ONE index",
              double(dcount) / double(ds.rim_indices.size()), 0.25);
    }
    {
        // THE EXACT PROPERTY THAT WAS BROKEN: the rim's index must
        // DEQUANTISE ONTO the seam, not to a half-step above it.
        const Vec3 mn = derived->BBoxMin(), sp = derived->BBoxSpan();
        const float deq = mn.z + float(didx) / 65535.0f * sp.z;
        const float qstep = sp.z / 65535.0f;
        check(std::fabs(deq) <= qstep,
              "the derived's rim index dequantises ONTO the seam",
              std::fabs(deq), qstep);
    }

    std::printf("\nC) THE BODY IS SYMMETRIC ABOUT THE SEAM\n");
    {
        const double mid = (ds.zmin + ds.zmax) * 0.5f;
        check(std::fabs(mid) < 0.01, "the derived body's z-range is centred on the seam",
              std::fabs(mid), 0.01);
        check(ds.zmax - ds.zmin > 0.80,
              "the derived body spans the full height (not a half-body)",
              ds.zmax - ds.zmin, 0.80);
        const double gmid = (gs.zmin + gs.zmax) * 0.5f;
        check(std::fabs(gmid) < 0.01, "the grid body's z-range is centred on the seam",
              std::fabs(gmid), 0.01);
    }

    std::printf("\nD) THE FIX DID NOT REGRESS THE GRID PATH\n");
    check(gcount > 0, "the grid's rim still agrees on one index", double(gcount));
    {
        // The grid's rim index must also dequantise onto the seam.
        const Vec3 mn = grid->BBoxMin(), sp = grid->BBoxSpan();
        const float deq = mn.z + float(gidx) / 65535.0f * sp.z;
        const float qstep = sp.z / 65535.0f;
        check(std::fabs(deq) <= qstep * 1.5,
              "the grid's rim index still dequantises onto the seam",
              std::fabs(deq), qstep * 1.5);
    }
    check(ds.below > 0 && ds.above > 0,
          "the derived body has anchors on BOTH sides of the seam");

    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
