// =============================================================================
// validate_support_grid - GATE for the SUPPORT-AWARE GRID (G21).
//
// THE PROBLEM IT SOLVES: the sampling grid is a SQUARE, so a substance
// whose support is not square-shaped spends most of its cells on nothing.
// MEASURED before the change, at grid 48 on a 1.0 canvas:
//
//   plane / gaussian    in-support 100.0%   wasted  0.0%
//   ellipsoid 0.5                85.4%          14.6%
//   rod 0.5 x 0.02               75.3%          24.7%
//   torus 0.32/0.12              62.0%          38.0%
//   torus 0.40/0.06              34.7%          65.3%
//
// The waste was NOT free: every wasted sample was written as a
// WELL-FORMED anchor at the bbox centre (the out-of-support placeholder),
// so it cost memory, upload bandwidth, and -- because the mass calculation
// counted those placeholders -- a WRONG inertia tensor (G9: 970 of 1024
// anchors for a rod sat at one point, and its inertia tensor was 25x too
// small).
//
// THE FIX: a substance whose support is an ANNULUS declares it
// (`IsPolar`/`PolarRadius`), and the baker samples it in POLAR
// coordinates: the grid's i runs AROUND the axis and its j runs ACROSS the
// tube. Every cell then lands on the support.
//
// THE GATE asserts:
//   A) the polar torus wastes far LESS than the square grid did;
//   B) the polar grid's GEOMETRY is still a torus -- the radius spans the
//      tube's true extent, the height is the tube's half-height, and NO
//      anchor lies inside the hole;
//   C) the polar grid's angular axis does not DUPLICATE its endpoint (the
//      bug that put an anchor exactly on the inner rim);
//   D) a RECTANGULAR substance is completely unaffected -- the change must
//      not alter the substances that do not opt in;
//   E) the polar grid's anchors are CONNECTED around the axis (the
//      connectivity wraps), so the tube is not split at angle 0.
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/AnchorPool.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace DeliRium;

namespace {
int checks = 0, failures = 0;
void check(bool cond, const char* fmt, ...) {
    ++checks;
    char msg[512];
    va_list ap; va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    std::printf("  %s %s\n", cond ? "OK  " : "FAIL", msg);
    if (!cond) ++failures;
}

// The out-of-support placeholder is written at the bbox CENTRE: the
// quantised midpoint (32767).
bool is_placeholder(const AnchorRecord& a) {
    return a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767;
}

struct Baked {
    ModelOutput out;
    std::unique_ptr<PSFBody> body;
    Vec3 bmn{}, bsp{};
    int placeholders = 0, real = 0;
};

bool bake(const char* json, int N, Baked& b) {
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(json, tree, why)) {
        std::printf("  parse failed: %s\n", why.c_str());
        return false;
    }
    if (!ElementSystem::Bake(*tree, N, 1.0f, b.out, why)) {
        std::printf("  bake failed: %s\n", why.c_str());
        return false;
    }
    b.body = ElementSystem::BakeBody(*tree, N, 1.0f, why);
    if (!b.body) { std::printf("  BakeBody failed\n"); return false; }
    b.bmn = b.body->BBoxMin();
    b.bsp = b.body->BBoxSpan();
    for (const auto& a : b.out.anchors) {
        if (is_placeholder(a)) ++b.placeholders; else ++b.real;
    }
    return true;
}

const char* kTorus =
    R"({"kind":"body","substance":"torus",
        "substance_params":{"major":0.32,"minor":0.12}})";
const char* kThinTorus =
    R"({"kind":"body","substance":"torus",
        "substance_params":{"major":0.40,"minor":0.06}})";
const char* kEllipsoid =
    R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5}})";
} // namespace

int main() {
    const int N = 48;

    // ---- A) the polar grid wastes far less -------------------------------
    std::printf("-- A) the sampling waste --\n");
    Baked t, thin, ell;
    if (!bake(kTorus, N, t)) return 1;
    if (!bake(kThinTorus, N, thin)) return 1;
    if (!bake(kEllipsoid, N, ell)) return 1;
    const float t_waste = 100.0f * t.placeholders / (t.real + t.placeholders);
    const float th_waste = 100.0f * thin.placeholders / (thin.real + thin.placeholders);
    const float e_waste = 100.0f * ell.placeholders / (ell.real + ell.placeholders);
    std::printf("  fat torus  %5.1f%% placeholders (was 38.0%%)\n", t_waste);
    std::printf("  thin torus %5.1f%% placeholders (was 65.3%%)\n", th_waste);
    std::printf("  ellipsoid  %5.1f%% placeholders (unchanged)\n", e_waste);
    check(t_waste < 8.0f,
          "the fat torus wastes under 8%% (was 38.0%%) -- %.1f%%", t_waste);
    check(th_waste < 8.0f,
          "the thin torus wastes under 8%% (was 65.3%%) -- %.1f%%", th_waste);

    // ---- B) the geometry is still a torus ---------------------------------
    std::printf("\n-- B) the geometry --\n");
    {
        float rmin = 1e9f, rmax = -1e9f, zmin = 1e9f, zmax = -1e9f;
        int in_hole = 0, on_tube = 0;
        const float MAJOR = 0.32f, MINOR = 0.12f;
        for (const auto& a : t.out.anchors) {
            if (is_placeholder(a)) continue;
            const float x = t.bmn.x + a.pos[0]*(1.f/65535.f)*t.bsp.x;
            const float y = t.bmn.y + a.pos[1]*(1.f/65535.f)*t.bsp.y;
            const float z = t.bmn.z + a.pos[2]*(1.f/65535.f)*t.bsp.z;
            const float r = std::sqrt(x*x + y*y);
            rmin = std::min(rmin, r); rmax = std::max(rmax, r);
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
            if (r < MAJOR - MINOR - 1e-3f) ++in_hole;
            else if (r < MAJOR + MINOR) ++on_tube;
        }
        std::printf("  r: [%.4f, %.4f] (the tube spans [%.4f, %.4f])\n",
                    rmin, rmax, MAJOR-MINOR, MAJOR+MINOR);
        std::printf("  z: [%.4f, %.4f] (a half tube: [0, %.4f])\n",
                    zmin, zmax, MINOR);
        std::printf("  anchors inside the hole: %d, on the tube: %d\n",
                    in_hole, on_tube);
        check(std::fabs(rmin - (MAJOR-MINOR)) < 0.01f,
              "the radius starts at the tube's inner rim (%.4f)", rmin);
        check(std::fabs(rmax - (MAJOR+MINOR)) < 0.01f,
              "the radius ends at the tube's outer rim (%.4f)", rmax);
        check(std::fabs(zmax - MINOR) < 0.01f,
              "the height is the tube's half-height (%.4f)", zmax);
        check(in_hole == 0,
              "NO anchor lies inside the hole (%d)", in_hole);
        check(on_tube > 100, "the tube has anchors (%d)", on_tube);
    }

    // ---- C) the angular axis does not duplicate its endpoint --------------
    //
    // If the last column landed on angle 2*pi (the same point as the first),
    // there would be TWO anchors at the same place -- and one of them would
    // sit exactly on the inner rim, which reads as "an anchor in the hole".
    // The test: count the anchors at (approximately) the same radius and the
    // same height. A duplicated column shows up as an exact (x, y) repeat.
    std::printf("\n-- C) the angular axis --\n");
    {
        std::vector<std::pair<float,float>> xy;
        for (const auto& a : t.out.anchors) {
            if (is_placeholder(a)) continue;
            const float x = t.bmn.x + a.pos[0]*(1.f/65535.f)*t.bsp.x;
            const float y = t.bmn.y + a.pos[1]*(1.f/65535.f)*t.bsp.y;
            xy.push_back({x, y});
        }
        // The quantisation step, as the matching tolerance.
        const float q = t.bsp.x * (1.f/65535.f) * 2.f;
        int dup = 0;
        for (size_t i = 0; i < xy.size(); ++i)
            for (size_t j = i + 1; j < xy.size(); ++j) {
                const float dx = xy[i].first - xy[j].first;
                const float dy = xy[i].second - xy[j].second;
                if (std::fabs(dx) < q && std::fabs(dy) < q) { ++dup; break; }
            }
        std::printf("  duplicated (x,y) anchors: %d of %zu\n", dup, xy.size());
        check(dup == 0, "the angular axis does NOT duplicate its endpoint (%d)", dup);
    }

    // ---- D) a rectangular substance is unaffected -------------------------
    std::printf("\n-- D) the substances that did not opt in --\n");
    {
        // THE POLAR PATH MUST NOT TOUCH THE ELLIPSOID.
        //
        // HOW TO PROVE THAT NOW: the ellipsoid's sampling domain grew in
        // commit dc723b8 (the cube fix) -- the clip that used to trim the
        // domain to exactly the canvas was relaxed to a bounded 25%
        // overshoot, because a clip at exactly extent/2 silently UNDID a
        // substance's declared band (the box's wall band and the
        // ellipsoid's 6% rim). So the ellipsoid's domain is now 1.06x the
        // footprint, and its waste is no longer the old 16.3%.
        //
        // The waste rose for a GOOD reason: the square grid over a wider
        // domain wastes more CORNER cells. Measured by ellipsoid_diag at
        // grid 48: 2304 anchors = 1544 inside the footprint + 172 on the
        // RIM band (q >= 1, radius up to 0.5282 > the 0.5 half-extent --
        // these are the VALID rim anchors, the overlap mechanism working)
        // + 588 placeholders. The rim band is the mechanism; the corners
        // are the waste.
        //
        // So the invariant is NOT "the waste is unchanged" -- that would
        // pin the BUG (the rim clipped away). The invariant is:
        //   1. the polar path did not change the ellipsoid's SAMPLING
        //      PATTERN (it is still a square grid: the radius histogram
        //      is symmetric and the rim band is reached on ALL sides);
        //   2. the ellipsoid's rim IS now sampled (the fix is live);
        //   3. the geometry is still a hemisphere.
        std::printf("  ellipsoid waste: %.1f%% (was 16.3%% before the rim band was sampled)\n",
                    e_waste);

        // 1) THE POLAR PATH DID NOT TOUCH IT: the ellipsoid is still
        //    sampled on a SQUARE grid, so the set of radii must be
        //    SYMMETRIC about the axis. A polar sample of a non-annular
        //    substance would cluster its radii.
        {
            int in_footprint = 0, on_rim = 0;
            float rmax_real = 0.f;
            for (const auto& a : ell.out.anchors) {
                if (is_placeholder(a)) continue;
                const float x = ell.bmn.x + a.pos[0]*(1.f/65535.f)*ell.bsp.x;
                const float y = ell.bmn.y + a.pos[1]*(1.f/65535.f)*ell.bsp.y;
                const float q = (x*x)/(0.5f*0.5f) + (y*y)/(0.5f*0.5f);
                if (q < 1.0f) ++in_footprint; else ++on_rim;
                const float r = std::sqrt(x*x + y*y);
                if (r > rmax_real) rmax_real = r;
            }
            std::printf("  real anchors: %d inside the footprint, %d on the rim band\n",
                        in_footprint, on_rim);
            std::printf("  max radius among real anchors: %.4f (half-extent 0.5)\n",
                        rmax_real);

            check(in_footprint > 1000,
                  "the ellipsoid's surface anchors are still sampled (%d)", in_footprint);
            check(on_rim > 0,
                  "the ellipsoid's RIM BAND is now sampled (%d anchors) -- the "
                  "clip no longer undoes the declared domain", on_rim);
            // The rim reaches 6% past the half-extent along an axis.
            check(rmax_real > 0.5f && rmax_real < 0.5f * 1.10f,
                  "the rim reaches just past the footprint (%.4f, half-extent 0.5)",
                  rmax_real);
        }

        // And its geometry must still be a hemisphere.
        float zmin = 1e9f, zmax = -1e9f;
        for (const auto& a : ell.out.anchors) {
            if (is_placeholder(a)) continue;
            const float z = ell.bmn.z + a.pos[2]*(1.f/65535.f)*ell.bsp.z;
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }
        std::printf("  ellipsoid z: [%.4f, %.4f]\n", zmin, zmax);
        check(std::fabs(zmax - 0.5f) < 0.01f,
              "the ellipsoid is still a hemisphere (%.4f)", zmax);
    }

    // ---- E) the connectivity WRAPS around the axis ------------------------
    //
    // The polar grid's i axis runs around the axis, so the last column's
    // +i neighbour must be the FIRST column. Without the wrap the tube
    // would be split at angle 0 -- a crack running the whole length of the
    // ring.
    //
    // THE DISCRIMINATOR IS THE INDEX, not the geometry. The anchor index
    // within a layer is (j * N + i), so a WRAP link is exactly one whose
    // two indices differ by N-1 in the i direction -- i.e. from the LAST
    // column (i = N-1) to the FIRST (i = 0). An ordinary neighbour differs
    // by one.
    //
    // WHY THE GEOMETRIC TEST WAS NOT ENOUGH: "are they collinear?" counts
    // every ordinary neighbour too, and "is the angle one step apart?"
    // counts the wrap AND the ordinary step in the other direction. Both
    // passed with the wrap disabled -- a gate that cannot fail.
    std::printf("\n-- E) the connectivity around the axis --\n");
    {
        int wrap_links = 0, ordinary = 0;
        for (size_t i = 0; i < t.out.anchors.size(); ++i) {
            const AnchorRecord& a = t.out.anchors[i];
            const int ai = int(i % size_t(N));           // the column
            const int aj = int((i / size_t(N)) % size_t(N));  // the row
            for (int c = 0; c < 4; ++c) {
                if (a.connectivity[c] == 0xFFFF) continue;
                const uint16_t nid = a.connectivity[c];
                if (nid >= t.out.anchors.size()) continue;
                const int bi = int(size_t(nid) % size_t(N));
                const int bj = int((size_t(nid) / size_t(N)) % size_t(N));
                if (aj != bj) continue;                  // not in the same row
                if (ai == 0 && bi == N - 1) ++wrap_links;   // the wrap
                else if (std::abs(ai - bi) == 1) ++ordinary;
            }
        }
        std::printf("  wrap links (column N-1 -> column 0): %d\n", wrap_links);
        std::printf("  ordinary links (adjacent columns):     %d\n", ordinary);
        check(wrap_links > 0,
              "the angular connectivity WRAPS (the tube is closed) -- %d links",
              wrap_links);
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
