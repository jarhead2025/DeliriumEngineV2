// =============================================================================
// validate_rim_overlap - GATE for the closed body's OVERLAPPING rim.
//
// G20's question, as the user framed it (2026-11-10): "we can save compute
// power by having overlapping surfaces where their seams never cross, when
// computing both they average out". So G20 is NOT "eliminate the overlap" --
// it is PROVE the overlap is safe.
//
// WHAT THE OVERLAP IS: a closed body is TWO sheets (an upper one and a
// flipped lower one) that meet at a seam plane. Each sheet's edge is
// EXTENDED past its footprint into a rim band, so the two sheets OVERLAP in
// an annulus on the seam plane instead of meeting at a knife edge where
// their normals would oppose.
//
// WHY THE OVERLAP IS SAFE (the mechanism, and what this gate checks):
//
//   1. THE SEAMS NEVER CROSS. The rim is a flat annulus at z = 0 on BOTH
//      sheets. The flip is in Z; the rim is IN the XY plane. So the flip
//      does NOT mirror the rim: both sheets' rims carry the SAME normal --
//      the footprint's outward gradient. In the overlap annulus the two
//      surfaces agree about which way is "out", so the overlap is shaded as
//      ONE continuous surface, and the averaging the user describes is an
//      average of two surfaces that mean the same thing.
//
//   2. THE OVERLAP IS COINCIDENT. Measured: the nearest-neighbour distance
//      from every upper-rim anchor to the lower sheet's seam anchors is
//      0.000000 -- the two rims are the SAME annulus, not two adjacent
//      ones. That is the strongest form of "they average out": there is
//      nothing to cross.
//
//   3. THE DEPTH TIE IS THE REAL RISK, AND IT IS BOUNDED. Two coplanar
//      surfaces at the same depth are exactly the case the depth-tie
//      diagnostic (fix 33) exists for: if they carried different colours,
//      the atom_min winner would be arbitrary. But both rims carry the
//      SAME colour (the sheet's own), so the tie is between two surfaces
//      that would render IDENTICALLY -- the winner is irrelevant.
//
// This gate measures all three, at two grid resolutions, and fails loudly if
// any of them stops holding.
//
// THE MEASUREMENT THAT PROVED IT (the first version of this gate reported
// "the rims do not overlap"):
//   it classified the seam anchors by the SIGN OF Z -- upper if z >= 0,
//   lower if z < 0. But BOTH sheets' rims are at exactly z = 0, so every
//   rim anchor landed in the "upper" bucket and the lower bucket was
//   empty. The rims never failed to overlap; the CLASSIFICATION did.
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace DeliRium;

namespace {
int checks = 0, failures = 0;
void check(bool cond, const char* msg) {
    ++checks;
    std::printf("  %s %s\n", cond ? "OK  " : "FAIL", msg);
    if (!cond) ++failures;
}
} // namespace

// The CORRECT closure: same origin, negative z scale (the rule
// validate_closed_body pins).
static const char* kClosed =
    R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
        "children":[{"kind":"lower","substance":"ellipsoid",
                     "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                     "scale":[1,1,-1],"origin":[0,0,0]}]})";

int main() {
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(kClosed, tree, why)) {
        std::printf("FAIL: parse: %s\n", why.c_str());
        return 1;
    }

    for (int N : {32, 48}) {
        std::printf("\n-- grid %d --\n", N);
        ModelOutput out;
        if (!ElementSystem::Bake(*tree, N, 1.0f, out, why)) {
            std::printf("FAIL: bake: %s\n", why.c_str());
            return 1;
        }
        auto body = ElementSystem::BakeBody(*tree, N, 1.0f, why);
        if (!body) { std::printf("FAIL: BakeBody: %s\n", why.c_str()); return 1; }
        const Vec3 bmn = body->BBoxMin(), bsp = body->BBoxSpan();

        // The anchor layers: layer 0 = the upper sheet, layer 1 = the lower.
        const size_t per = size_t(N) * N;
        if (out.anchors.size() < 2 * per) {
            std::printf("FAIL: expected %zu anchors, got %zu\n",
                        2 * per, out.anchors.size());
            return 1;
        }

        struct P { float x, y, z; Vec3 n; uint8_t col[3]; };
        auto load = [&](int layer, std::vector<P>& v) {
            for (size_t k = 0; k < per; ++k) {
                const auto& a = out.anchors[size_t(layer) * per + k];
                P p;
                p.x = bmn.x + a.pos[0] * (1.f/65535.f) * bsp.x;
                p.y = bmn.y + a.pos[1] * (1.f/65535.f) * bsp.y;
                p.z = bmn.z + a.pos[2] * (1.f/65535.f) * bsp.z;
                p.n = Vec3{ a.normal[0]*(2.f/255.f)-1.f, a.normal[1]*(2.f/255.f)-1.f,
                            a.normal[2]*(2.f/255.f)-1.f };
                const float l = std::sqrt(p.n.x*p.n.x+p.n.y*p.n.y+p.n.z*p.n.z);
                if (l > 1e-9f) { p.n.x/=l; p.n.y/=l; p.n.z/=l; }
                for (int c = 0; c < 3; ++c) p.col[c] = a.color[c];
                v.push_back(p);
            }
        };
        std::vector<P> L0, L1;
        load(0, L0); load(1, L1);

        // ---- the rim band: the anchors on the seam plane -------------------
        // The rim is at z = 0 EXACTLY (a flat annulus in the XY plane), so
        // the band is |z| <= the quantisation step, NOT a sign test.
        const float ztol = bsp.z * (1.f/65535.f) * 2.f;
        std::vector<P> rim0, rim1;
        for (const P& p : L0) if (std::fabs(p.z) <= ztol) rim0.push_back(p);
        for (const P& p : L1) if (std::fabs(p.z) <= ztol) rim1.push_back(p);
        std::printf("rim anchors: upper %zu, lower %zu (z tolerance %.6f)\n",
                    rim0.size(), rim1.size(), ztol);
        check(!rim0.empty() && !rim1.empty(),
              "BOTH sheets contribute anchors to the seam plane");

        // ---- 1) the rims are COINCIDENT (the seams never cross) ------------
        //
        // The nearest-neighbour distance from every upper-rim anchor to the
        // lower sheet's rim. Zero means the SAME annulus; the grid step
        // would mean two ADJACENT annuli that leave a gap.
        float worst = 0.f;
        for (const P& p : rim0) {
            float d = 1e9f;
            for (const P& q : rim1) {
                const float dx = p.x-q.x, dy = p.y-q.y;
                d = std::min(d, std::sqrt(dx*dx+dy*dy));
            }
            if (d < 1e8f) worst = std::max(worst, d);
        }
        const float grid_step = 2.f * 0.53f / float(N - 1);
        std::printf("nearest-neighbour distance upper-rim -> lower-rim: "
                    "worst %.6f (the grid step is %.6f)\n", worst, grid_step);
        check(worst <= ztol * 2.f,
              "the two rims are COINCIDENT (the seams never cross)");

        // ---- 2) the rims' NORMALS agree (so the overlap averages cleanly) --
        float worst_dot = 1.f;
        int compared = 0;
        for (const P& p : rim0) {
            const P* best = nullptr;
            float bd = 1e9f;
            for (const P& q : rim1) {
                const float dx = p.x-q.x, dy = p.y-q.y;
                const float d = std::sqrt(dx*dx+dy*dy);
                if (d < bd) { bd = d; best = &q; }
            }
            if (!best || bd > ztol * 2.f) continue;
            const float dot = p.n.x*best->n.x + p.n.y*best->n.y + p.n.z*best->n.z;
            worst_dot = std::min(worst_dot, dot);
            ++compared;
        }
        std::printf("the overlapping rims' normal agreement: worst dot %.4f "
                    "(%d pairs)\n", worst_dot, compared);
        check(compared > 0 && worst_dot > 0.9f,
              "the overlapping rims' normals AGREE (one continuous surface)");

        // ---- 3) the tie is benign: the rims carry the SAME colour ----------
        //
        // Two coplanar surfaces at the same depth are the depth-tie case
        // (fix 33): if they carried different colours, the atom_min winner
        // would be arbitrary. They carry the same colour, so the tie is
        // between two surfaces that would render IDENTICALLY.
        int colour_mismatch = 0, colour_pairs = 0;
        for (const P& p : rim0) {
            const P* best = nullptr;
            float bd = 1e9f;
            for (const P& q : rim1) {
                const float dx = p.x-q.x, dy = p.y-q.y;
                const float d = std::sqrt(dx*dx+dy*dy);
                if (d < bd) { bd = d; best = &q; }
            }
            if (!best || bd > ztol * 2.f) continue;
            ++colour_pairs;
            for (int c = 0; c < 3; ++c)
                if (p.col[c] != best->col[c]) ++colour_mismatch;
        }
        std::printf("colour mismatches in the overlap: %d over %d pairs\n",
                    colour_mismatch, colour_pairs);
        check(colour_pairs > 0 && colour_mismatch == 0,
              "the overlapping rims carry the SAME colour (the tie is benign)");

        // ---- 4) the cost: how much of the body is the overlap? -------------
        const size_t overlap = rim0.size();
        std::printf("the overlap is %zu of %zu anchors (%.1f%%)\n",
                    overlap, out.anchors.size(),
                    100.0 * overlap / out.anchors.size());
        // The cost must be BOUNDED -- the rim is a fraction of the footprint,
        // so the overlap is a small fraction of the body.
        check(100.0 * overlap / out.anchors.size() < 40.0,
              "the overlap is a BOUNDED fraction of the body");
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
