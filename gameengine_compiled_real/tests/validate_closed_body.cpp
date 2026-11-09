// =============================================================================
// validate_closed_body - GATE for closed-body composition.
//
// A PSF body is a graph z = f(x,y) over a patch grid. A graph cannot express
// a CLOSED surface: an ellipsoid substance describes the UPPER hemisphere
// only. A closed body must therefore be composed of SEVERAL sheets that meet
// at a seam -- and the rule for making them meet is easy to get wrong.
//
// This gate pins that rule, which is MEASURED rather than assumed:
//
//   world.z = origin.z + local.z * scale.z
//   with local.z spanning [0, sz] for an ellipsoid:
//      scale.z = +1  ->  the sheet spans [origin.z, origin.z + sz]   (UP)
//      scale.z = -1  ->  the sheet spans [origin.z - sz, origin.z]   (DOWN)
//
// A negative z scale ALREADY flips the sheet downward from the seam plane
// it shares with its parent. So the lower half of a closed sphere keeps the
// SAME origin as the upper half and only flips the scale.
//
// The gate asserts:
//   - the CORRECT rule (same origin, negative scale) spans exactly
//     [-sz, +sz] and is centred on the origin
//   - the lower sheet's normals point DOWN (the body faces outward)
//   - an OFFSET origin leaves a GAP (the body does not close)
//   - a POSITIVE scale closes the geometry but turns the lower sheet
//     INSIDE-OUT (it renders as a hollow shell)
//   - the baker writes ONE anchor grid per sheet, not one per sheet per
//     sheet -- i.e. anchors == sheets * n^2. (A missing closing brace once
//     made the anchor loop nest inside the sampling loop, producing 2x the
//     anchors for a 2-sheet tree; the surface count hid it.)
//   - every surface references an anchor inside the blob
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/AnchorPool.h"
#include "delirium/core/Types.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::AI;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

namespace {

// Bake a two-sheet sphere and report what the bake produced.
struct Baked {
    bool ok = false;
    std::unique_ptr<PSFBody> body;
    float z_lo = 0.f, z_hi = 0.f, span = 0.f, centre = 0.f;
    int lower_down = 0, lower_up = 0;   // lower-sheet normal orientation
    size_t anchors = 0, surfaces = 0;
    std::string why;
};

Baked BakeTree(const char* json, int grid_n = 48) {
    Baked b;
    std::unique_ptr<Element> tree;
    if (!ParseElementTree(json, tree, b.why)) return b;
    ModelOutput mo;
    if (!ElementSystem::Bake(*tree, grid_n, 1.0f, mo, b.why)) return b;
    b.anchors = mo.anchors.size();
    b.surfaces = mo.surfaces.size();
    b.body = ElementSystem::BakeBody(*tree, grid_n, 1.0f, b.why);
    if (!b.body) return b;

    // Measure the ANCHORS (the bake's ground truth), dequantized with the
    // body's own bbox -- exactly what the renderer reads.
    const AnchorPool& pool = b.body->AnchorsConst();
    const std::vector<AnchorRecord>* blob = pool.Blob();
    if (!blob) return b;
    const Vec3 mn = pool.BBoxMin(), sp = pool.BBoxSpan();
    bool first = true;
    for (const AnchorRecord& a : *blob) {
        // out-of-support anchors are parked at the bbox centre; they carry no
        // surface, so they must not widen the extent.
        if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767) continue;
        const float z = mn.z + sp.z * (a.pos[2] * (1.0f / 65535.0f));
        if (first) { b.z_lo = b.z_hi = z; first = false; }
        else { b.z_lo = std::min(b.z_lo, z); b.z_hi = std::max(b.z_hi, z); }
        const float nz = a.normal[2] * (2.0f / 255.0f) - 1.0f;
        if (z < -0.05f && nz > 0.05f)  ++b.lower_up;
        if (z < -0.05f && nz < -0.05f) ++b.lower_down;
    }
    b.span = b.z_hi - b.z_lo;
    b.centre = 0.5f * (b.z_lo + b.z_hi);
    b.ok = true;
    return b;
}

// The CORRECT closure: same origin, negative z scale.
const char* kCorrect =
    R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
        "children":[{"kind":"lower","substance":"ellipsoid",
                     "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                     "scale":[1,1,-1],"origin":[0,0,0]}]})";

// The OFFSET variant: a gap between the halves.
const char* kOffset =
    R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
        "children":[{"kind":"lower","substance":"ellipsoid",
                     "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                     "scale":[1,1,-1],"origin":[0,0,-0.5]}]})";

// The UNFLIPPED variant: closes geometrically, but inside-out.
const char* kUnflipped =
    R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
        "children":[{"kind":"lower","substance":"ellipsoid",
                     "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                     "scale":[1,1,1],"origin":[0,0,-0.5]}]})";

} // namespace

int main() {
    std::printf("== closed-body composition gate ==\n");

    // --- A) the CORRECT rule closes the body ----------------------------
    {
        const Baked b = BakeTree(kCorrect);
        CHECK(b.ok, "A: the correct rule bakes: %s", b.why.c_str());
        if (b.ok) {
            std::printf("  A: anchors=%zu surfaces=%zu z=[%+.4f,%+.4f] span=%.4f centre=%+.4f\n",
                        b.anchors, b.surfaces, b.z_lo, b.z_hi, b.span, b.centre);
            CHECK(std::fabs(b.span - 1.0f) < 0.06f,
                  "A: a closed radius-0.5 sphere spans 1.0 (got %.4f)", b.span);
            CHECK(std::fabs(b.centre) < 0.06f,
                  "A: a closed sphere is centred on the origin (got %+.4f)", b.centre);
            // The lower sheet must face DOWN, or the body is inside-out.
            CHECK(b.lower_up == 0,
                  "A: no lower-sheet normal faces up (up=%d)", b.lower_up);
            CHECK(b.lower_down > 0,
                  "A: the lower sheet's normals face down (down=%d)", b.lower_down);
        }
    }

    // --- B) an OFFSET origin leaves a gap -------------------------------
    {
        const Baked b = BakeTree(kOffset);
        CHECK(b.ok, "B: the offset variant bakes: %s", b.why.c_str());
        if (b.ok) {
            std::printf("  B: z=[%+.4f,%+.4f] span=%.4f\n", b.z_lo, b.z_hi, b.span);
            CHECK(b.span > 1.2f,
                  "B: an offset origin opens a gap (span %.4f, expected > 1.2)", b.span);
        }
    }

    // --- C) an UNFLIPPED lower sheet is inside-out -----------------------
    {
        const Baked b = BakeTree(kUnflipped);
        CHECK(b.ok, "C: the unflipped variant bakes: %s", b.why.c_str());
        if (b.ok) {
            std::printf("  C: z=[%+.4f,%+.4f] span=%.4f  lower down=%d up=%d\n",
                        b.z_lo, b.z_hi, b.span, b.lower_down, b.lower_up);
            CHECK(b.lower_down == 0,
                  "C: an unflipped lower sheet has NO downward normal (down=%d)",
                  b.lower_down);
            CHECK(b.lower_up > 0,
                  "C: an unflipped lower sheet faces UP (up=%d)", b.lower_up);
        }
    }

    // --- D) ONE anchor grid per sheet (not one per sheet per sheet) ------
    // This is the invariant that a missing closing brace broke: the anchor
    // loop was nested inside the per-sheet sampling loop and ran once per
    // sheet, producing sheets^2 * n^2 anchors instead of sheets * n^2.
    for (int grid_n : {16, 32, 48}) {
        const Baked b = BakeTree(kCorrect, grid_n);
        CHECK(b.ok, "D: bakes at grid_n=%d: %s", grid_n, b.why.c_str());
        if (!b.ok) continue;
        const size_t sheets = 2;
        const size_t expect_anchors = sheets * size_t(grid_n) * size_t(grid_n);
        std::printf("  D: grid_n=%2d anchors=%zu (expect %zu)\n",
                    grid_n, b.anchors, expect_anchors);
        CHECK(b.anchors == expect_anchors,
              "D: anchors == sheets * n^2 at grid_n=%d (got %zu, want %zu)",
              grid_n, b.anchors, expect_anchors);
    }

    // --- E) every surface references an anchor inside the blob ----------
    {
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(ParseElementTree(kCorrect, tree, why), "E: parse: %s", why.c_str());
        if (tree) {
            ModelOutput mo;
            CHECK(ElementSystem::Bake(*tree, 48, 1.0f, mo, why), "E: bake: %s", why.c_str());
            unsigned max_ref = 0;
            for (const auto& s : mo.surfaces)
                for (int k = 0; k < 4; ++k)
                    max_ref = std::max(max_ref, (unsigned)s.anchor_idx[k]);
            std::printf("  E: surface anchor_idx max=%u, blob=%zu\n", max_ref, mo.anchors.size());
            CHECK(max_ref < mo.anchors.size(),
                  "E: every surface references an anchor inside the blob");
            // The referenced range must cover the blob, and the only
            // UNREFERENCED anchors must be the out-of-support ones.
            //
            // The baker writes a well-formed anchor record for EVERY grid
            // cell, including the ones whose substance has no surface there
            // (outside a bounded domain). Those are parked at the bbox
            // centre with pos = 32767 and carry no surface -- they exist so
            // the anchor index space stays dense. They are NOT orphans in
            // the "wasted upload" sense; they are the documented fallback.
            //
            // What WOULD be a real defect is a referenced index outside the
            // blob (checked above), or a non-fallback anchor that no
            // surface uses.
            size_t referenced = 0, fallback = 0, orphan = 0;
            std::vector<char> seen(mo.anchors.size(), 0);
            for (const auto& s : mo.surfaces)
                for (int k = 0; k < 4; ++k) {
                    const unsigned a = s.anchor_idx[k];
                    if (a < seen.size() && !seen[a]) { seen[a] = 1; ++referenced; }
                }
            for (size_t i = 0; i < mo.anchors.size(); ++i) {
                if (seen[i]) continue;
                const AnchorRecord& a = mo.anchors[i];
                if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767)
                    ++fallback;      // out-of-support placeholder
                else
                    ++orphan;        // a REAL anchor no surface uses -- a defect
            }
            std::printf("  E: referenced=%zu fallback=%zu orphan=%zu of %zu\n",
                        referenced, fallback, orphan, mo.anchors.size());
            CHECK(orphan == 0,
                  "E: no orphan anchors (%d referenced-but-unused non-fallback anchors)",
                  orphan);
            CHECK(referenced + fallback == mo.anchors.size(),
                  "E: every anchor is either referenced or an out-of-support fallback");
        }
    }

    std::printf("\n%s (%d checks, %d failures)\n",
                g_fail ? "CLOSED-BODY VALIDATION FAILED" : "closed-body validation passed",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
