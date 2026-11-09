// =============================================================================
// closed_body_probe - WHICH origin/scale actually CLOSES a body?
//
// The doc-learner system prompt tells the model:
//
//   "a sphere = an ellipsoid + an ellipsoid below it
//    (origin z = -0.5, scale z negative to flip it)"
//
// and its worked example uses "origin":[0,0,0.4].
//
// BOTH cannot be right. The sheet's world height is
//
//     world.z = origin.z + local.z * scale.z
//
// with local.z = sz*sqrt(1-q) for an ellipsoid, so local.z spans [0, sz]:
//
//   scale.z = +1  ->  sheet spans [origin.z, origin.z + sz]   (UP)
//   scale.z = -1  ->  sheet spans [origin.z - sz, origin.z]   (DOWN)
//
// A negative scale ALREADY flips the sheet downward from the seam plane it
// shares with the parent. Adding an origin offset on top shifts the lower
// sheet OFF the seam, so the body does not close -- it opens a gap (or
// overlaps, if the offset goes the other way).
//
// The correct rule is therefore:
//   origin = [0,0,0], scale = [1,1,-1]  ->  upper [0, sz], lower [-sz, 0]
//
// THIS PROBE MEASURES THE ANCHORS -- the quantized positions the baker
// actually wrote, dequantized with the body's own bbox. The anchors are the
// ground truth of the bake. The patch evaluation is reported separately,
// because the LSQ fit on top of the anchors is a DIFFERENT concern (and it
// adds the seam/stitch displacement), so conflating the two hides which one
// is wrong.
//
// The closure criterion is the SEAM, not the extent:
//   * the upper sheet's crown is at z = +sz,
//   * the lower sheet's crown is at z = -sz,
//   * the two sheets must MEET at z = 0 (the seam),
// so the union must span exactly [-sz, +sz] and be centred on z = 0.
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/AnchorPool.h"
#include "delirium/core/Types.h"   // AnchorRecord

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::AI;

struct Candidate {
    const char* label;
    const char* json;
    bool expect_closed;
};

// A sphere of radius 0.5 = upper ellipsoid + a second sheet below it.
static const Candidate kCandidates[] = {
    { "A) PROMPT TEXT: origin z=-0.5, scale z=-1",
      R"({"kind":"body","substance":"ellipsoid",
          "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
          "children":[{"kind":"lower","substance":"ellipsoid",
                       "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                       "scale":[1,1,-1],"origin":[0,0,-0.5]}]})" , false },

    { "B) PROMPT EXAMPLE: origin z=+0.4, scale z=-1",
      R"({"kind":"body","substance":"ellipsoid",
          "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
          "children":[{"kind":"lower","substance":"ellipsoid",
                       "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                       "scale":[1,1,-1],"origin":[0,0,0.4]}]})" , false },

    { "C) SAME ORIGIN, NO FLIP: origin z=-0.5, scale z=+1",
      R"({"kind":"body","substance":"ellipsoid",
          "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
          "children":[{"kind":"lower","substance":"ellipsoid",
                       "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                       "scale":[1,1,1],"origin":[0,0,-0.5]}]})" , false },
    { "D) MEASURED RULE: origin z=0, scale z=-1",
      R"({"kind":"body","substance":"ellipsoid",
          "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
          "children":[{"kind":"lower","substance":"ellipsoid",
                       "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                       "scale":[1,1,-1],"origin":[0,0,0]}]})" , true },
};

int main() {
    std::printf("=== CLOSED BODY PROBE: which origin/scale closes a sphere? ===\n");
    std::printf("(radius 0.5 -> a closed body must span exactly [-0.5, +0.5]\n");
    std::printf(" and its centre must be at z = 0)\n");

    int failures = 0;
    for (const Candidate& c : kCandidates) {
        std::unique_ptr<Element> tree;
        std::string why;
        if (!ParseElementTree(c.json, tree, why)) {
            std::printf("\n%s\n  parse failed: %s\n", c.label, why.c_str());
            ++failures;
            continue;
        }
        std::string bw;
        auto body = ElementSystem::BakeBody(*tree, 48, 1.0f, bw);
        if (!body) {
            std::printf("\n%s\n  bake FAILED: %s\n", c.label, bw.c_str());
            ++failures;
            continue;
        }

        // --- WHERE DO THE RECORDS COME FROM? ---------------------------------
        // Bake() and BakeBody() are two steps. If the record counts disagree,
        // something is being appended twice (a stale/duplicated write), and
        // every extent measured below is polluted by the duplicate. So the
        // raw ModelOutput is compared against the materialized body.
        {
            ModelOutput mo;
            std::string bwhy;
            const bool ok = ElementSystem::Bake(*tree, 48, 1.0f, mo, bwhy);
            std::printf("\n%s\n", c.label);
            std::printf("  Bake()      anchors=%zu surfaces=%zu%s\n",
                        mo.anchors.size(), mo.surfaces.size(),
                        ok ? "" : (" FAILED: " + bwhy).c_str());
            std::printf("  BakeBody()  anchors=%zu patches=%zu  seams=%zu\n",
                        body->AnchorsConst().Count(), body->Patches().size(),
                        body->Seams().SeamCount());
            const int per48 = 48 * 48;
            std::printf("  => %zu blocks of %d anchors (nsheets=%zu)\n",
                        mo.anchors.size() / (size_t)per48, per48,
                        mo.anchors.size() / (size_t)per48);
        }

        // --- ANCHOR extent (the bake's ground truth) ------------------------
        const AnchorPool& pool = body->AnchorsConst();
        const std::vector<AnchorRecord>* blob = pool.Blob();
        const Vec3 mn = pool.BBoxMin(), sp = pool.BBoxSpan();
        float alo = 1e30f, ahi = -1e30f;
        int nvalid = 0;
        if (blob) {
            for (const AnchorRecord& a : *blob) {
                // out-of-support anchors are parked at the bbox centre with
                // pos = 32767/65535 = 0.5; they carry no surface, so they are
                // excluded from the extent (they would only dilute it).
                if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767)
                    continue;
                const float z = mn.z + sp.z * (a.pos[2] * (1.0f / 65535.0f));
                alo = std::min(alo, z);
                ahi = std::max(ahi, z);
                ++nvalid;
            }
        }
        // --- PATCH extent (what the renderer draws) -------------------------
        float plo = 1e30f, phi = -1e30f;
        for (const auto& p : body->Patches()) {
            for (int a = 0; a <= 4; ++a)
                for (int b = 0; b <= 4; ++b) {
                    const Vec3 q = p.Evaluate(a / 4.f, b / 4.f);
                    plo = std::min(plo, q.z);
                    phi = std::max(phi, q.z);
                }
        }

        const float aspan = ahi - alo, ace = 0.5f * (alo + ahi);
        const float pspan = phi - plo, pce = 0.5f * (plo + phi);
        const bool geom_closed = std::fabs(aspan - 1.0f) < 0.06f &&
                                 std::fabs(ace) < 0.06f;

        // --- NORMAL ORIENTATION: a closed body must face OUTWARD everywhere.
        // For a sphere centred at the origin, the outward normal at a point
        // p is p/|p|. The lower sheet's normals must point DOWN (negative z)
        // at its crown, which only happens if the sheet was FLIPPED.
        // Candidate C closes geometrically but without a flip, so its lower
        // sheet still faces UP -- the body is inside-out.
        //
        // Measured on the anchor records (the bake's own output), not on the
        // patch evaluation, so the seam/stitch displacement cannot skew it.
        //
        // The anchors are laid out as one contiguous BLOCK PER SHEET
        // (sheet s owns [s*N*N, (s+1)*N*N)), so the per-sheet breakdown
        // below separates "the lower sheet's own normals are wrong" from
        // "my region test is catching the wrong sheet".
        int n_up = 0, n_down = 0;
        const int N = 48, per = N * N;
        if (blob) {
            const int nsheets = (int)(blob->size() / (size_t)per);
            for (int s = 0; s < nsheets; ++s) {
                float slo = 1e30f, shi = -1e30f;
                int sup = 0, up = 0, down = 0, flat = 0;
                for (int k = 0; k < per; ++k) {
                    const AnchorRecord& a = (*blob)[size_t(s) * per + k];
                    if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767)
                        continue;
                    const float z = mn.z + sp.z * (a.pos[2] * (1.0f / 65535.0f));
                    const float nz = a.normal[2] * (2.0f / 255.0f) - 1.0f;
                    slo = std::min(slo, z); shi = std::max(shi, z);
                    ++sup;
                    if (nz > 0.05f) ++up;
                    else if (nz < -0.05f) ++down;
                    else ++flat;
                }
                std::printf("    sheet %d: z=[%+.4f,%+.4f] in-support=%d "
                            "normals up=%d down=%d flat=%d\n",
                            s, slo, shi, sup, up, down, flat);
            }
            for (const AnchorRecord& a : *blob) {
                if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767)
                    continue;
                const float z = mn.z + sp.z * (a.pos[2] * (1.0f / 65535.0f));
                const float nz = a.normal[2] * (2.0f / 255.0f) - 1.0f;
                if (z < -0.05f && nz > 0.05f)  ++n_up;    // lower sheet facing up
                if (z < -0.05f && nz < -0.05f) ++n_down;  // lower sheet facing down
            }
        }
        // A flipped lower sheet has every lower normal facing down.
        const bool outward = (n_up == 0 && n_down > 0);

        // CLOSED means closed AND outward-facing. A body that closes
        // geometrically but is inside-out renders as an inside shell (the
        // back faces cull away), which is NOT a closed body.
        const bool closed = geom_closed && outward;

        std::printf("\n%s\n", c.label);
        std::printf("  anchors=%zu (in-support %d)  patches=%zu\n",
                    blob ? blob->size() : 0, nvalid, body->Patches().size());
        std::printf("  ANCHOR z=[%+.4f, %+.4f]  span=%.4f  centre=%+.4f\n",
                    alo, ahi, aspan, ace);
        std::printf("  PATCH  z=[%+.4f, %+.4f]  span=%.4f  centre=%+.4f\n",
                    plo, phi, pspan, pce);
        std::printf("  geometry: %s   lower-sheet normals: down=%d up=%d -> %s\n",
                    geom_closed ? "closed" : "not closed",
                    n_down, n_up, outward ? "OUTWARD (correct)" : "INSIDE-OUT");
        std::printf("  %s\n", closed ? "OK   -- CLOSED (full diameter, centred, outward)"
                                     : "FAIL -- not a closed outward body");

        // The probe's job: confirm the expectation, not just report a number.
        if (closed != c.expect_closed) {
            std::printf("  FAIL -- expectation mismatch (expected %s)\n",
                        c.expect_closed ? "CLOSED" : "NOT closed");
            ++failures;
        }
        if (c.expect_closed && !outward) {
            std::printf("  FAIL -- closed but the lower sheet is inside-out\n");
            ++failures;
        }
    }

    std::printf("\n%s\n", failures == 0
        ? "PROBE PASSED -- the measured rule is origin z=0, scale z=-1"
        : "PROBE FAILED -- the rule is not what the probe assumed");
    return failures == 0 ? 0 : 1;
}
