// =============================================================================
// validate_coarseness - GATE for the coarseness budget.
//
// WHY (Josias, 2026-11-08): "make them complete in representation, denoting
// the coarseness for the sake of speedy computation". The model should
// choose an ABSTRACTION LEVEL, and the level should derive both budgets the
// engine spends -- surface resolution AND skeleton density -- from ONE
// choice.
//
// WHAT IT ASSERTS (invariants, not snapshots):
//
//   A) A tree states a coarseness level ("lod", 0..3), and an out-of-range
//      level is REJECTED rather than clamped (a clamped level silently
//      ignores the model's intent).
//   B) The level DERIVES the surface resolution: a finer level bakes more
//      patches. The relationship is monotonic -- that is the whole point.
//   C) A coarser level keeps FEWER bones. A joint nobody can see is a joint
//      nobody can afford to solve.
//   D) A tree that states no level gets the default (medium), and an
//      explicit grid_n still overrides the tree (a caller that knows its
//      own budget should not have to argue with the model).
//   E) The SYSTEM PROMPT teaches the levels -- otherwise the model never
//      uses them.
//   F) Coarseness does not change the SHAPE: a coarse bake must still
//      reproduce the field's extent within the grid's own tolerance. A
//      cheaper bake that is a different object is not a cheaper bake.
// =============================================================================
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"

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

// A body with two bones, so the skeleton's density can be compared.
static const char* kTree(int lod) {
    static std::string s;
    s = "{\"kind\":\"assembly\"";
    if (lod >= 0) s += ",\"lod\":" + std::to_string(lod);
    s += R"(,"children":[
      {"kind":"housing","substance":"ellipsoid",
       "substance_params":{"sx":0.4,"sy":0.4,"sz":0.4}},
      {"kind":"shaft","bone":{"name":"shaft","head":[0,0,-0.3],"tail":[0,0,0.3],
         "radius":0.05,"mass":0.4},
       "substance":"ellipsoid","substance_params":{"sx":0.05,"sy":0.05,"sz":0.3}},
      {"kind":"rotor","bone":{"name":"rotor","parent":"shaft",
         "head":[0,0,0],"tail":[0.2,0,0],"radius":0.12,"mass":0.2},
       "substance":"ellipsoid","substance_params":{"sx":0.12,"sy":0.12,"sz":0.05}}]})";
    return s.c_str();
}

// Bake at the level's own resolution (grid_n < 0 -> the level decides).
static bool BakeAtLod(const char* json, ModelOutput& out, std::string& why) {
    std::unique_ptr<Element> tree;
    if (!ParseElementTree(json, tree, why)) return false;
    return ElementSystem::BakeLod(*tree, -1, 1.0f, out, why);
}

int main() {
    std::printf("== coarseness budget gate ==\n");

    // --- A) the level parses, and an out-of-range level is REJECTED --------
    {
        std::unique_ptr<Element> tree;
        std::string why;
        for (int lod = 0; lod <= 3; ++lod) {
            CHECK(ParseElementTree(kTree(lod), tree, why),
                  "A: lod=%d parses: %s", lod, why.c_str());
            if (tree) {
                CHECK(tree->lod == lod, "A: lod=%d is kept on the tree", lod);
                CHECK(ElementSystem::LodOf(*tree) == lod,
                      "A: LodOf reports lod=%d", lod);
            }
        }
        // A tree with no level uses the default.
        CHECK(ParseElementTree(kTree(-1), tree, why), "A: a tree with no lod parses");
        if (tree) {
            CHECK(tree->lod < 0, "A: an unstated lod stays unstated");
            CHECK(ElementSystem::LodOf(*tree) == 1,
                  "A: an unstated lod defaults to medium (1)");
        }
        // Out of range: REJECTED, not clamped.
        for (int bad : {-1, 4, 99}) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                R"({"kind":"body","lod":%d,"substance":"plane"})", bad);
            CHECK(!ParseElementTree(buf, tree, why),
                  "A: lod=%d is REJECTED (not clamped)", bad);
            CHECK(why.find("lod") != std::string::npos,
                  "A: the rejection mentions lod", "why=%s", why.c_str());
        }
        // A non-numeric lod is rejected too.
        CHECK(!ParseElementTree(R"({"kind":"body","lod":"fine","substance":"plane"})",
                                tree, why),
              "A: a non-numeric lod is REJECTED");
    }

    // --- B) the level derives the resolution, monotonically -----------------
    {
        // The loop runs FINE -> COARSE (lod 0 -> 3), so each level must have
        // FEWER patches than the one before it. A finer level spends more.
        int prev_patches = -1;
        for (int lod = 0; lod <= 3; ++lod) {
            ModelOutput mo;
            std::string why;
            CHECK(BakeAtLod(kTree(lod), mo, why),
                  "B: lod=%d bakes at its own resolution: %s", lod, why.c_str());
            if (mo.surfaces.empty()) continue;
            const int patches = int(mo.surfaces.size());
            if (prev_patches >= 0)
                CHECK(patches < prev_patches,
                      "B: lod=%d has FEWER patches than the finer level",
                      lod, "got %d, previous %d", patches, prev_patches);
            else
                CHECK(patches > 0, "B: lod=%d produced patches", lod);
            prev_patches = patches;
            // The resolution must be the one the level names: the anchor
            // grid is sheets * n^2, so at least n^2 anchors for one sheet.
            const int grid = Element::GridForLod(lod);
            CHECK(mo.anchors.size() >= size_t(grid) * size_t(grid),
                  "B: lod=%d's anchor grid is at least the level's resolution",
                  lod, "got %zu, want >= %d", mo.anchors.size(), grid * grid);
        }
    }

    // --- C) a coarser level keeps fewer bones ------------------------------
    {
        // The tree has two bones. At the finest level both survive; at the
        // coarsest, the tree's budget says keep ~40% -- so at least one is
        // dropped. The invariant is a COUNT, not a snapshot.
        int bones_fine = 0, bones_coarse = 0;
        for (int lod : {0, 3}) {
            ModelOutput mo;
            std::string why;
            if (!BakeAtLod(kTree(lod), mo, why)) continue;
            const int n = int(mo.bones.size());
            if (lod == 0) bones_fine = n; else bones_coarse = n;
        }
        CHECK(bones_fine == 2, "C: the finest level keeps both bones",
              "got %d", bones_fine);
        CHECK(bones_coarse < bones_fine,
              "C: the coarsest level keeps FEWER bones",
              "coarse %d vs fine %d", bones_coarse, bones_fine);
    }

    // --- D) an explicit grid_n overrides the tree --------------------------
    {
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(ParseElementTree(kTree(0), tree, why), "D: the fine tree parses");
        if (tree) {
            // lod 0 wants grid 48. An explicit grid of 8 must win.
            ModelOutput mo;
            CHECK(ElementSystem::BakeLod(*tree, 8, 1.0f, mo, why),
                  "D: an explicit grid_n bakes: %s", why.c_str());
            // 3 sheets at grid 8 -> 3 * 64 = 192 anchors (minus out-of-support
            // ones are still written). The point is it is FAR below the
            // lod-0 resolution of 48.
            CHECK(mo.anchors.size() < size_t(48) * size_t(48),
                  "D: the explicit grid_n overrides the tree's level",
                  "got %zu anchors", mo.anchors.size());
        }
    }

    // --- E) the system prompt teaches the levels ---------------------------
    {
        const std::string prompt = ElementTreeSystemPrompt();
        CHECK(prompt.find("\"lod\"") != std::string::npos,
              "E: the system prompt mentions lod");
        CHECK(prompt.find("coarseness") != std::string::npos ||
              prompt.find("COARSENESS") != std::string::npos,
              "E: the system prompt explains coarseness");
        CHECK(prompt.find("sketch") != std::string::npos,
              "E: the system prompt names the levels");
    }

    // --- F) coarseness does not change the SHAPE ---------------------------
    {
        // A coarse bake must still reproduce the field's extent. A cheaper
        // bake that is a DIFFERENT object is not a cheaper bake.
        for (int lod : {0, 2, 3}) {
            std::unique_ptr<Element> tree;
            std::string why;
            if (!ParseElementTree(kTree(lod), tree, why)) continue;
            ModelOutput mo;
            if (!ElementSystem::BakeLod(*tree, -1, 1.0f, mo, why)) continue;
            // The housing's sheet spans the canvas: its anchors must reach
            // roughly the full extent in x and y.
            float xlo = 1e30f, xhi = -1e30f, ylo = 1e30f, yhi = -1e30f;
            const Vec3 mn = Vec3{0,0,0}, span = Vec3{0,0,0};
            (void)mn; (void)span;
            // The anchors are quantized over the bake's bbox, so compare the
            // QUANTIZED extent: it must span most of the box in x and y.
            uint16_t qxlo = 65535, qxhi = 0, qylo = 65535, qyhi = 0;
            for (const auto& a : mo.anchors) {
                if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767) continue;
                qxlo = std::min(qxlo, a.pos[0]); qxhi = std::max(qxhi, a.pos[0]);
                qylo = std::min(qylo, a.pos[1]); qyhi = std::max(qyhi, a.pos[1]);
            }
            if (qxhi <= qxlo) continue;
            const float fx = float(qxhi - qxlo) / 65535.f;
            const float fy = float(qyhi - qylo) / 65535.f;
            CHECK(fx > 0.5f && fy > 0.5f,
                  "F: lod=%d still spans the object in x and y",
                  lod, "fx=%.3f fy=%.3f", fx, fy);
        }
    }

    std::printf("\n%s (%d checks, %d failures)\n",
                g_fail ? "COARSENESS VALIDATION FAILED" : "coarseness validation passed",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
