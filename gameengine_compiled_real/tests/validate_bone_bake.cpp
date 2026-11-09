// =============================================================================
// validate_bone_bake - GATE for baking the skeleton into the frozen records.
//
// The element tree can now express a SKELETON (validate_bone_tree). This
// gate asserts that the baker actually WRITES it into the engine's frozen
// 64-byte records -- because a vocabulary that parses but does not reach the
// records is just a pretty JSON schema.
//
// WHAT IT ASSERTS (invariants, not snapshots):
//
//   A) A tree with bones emits BoneRecords: one per bone, in the flat list
//      order the parser resolved the parent indices in.
//   B) The BoneRecord's parent index points at the RIGHT bone. This is the
//      load-bearing check: the parser resolves a parent by NAME into an
//      index, and the baker emits records in its own order. If the two
//      orders disagree, the skeleton still validates -- and articulates
//      wrongly.
//   C) The bind pose is in WORLD space: the pivot is the element's head
//      transformed by the element's origin and scale, and the length is the
//      segment's length.
//   D) The authored physical properties survive into the record's
//      reserved bytes (radius, stiffness, mass), so the inertia and field
//      layers can read them without re-parsing the tree.
//   E) ANCHORS ARE BOUND: an anchor on a bone's sheet carries that bone's
//      index in bone_index; an anchor on a boneless sheet keeps the
//      0xFFFFFFFF sentinel.
//   F) CONNECTIVITY respects the articulation: an anchor is connected only
//      to neighbours on the SAME bone. Two anchors on different bones must
//      NOT be connected -- they can move independently, and an edge
//      between them would constrain a free degree of freedom.
//   G) The frozen 64-byte layout still holds, and Validate() still passes.
//   H) A boneless tree still bakes to a body with NO bones and unbound
//      anchors -- the vocabulary is additive.
// =============================================================================
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ReconstructionModel.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
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

// A mechanism: a housing, a shaft through it, a rotor on the shaft.
// The shaft and the rotor have DIFFERENT bones, so their anchors must not
// be connected to each other.
static const char* kMechanism = R"({
  "kind":"assembly","children":[
    {"kind":"housing","substance":"ellipsoid",
     "substance_params":{"sx":0.4,"sy":0.4,"sz":0.4}},
    {"kind":"shaft","bone":{"name":"shaft","head":[0,0,-0.3],"tail":[0,0,0.3],
       "radius":0.05,"mass":0.4,"stiffness":0.8},
     "substance":"ellipsoid","substance_params":{"sx":0.05,"sy":0.05,"sz":0.3}},
    {"kind":"rotor","bone":{"name":"rotor","parent":"shaft",
       "head":[0,0,0],"tail":[0.2,0,0],"radius":0.12,"mass":0.2},
     "substance":"ellipsoid","substance_params":{"sx":0.12,"sy":0.12,"sz":0.05}}]})";

int main() {
    std::printf("== bone bake gate ==\n");

    std::unique_ptr<Element> tree;
    std::string why;
    if (!ParseElementTree(kMechanism, tree, why)) {
        std::printf("FAIL parse: %s\n", why.c_str());
        return 1;
    }

    ModelOutput mo;
    std::string bwhy;
    const bool baked = ElementSystem::Bake(*tree, 16, 1.0f, mo, bwhy);
    CHECK(baked, "A: the mechanism bakes: %s", bwhy.c_str());
    if (!baked) { std::printf("\nBONE-BAKE VALIDATION FAILED\n"); return 1; }

    // --- A) one BoneRecord per bone ---------------------------------------
    CHECK(mo.bones.size() == 2,
          "A: two BoneRecords were emitted", "got %zu", mo.bones.size());

    // --- B) the parent index points at the RIGHT bone ----------------------
    //
    // The parser resolved 'rotor' -> parent 'shaft'. The bone list order is
    // the tree order: shaft (0), rotor (1). So rotor's parent must be 0.
    // The check is made POSITIONAL, not by trusting the parser: find the
    // record whose bind_pos matches the shaft's pivot, and confirm the
    // rotor's parent names THAT index.
    {
        int shaft_idx = -1, rotor_idx = -1;
        for (size_t i = 0; i < mo.bones.size(); ++i) {
            const BoneRecord& b = mo.bones[i];
            // The shaft's pivot is (0,0,-0.3); the rotor's is (0,0,0).
            if (std::fabs(b.bind_pos[2] + 0.3f) < 1e-4f) shaft_idx = int(i);
            if (std::fabs(b.bind_pos[2]) < 1e-4f &&
                std::fabs(b.bind_pos[0]) < 1e-4f) rotor_idx = int(i);
        }
        CHECK(shaft_idx >= 0, "B: the shaft's bone record was found");
        CHECK(rotor_idx >= 0, "B: the rotor's bone record was found");
        if (shaft_idx >= 0 && rotor_idx >= 0) {
            CHECK(mo.bones[shaft_idx].parent == 0xFFFFFFFFu,
                  "B: the shaft is a root (parent = none)",
                  "got %u", mo.bones[shaft_idx].parent);
            CHECK(mo.bones[rotor_idx].parent == uint32_t(shaft_idx),
                  "B: the rotor's parent index points at the SHAFT",
                  "got %u, want %d", mo.bones[rotor_idx].parent, shaft_idx);
        }
    }

    // --- C) the bind pose is in WORLD space ---------------------------------
    {
        // The shaft's element has no origin offset, so its pivot is (0,0,-0.3)
        // and the segment is 0.6 long.
        const BoneRecord* shaft = nullptr;
        for (const auto& b : mo.bones)
            if (std::fabs(b.bind_pos[2] + 0.3f) < 1e-4f) { shaft = &b; break; }
        CHECK(shaft != nullptr, "C: the shaft's record exists");
        if (shaft) {
            CHECK(std::fabs(shaft->bind_pos[0]) < 1e-4f &&
                  std::fabs(shaft->bind_pos[1]) < 1e-4f &&
                  std::fabs(shaft->bind_pos[2] + 0.3f) < 1e-4f,
                  "C: the pivot is the authored head in world space",
                  "got (%f,%f,%f)", shaft->bind_pos[0], shaft->bind_pos[1], shaft->bind_pos[2]);
            CHECK(std::fabs(shaft->length - 0.6f) < 1e-4f,
                  "C: the segment length is the head->tail distance",
                  "got %f, want 0.6", shaft->length);
            // The bind rotation is the identity quaternion.
            CHECK(std::fabs(shaft->bind_rot[3] - 1.f) < 1e-5f &&
                  std::fabs(shaft->bind_rot[0]) < 1e-5f,
                  "C: the bind rotation is the identity quaternion");
        }
    }

    // --- D) the authored physical properties survive -----------------------
    {
        const BoneRecord* shaft = nullptr;
        for (const auto& b : mo.bones)
            if (std::fabs(b.bind_pos[2] + 0.3f) < 1e-4f) { shaft = &b; break; }
        CHECK(shaft != nullptr, "D: the shaft's record exists");
        if (shaft) {
            // reserved[0..3] = the authored MASS as a float (unbounded --
            // a normalised byte would clamp it), [4] = radius (normalised),
            // [5] = stiffness (already 0..1).
            float mass = 0.f;
            std::memcpy(&mass, shaft->reserved, sizeof(float));
            CHECK(std::fabs(mass - 0.4f) < 1e-5f,
                  "D: the authored mass round-trips UNCLAMPED",
                  "got %f, want 0.4", mass);
            CHECK(std::fabs(shaft->reserved[4] / 255.f - 0.05f) < 0.01f,
                  "D: the radius round-trips", "got %f", shaft->reserved[4] / 255.f);
            CHECK(std::fabs(shaft->reserved[5] / 255.f - 0.8f) < 0.01f,
                  "D: the stiffness round-trips",
                  "got %f", shaft->reserved[5] / 255.f);
        }
    }

    // --- E) anchors are bound to their bone --------------------------------
    {
        // The shaft's sheet is the SECOND leaf, the rotor's the THIRD. Their
        // anchors must carry different bone indices.
        std::set<uint32_t> bound;
        int unbound = 0;
        for (const auto& a : mo.anchors) {
            if (a.bone_index == 0xFFFFFFFFu) ++unbound;
            else bound.insert(a.bone_index);
        }
        CHECK(bound.size() == 2,
              "E: anchors are bound to TWO different bones", "got %zu", bound.size());
        // The housing (the FIRST leaf) has NO bone, so ITS anchors must be
        // unbound. The invariant is therefore not "no unbound anchors" but
        // "the unbound ones are exactly the housing's": the housing is one
        // leaf, so it owns exactly N*N anchors.
        const int N = 16;
        CHECK(unbound == N * N,
              "E: the unbound anchors are exactly the boneless housing's",
              "got %d, want %d", unbound, N * N);
    }

    // --- F) connectivity respects the articulation -------------------------
    {
        // Every connectivity edge must join two anchors on the SAME bone.
        int cross = 0, edges = 0;
        for (size_t i = 0; i < mo.anchors.size(); ++i)
            for (int k = 0; k < 4; ++k) {
                const uint16_t c = mo.anchors[i].connectivity[k];
                if (c == 0xFFFF) continue;
                ++edges;
                if (c >= mo.anchors.size()) { ++cross; continue; }
                if (mo.anchors[c].bone_index != mo.anchors[i].bone_index) ++cross;
            }
        CHECK(edges > 0, "F: the anchors have connectivity edges", "got %d", edges);
        CHECK(cross == 0,
              "F: NO connectivity edge crosses a bone boundary",
              "got %d cross-bone edges of %d", cross, edges);
        // The boneless housing's anchors are all on the same "bone" (none),
        // so they may connect to each other -- that is fine, they are one
        // rigid piece.
    }

    // --- G) the frozen layout still holds ----------------------------------
    {
        CHECK(sizeof(AnchorRecord) == 64, "G: AnchorRecord is still 64 bytes");
        CHECK(sizeof(BoneRecord) == 64, "G: BoneRecord is still 64 bytes");
        std::string vwhy;
        CHECK(ReconstructionModel::Validate(mo, vwhy),
              "G: the records with bones still validate: %s", vwhy.c_str());
    }

    // --- H) a boneless tree is unchanged -----------------------------------
    {
        const char* plain = R"({
          "kind":"body","substance":"ellipsoid",
          "substance_params":{"sx":0.4,"sy":0.4,"sz":0.5}})";
        std::unique_ptr<Element> t2;
        std::string w2;
        CHECK(ParseElementTree(plain, t2, w2), "H: the boneless tree parses");
        if (t2) {
            ModelOutput mo2;
            std::string bw2;
            CHECK(ElementSystem::Bake(*t2, 16, 1.0f, mo2, bw2),
                  "H: the boneless tree bakes: %s", bw2.c_str());
            CHECK(mo2.bones.empty(), "H: a boneless tree emits NO bones",
                  "got %zu", mo2.bones.size());
            int unbound2 = 0;
            for (const auto& a : mo2.anchors)
                if (a.bone_index == 0xFFFFFFFFu) ++unbound2;
            CHECK(unbound2 == (int)mo2.anchors.size(),
                  "H: every anchor of a boneless tree stays unbound",
                  "got %d of %zu", unbound2, mo2.anchors.size());
        }
    }

    std::printf("\n%s (%d checks, %d failures)\n",
                g_fail ? "BONE-BAKE VALIDATION FAILED" : "bone-bake validation passed",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
