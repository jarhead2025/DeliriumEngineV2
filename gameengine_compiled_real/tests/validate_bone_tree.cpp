// =============================================================================
// validate_bone_tree - GATE for the bone/articulation vocabulary.
//
// WHY THIS EXISTS (Josias, 2026-11-08): "object reconstruction is encoded
// based on line segment (bones) or points in space (articulations)". The
// engine's frozen 64-byte records already carry a skeleton (BoneRecord with
// parent/bind_pos/bind_rot/length, and AnchorRecord::bone_index) but the
// element tree could not EXPRESS one: a tree said "four overlapping
// ellipsoids" and nothing about which part rotates relative to which.
//
// This gate pins the vocabulary's INVARIANTS (not snapshots):
//
//   A) a tree with bones parses, and each bone keeps its name, segment,
//      radius, mass, stiffness and named joints;
//   B) a bone's parent is given by NAME and resolves to the right INDEX;
//   C) an UNKNOWN parent name is REJECTED -- silently re-rooting a bone
//      whose parent was misspelled produces a skeleton that looks fine and
//      articulates wrongly;
//   D) two bones with the SAME name are REJECTED (an ambiguous parent
//      reference is a coin flip);
//   E) a bone that is its own ancestor is REJECTED (the hierarchy would be
//      cyclic and every traversal would recurse forever);
//   F) a bone needs both a head and a tail -- a segment with one end is
//      not a segment;
//   G) negative mass/radius and an out-of-range stiffness are REJECTED;
//   H) a tree WITHOUT bones still parses and still bakes (the vocabulary
//      is additive -- a simple object must not have to declare a skeleton);
//   I) a bone with a substance and a bone without one are BOTH valid (a
//      rigid part with skin, and a bare joint/axis);
//   J) the SYSTEM PROMPT teaches the vocabulary -- otherwise the model
//      never uses it.
// =============================================================================
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"

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

// Collect the bones of a tree in the same order the parser assigns indices.
static void Bones(const Element& e, std::vector<const Bone*>& out) {
    if (e.has_bone) out.push_back(&e.bone);
    for (const auto& c : e.children) Bones(*c, out);
}

static const Bone* FindBone(const Element& e, const char* name) {
    std::vector<const Bone*> bs;
    Bones(e, bs);
    for (const Bone* b : bs) if (b->name == name) return b;
    return nullptr;
}

int main() {
    std::printf("== bone/articulation vocabulary gate ==\n");

    // --- A+B) a mechanism tree parses, and the hierarchy resolves -------
    {
        const char* json = R"({
          "kind":"assembly","children":[
            {"kind":"housing","substance":"ellipsoid",
             "substance_params":{"sx":0.4,"sy":0.4,"sz":0.4}},
            {"kind":"shaft","bone":{"name":"shaft","head":[0,0,-0.3],"tail":[0,0,0.3],
               "radius":0.05,"mass":0.4,"stiffness":0.8,"joints":["front","rear"]},
             "substance":"plane"},
            {"kind":"rotor","bone":{"name":"rotor","parent":"shaft",
               "head":[0,0,0],"tail":[0.2,0,0],"radius":0.12,"mass":0.2},
             "substance":"ellipsoid","substance_params":{"sx":0.12,"sy":0.12,"sz":0.05}}]})";
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(ParseElementTree(json, tree, why), "A: the mechanism tree parses: %s", why.c_str());
        if (tree) {
            std::vector<const Bone*> bs;
            Bones(*tree, bs);
            CHECK(bs.size() == 2, "A: two bones were collected", "got %zu", bs.size());

            const Bone* shaft = FindBone(*tree, "shaft");
            const Bone* rotor = FindBone(*tree, "rotor");
            CHECK(shaft != nullptr, "A: the 'shaft' bone exists");
            CHECK(rotor != nullptr, "A: the 'rotor' bone exists");
            if (shaft && rotor) {
                // A: the bone keeps everything it was given.
                CHECK(std::fabs(shaft->head.z + 0.3f) < 1e-5f &&
                      std::fabs(shaft->tail.z - 0.3f) < 1e-5f,
                      "A: the segment's head and tail survive");
                CHECK(std::fabs(shaft->radius - 0.05f) < 1e-6f,
                      "A: the radius survives", "got %f", shaft->radius);
                CHECK(std::fabs(shaft->mass - 0.4f) < 1e-6f,
                      "A: the mass survives", "got %f", shaft->mass);
                CHECK(std::fabs(shaft->stiffness - 0.8f) < 1e-6f,
                      "A: the stiffness survives", "got %f", shaft->stiffness);
                CHECK(shaft->joints.size() == 2 &&
                      shaft->joints[0] == "front" && shaft->joints[1] == "rear",
                      "A: the named articulations survive");
                // B: the parent NAME resolved to the right INDEX.
                CHECK(shaft->parent == -1, "B: a bone with no named parent is a root",
                      "got %d", shaft->parent);
                CHECK(rotor->parent == 0,
                      "B: 'rotor' resolves to the INDEX of 'shaft'",
                      "got %d", rotor->parent);
                CHECK(rotor->parent_name == "shaft",
                      "B: the parent name is retained for inspection");
            }
        }
    }

    // --- C) an unknown parent name is REJECTED --------------------------
    {
        const char* json = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"child","bone":{"name":"child","parent":"nosuchbone",
              "head":[0,0,0],"tail":[0,0,1]}}]})";
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(!ParseElementTree(json, tree, why),
              "C: a bone naming a nonexistent parent is REJECTED");
        CHECK(!why.empty(), "C: the rejection explains why");
        CHECK(tree == nullptr, "C: a rejected tree leaves the output untouched");
    }

    // --- D) two bones with the same name are REJECTED --------------------
    {
        const char* json = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"a","bone":{"name":"shaft","head":[0,0,0],"tail":[0,0,1]}},
            {"kind":"b","bone":{"name":"shaft","head":[0,0,0],"tail":[0,1,0]}}]})";
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(!ParseElementTree(json, tree, why),
              "D: two bones with the same name are REJECTED");
        CHECK(why.find("ambiguous") != std::string::npos ||
              why.find("shaft") != std::string::npos,
              "D: the rejection names the ambiguity", "why=%s", why.c_str());
    }

    // --- E) a bone that is its own ancestor is REJECTED ------------------
    {
        // 'a' is the parent of 'b', and 'b' is the parent of 'a' -> a cycle.
        const char* json = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"a","bone":{"name":"a","parent":"b","head":[0,0,0],"tail":[0,0,1]}},
            {"kind":"b","bone":{"name":"b","parent":"a","head":[0,0,0],"tail":[0,1,0]}}]})";
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(!ParseElementTree(json, tree, why),
              "E: a cyclic bone hierarchy is REJECTED");
        CHECK(!why.empty(), "E: the rejection explains why");
    }

    // --- F) a bone needs BOTH a head and a tail --------------------------
    {
        std::unique_ptr<Element> tree;
        std::string why;
        const char* no_tail = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"x","bone":{"name":"x","head":[0,0,0]}}]})";
        CHECK(!ParseElementTree(no_tail, tree, why),
              "F: a bone without a tail is REJECTED");
        CHECK(why.find("tail") != std::string::npos,
              "F: the rejection says the tail is missing", "why=%s", why.c_str());

        const char* no_head = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"x","bone":{"name":"x","tail":[0,0,1]}}]})";
        CHECK(!ParseElementTree(no_head, tree, why),
              "F: a bone without a head is REJECTED");
        CHECK(why.find("head") != std::string::npos,
              "F: the rejection says the head is missing", "why=%s", why.c_str());
    }

    // --- G) out-of-range bone properties are REJECTED --------------------
    {
        std::unique_ptr<Element> tree;
        std::string why;
        const char* neg_mass = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"x","bone":{"name":"x","head":[0,0,0],"tail":[0,0,1],"mass":-1}}]})";
        CHECK(!ParseElementTree(neg_mass, tree, why),
              "G: a negative mass is REJECTED");

        const char* neg_rad = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"x","bone":{"name":"x","head":[0,0,0],"tail":[0,0,1],"radius":-0.1}}]})";
        CHECK(!ParseElementTree(neg_rad, tree, why),
              "G: a negative radius is REJECTED");

        const char* stiff = R"({
          "kind":"body","substance":"plane","children":[
            {"kind":"x","bone":{"name":"x","head":[0,0,0],"tail":[0,0,1],"stiffness":5}}]})";
        CHECK(!ParseElementTree(stiff, tree, why),
              "G: a stiffness outside [0,1] is REJECTED");
    }

    // --- H) a tree WITHOUT bones still parses and still bakes ------------
    {
        const char* json = R"({
          "kind":"body","substance":"ellipsoid",
          "substance_params":{"sx":0.4,"sy":0.4,"sz":0.5}})";
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(ParseElementTree(json, tree, why), "H: a boneless tree still parses: %s", why.c_str());
        if (tree) {
            std::vector<const Bone*> bs;
            Bones(*tree, bs);
            CHECK(bs.empty(), "H: a boneless tree has no bones", "got %zu", bs.size());
            std::string bwhy;
            auto body = ElementSystem::BakeBody(*tree, 16, 1.0f, bwhy);
            CHECK(body != nullptr, "H: a boneless tree still bakes: %s", bwhy.c_str());
            if (body) CHECK(body->Patches().size() > 0,
                            "H: the baked boneless body has patches");
        }
    }

    // --- I) a bone with a substance and a bare bone are BOTH valid -------
    {
        // The 'axis' node has a bone but NO substance: a bare joint/axis.
        const char* json = R"({
          "kind":"body","substance":"ellipsoid",
          "substance_params":{"sx":0.4,"sy":0.4,"sz":0.4},
          "children":[
            {"kind":"axis","bone":{"name":"axis","head":[0,0,0],"tail":[0,0,0.5]}}]})";
        std::unique_ptr<Element> tree;
        std::string why;
        CHECK(ParseElementTree(json, tree, why),
              "I: a bare bone (no substance) is valid: %s", why.c_str());
        if (tree) {
            std::vector<const Bone*> bs;
            Bones(*tree, bs);
            CHECK(bs.size() == 1, "I: the bare bone was collected", "got %zu", bs.size());
            const Bone* axis = FindBone(*tree, "axis");
            CHECK(axis && !axis->name.empty(), "I: the bare bone kept its name");
        }
    }

    // --- J) the system prompt teaches the vocabulary ----------------------
    {
        const std::string prompt = ElementTreeSystemPrompt();
        CHECK(prompt.find("\"bone\"") != std::string::npos ||
              prompt.find("bone") != std::string::npos,
              "J: the system prompt mentions bones");
        CHECK(prompt.find("head") != std::string::npos,
              "J: the system prompt documents the bone's head");
        CHECK(prompt.find("tail") != std::string::npos,
              "J: the system prompt documents the bone's tail");
        CHECK(prompt.find("parent") != std::string::npos,
              "J: the system prompt documents the bone's parent");
        CHECK(prompt.find("joints") != std::string::npos,
              "J: the system prompt documents named articulations");
    }

    std::printf("\n%s (%d checks, %d failures)\n",
                g_fail ? "BONE-TREE VALIDATION FAILED" : "bone-tree validation passed",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
