// =============================================================================
// validate_model_lessons.cpp - The parser's ERRORS MUST TEACH.
//
// WHY THIS GATE EXISTS
//
// A model that gets a shape's PARAMETERS wrong is not "a bad model" -- it is
// a model doing arithmetic it cannot do. Every one of its measured failures
// was arithmetic or vocabulary:
//
//   1. "major": 0.44 -- the OUTER radius used as major (should be 0.32)
//   2. "mass": 7850 * (4/3 * Math.PI * ...) -- JAVASCRIPT SOURCE as a value
//   3. {"kind": "ring_housing", "substance": "ring_housing"} -- a PART'S NAME
//      in the SHAPE slot
//   4. two "children" arrays in one object -- the second SILENTLY DROPPED
//   5. a flat "unknown substance 'x'" -- which tells the model nothing
//
// The engine already owns the mathematics that fixes 1 and 2 (the geometric
// tools). The parser's job is to make the FAILURE teach, so the escalation
// attempt and the tool loop can correct the model instead of it repeating.
//
// The invariants here are about BEHAVIOUR, not about specific strings:
//   * a duplicate "children" array must not lose geometry;
//   * an expression-as-a-value must be rejected WITH a pointer at the tools;
//   * a part-name-as-substance must be diagnosed differently from an
//     invented substance.
// A message can be reworded freely without breaking this gate; only losing
// the behaviour breaks it.
// =============================================================================
#include <cstdio>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/DocLearner.h"

using namespace DeliRium;
using namespace DeliRium::AI;

static int g_checks = 0;
static int g_failures = 0;

static void Check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL  %s\n", what);
    } else {
        std::printf("  ok    %s\n", what);
    }
}

// Count the sheets in a tree (elements that HAVE a substance).
static size_t CountSheets(const Element& e) {
    size_t n = e.substance ? 1 : 0;
    for (const auto& c : e.children) n += CountSheets(*c);
    return n;
}

// Count the elements in a tree.
static size_t CountElements(const Element& e) {
    size_t n = 1;
    for (const auto& c : e.children) n += CountElements(*c);
    return n;
}

int main() {
    std::printf("=== validate_model_lessons ===\n");

    // -------------------------------------------------------------------------
    // 1. A DUPLICATE "children" ARRAY MUST NOT LOSE GEOMETRY.
    // -------------------------------------------------------------------------
    // MEASURED: a live 7b wrote two "children" arrays in one element; the
    // second held the mirrored lower half. `Find` returned the first, so the
    // lower half was silently discarded and the body baked OPEN with no
    // error anywhere.
    {
        const char* kJson = R"({
          "kind": "assembly",
          "substance": "ellipsoid",
          "substance_params": {"sx":0.5,"sy":0.5,"sz":0.5},
          "children": [
            {"kind":"pinion","substance":"ellipsoid",
             "substance_params":{"sx":0.1,"sy":0.1,"sz":0.3}}
          ],
          "children": [
            {"kind":"lower","substance":"ellipsoid",
             "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
             "scale":[1,1,-1],"origin":[0,0,0]}
          ]
        })";
        std::unique_ptr<Element> tree;
        std::string why;
        const bool ok = ParseElementTree(kJson, tree, why);
        Check(ok, "a tree with two \"children\" arrays PARSES");
        if (ok) {
            // The union of both arrays: assembly + pinion + lower = 3 elements.
            Check(CountElements(*tree) == 3,
                  "both \"children\" arrays contribute (3 elements, not 2)");
            Check(CountSheets(*tree) == 3,
                  "all three sheets survive (the lower half is not lost)");
            // The lower half must be flipped: its scale.z is negative.
            bool has_flipped = false;
            for (const auto& c : tree->children)
                if (c->scale.z < 0.f) has_flipped = true;
            Check(has_flipped, "the second array's flipped lower sheet survives");
        } else {
            std::printf("        why: %s\n", why.c_str());
        }
    }

    // -------------------------------------------------------------------------
    // 2. AN EXPRESSION IS NOT A VALUE -- and the error must POINT AT THE TOOLS.
    // -------------------------------------------------------------------------
    // MEASURED: a live 7b emitted "mass": 7850 * (4/3 * Math.PI * ...) and
    // the error was "expected ',' or '}' at byte 257" -- which names neither
    // the mistake nor the way out.
    {
        const char* kJson = R"({
          "kind": "ring",
          "substance": "torus",
          "bone": {
            "name": "ring", "head": [0,0,0], "tail": [0,0,0],
            "mass": 7850 * (4/3 * Math.PI * (Math.pow(0.44,3) - Math.pow(0.32,3)))
          }
        })";
        std::unique_ptr<Element> tree;
        std::string why;
        const bool ok = ParseElementTree(kJson, tree, why);
        Check(!ok, "an expression as a value is REJECTED");
        Check(why.find("EXPRESSION") != std::string::npos ||
              why.find("expression") != std::string::npos,
              "the error names the mistake (an expression, not a number)");
        Check(why.find("fit_sphere_to_mass") != std::string::npos ||
              why.find("volume_of") != std::string::npos,
              "the error POINTS AT THE TOOL that replaces the arithmetic");
    }

    // The same, at the JSON level: the reader must not report a byte offset
    // and stop there.
    {
        JsonValue jv;
        std::string why;
        const bool ok = ParseJson(R"({"a": 2 * 3})", jv, why);
        Check(!ok, "the JSON reader rejects an expression value");
        // CASE-INSENSITIVELY: the message's WORDING is not the contract, the
        // BEHAVIOUR is. The reader says "EXPRESSION" in capitals to make it
        // stand out to the model, and this gate must not rot when the
        // capitalisation changes -- a message can be reworded freely without
        // breaking this gate; only losing the behaviour breaks it.
        std::string lower = why;
        for (char& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
        Check(lower.find("expression") != std::string::npos,
              "the reader's error names the expression (not just a byte offset)");
    }

    // -------------------------------------------------------------------------
    // 3. A PART NAME IN THE SUBSTANCE SLOT IS A DIFFERENT MISTAKE.
    // -------------------------------------------------------------------------
    {
        std::unique_ptr<Element> tree;
        std::string why;
        // "ring_housing" has an underscore: it reads as a part name.
        const bool ok = ParseElementTree(
            R"({"kind":"ring_housing","substance":"ring_housing"})", tree, why);
        Check(!ok, "a part name in the substance slot is REJECTED");
        Check(why.find("PART NAME") != std::string::npos ||
              why.find("part's name") != std::string::npos ||
              why.find("part name") != std::string::npos,
              "a part name is diagnosed as a part name");
        Check(why.find("\"kind\"") != std::string::npos &&
              why.find("\"substance\"") != std::string::npos,
              "the error names BOTH fields and which goes where");
    }

    // An invented substance is a DIFFERENT diagnosis -- and it must still be
    // rejected (the parser must never degrade an unknown shape into a card).
    {
        std::unique_ptr<Element> tree;
        std::string why;
        const bool ok = ParseElementTree(
            R"({"kind":"part","substance":"dragon_scale"})", tree, why);
        Check(!ok, "an invented substance is still REJECTED");
        Check(why.find("unknown substance") != std::string::npos,
              "an invented substance is called an unknown substance");
        Check(why.find("torus") != std::string::npos,
              "the error lists the acceptable substance types");
    }

    // -------------------------------------------------------------------------
    // 4. THE TORUS OUTER-RADIUS MISTAKE.
    // -------------------------------------------------------------------------
    // "outer_radius" is converted to major correctly, and a contradiction
    // between "major" and "outer_radius" is reported with the tool that
    // resolves it.
    {
        std::unique_ptr<Element> tree;
        std::string why;
        const bool ok = ParseElementTree(
            R"({"kind":"ring","substance":"torus",
                "substance_params":{"outer_radius":0.44,"minor":0.12}})",
            tree, why);
        Check(ok, "an explicit \"outer_radius\" is accepted");
        if (ok) {
            // The built substance must have major = 0.44 - 0.12 = 0.32.
            //
            // Measured through the PUBLIC surface rather than by widening
            // the class for the test: `Domain()` reports the ring's OUTER
            // extent, which is major + minor. The wrong conversion (using
            // the outer radius as major) would give 0.44 + 0.12 = 0.56.
            const auto* sub = tree->substance.get();
            const auto* tor = dynamic_cast<const TorusSubstance*>(sub);
            Check(tor != nullptr, "the substance is a torus");
            if (tor) {
                float hx = 0.f, hy = 0.f;
                tor->Domain(hx, hy);
                Check(std::fabs(hx - 0.44f) < 1e-5f,
                      "outer_radius is converted to major (outer extent = 0.44, "
                      "not 0.56)");
            }
        } else {
            std::printf("        why: %s\n", why.c_str());
        }
    }
    {
        std::unique_ptr<Element> tree;
        std::string why;
        // major = 0.44 AND outer_radius = 0.44: the model used the outer
        // radius as major. That must be caught.
        const bool ok = ParseElementTree(
            R"({"kind":"ring","substance":"torus",
                "substance_params":{"major":0.44,"minor":0.12,"outer_radius":0.44}})",
            tree, why);
        Check(!ok, "major == outer_radius is caught (the outer-radius mistake)");
        Check(why.find("resolve_torus_radii") != std::string::npos,
              "the error names the tool that resolves it");
    }

    // -------------------------------------------------------------------------
    // 5. THE PROMPT MUST TEACH THE TWO NAME FIELDS.
    // -------------------------------------------------------------------------
    // A prompt that does not state the rule leaves the model to guess which
    // field takes the part's name -- and it guesses wrong, measurably.
    {
        const std::string p = DocToTreeSystemPrompt();
        Check(p.find("\"kind\"") != std::string::npos &&
              p.find("\"substance\"") != std::string::npos,
              "the prompt names both fields");
        Check(p.find("SHAPE TYPE") != std::string::npos,
              "the prompt says which field holds the SHAPE TYPE");
        Check(p.find("NUMBERS ARE VALUES") != std::string::npos ||
              p.find("not a formula") != std::string::npos ||
              p.find("not formulas") != std::string::npos,
              "the prompt says numbers are values, not formulas");
        Check(p.find("children") != std::string::npos,
              "the prompt mentions the children array");
    }

    // -------------------------------------------------------------------------
    // 6. THE TEACHING MUST NOT BREAK THE VALID CASES.
    // -------------------------------------------------------------------------
    {
        std::unique_ptr<Element> tree;
        std::string why;
        const bool ok = ParseElementTree(
            R"({"kind":"sphere","substance":"sphere",
                "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                "children":[{"kind":"lower","substance":"sphere",
                             "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                             "scale":[1,1,-1],"origin":[0,0,0]}]})",
            tree, why);
        Check(ok, "a correct closed sphere still parses");
        if (ok) {
            Check(CountSheets(*tree) == 2, "the closed sphere has two sheets");
        } else {
            std::printf("        why: %s\n", why.c_str());
        }
    }
    {
        // A grouping node with "substance": "" -- the spelling a 14B used.
        std::unique_ptr<Element> tree;
        std::string why;
        const bool ok = ParseElementTree(
            R"({"kind":"assembly","substance":"",
                "children":[{"kind":"shaft","substance":"ellipsoid",
                             "substance_params":{"sx":0.05,"sy":0.05,"sz":0.4}}]})",
            tree, why);
        Check(ok, "a grouping node with \"substance\": \"\" still parses");
        if (ok) Check(CountSheets(*tree) == 1,
                      "the grouping node contributes no sheet of its own");
    }
    {
        // The single "children" array case must be unchanged.
        std::unique_ptr<Element> tree;
        std::string why;
        const bool ok = ParseElementTree(
            R"({"kind":"a","substance":"plane",
                "children":[{"kind":"b","substance":"plane"}]})", tree, why);
        Check(ok && CountElements(*tree) == 2,
              "a single \"children\" array is unchanged (2 elements)");
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) std::printf("VALIDATE_MODEL_LESSONS PASSED\n");
    else std::printf("VALIDATE_MODEL_LESSONS FAILED\n");
    return g_failures == 0 ? 0 : 1;
}
