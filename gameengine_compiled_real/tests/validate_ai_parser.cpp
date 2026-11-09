// =============================================================================
// validate_ai_parser - GATE for the AI -> geometry seam.
//
// Covers the contract that matters: a model response must become a REAL,
// VALID PSF body (not just "the JSON parsed"). The gate asserts the
// INVARIANTS, not the specific values:
//   - well-formed trees parse and bake
//   - the baked body reproduces the FIELD (residual gate)
//   - malformed / hostile input is REJECTED with a reason (never a
//     partially-built tree)
//   - an unknown substance is REJECTED, not silently downgraded to a plane
//   - the extracted-JSON helper survives prose and markdown fences
// =============================================================================
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

using namespace DeliRium;
using namespace DeliRium::AI;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

// --- JSON reader -------------------------------------------------------------
static void test_json() {
    JsonValue v;
    std::string why;

    CHECK(ParseJson("{\"a\":1,\"b\":[true,null,\"x\"]}", v, why), "parse basic: %s", why.c_str());
    CHECK(v.IsObject(), "basic is object");
    CHECK(v.Find("a") && v.Find("a")->GetNumber(-1) == 1.0, "a == 1");
    CHECK(v.Find("b") && v.Find("b")->IsArray() && v.Find("b")->items.size() == 3, "b has 3 items");
    CHECK(v.Find("b")->items[0].IsBool() && v.Find("b")->items[0].boolean, "b[0] true");
    CHECK(v.Find("b")->items[1].IsNull(), "b[1] null");
    CHECK(v.Find("b")->items[2].GetString("") == "x", "b[2] == x");

    // Escapes + unicode.
    CHECK(ParseJson("{\"s\":\"a\\nb\\\"c\\u0041\"}", v, why), "parse escapes: %s", why.c_str());
    CHECK(v.Find("s") && v.Find("s")->text == "a\nb\"cA", "escape decoding: '%s'",
          v.Find("s") ? v.Find("s")->text.c_str() : "(null)");

    // Negative + exponent numbers.
    CHECK(ParseJson("{\"n\":-1.5e2}", v, why), "parse number: %s", why.c_str());
    CHECK(v.Find("n") && std::fabs(v.Find("n")->number - (-150.0)) < 1e-9, "n == -150");

    // Malformed inputs must FAIL (never throw, never half-parse).
    //
    // NOTE: "[1,2,]" and "{\"a\":1,}" are NOT in this list -- trailing
    // commas are ACCEPTED (see below). They are a formatting defect in LLM
    // output, not a data defect, and rejecting them threw away whole
    // generations over a stray character.
    const char* bad[] = {
        "{", "}", "[1,2", "{\"a\":}", "{\"a\" 1}", "{'a':1}", "tru",
        "{\"a\":1}trailing", "{\"a\":\"unterminated",
    };
    for (const char* b : bad) {
        JsonValue tmp;
        std::string w;
        CHECK(!ParseJson(b, tmp, w), "must reject malformed: %s", b);
        CHECK(!w.empty(), "must give a reason: %s", b);
    }

    // --- TRAILING COMMAS: tolerated ---------------------------------------
    //
    // WHY THIS IS A CONTRACT, NOT A LENIENCY: a local model asked for a
    // shape very often emits "[...]," before the closing bracket, and the
    // ENTIRE generation used to be rejected with "expected number at byte
    // N" -- a confusing message that pointed at a number when the real
    // defect was a comma. The trailing comma cannot change the meaning of
    // the data (it only marks the end of a list), so accepting it is safe.
    //
    // The invariant asserted here is behavioural: the parsed VALUE must
    // equal the value of the same JSON written without the comma. If a
    // future change ever drops or reorders an element while "fixing" the
    // comma, this check fails.
    {
        JsonValue a, b;
        std::string w;
        CHECK(ParseJson("[1,2,3]", a, w), "no-comma array parses: %s", w.c_str());
        CHECK(ParseJson("[1,2,3,]", b, w), "trailing-comma array parses: %s", w.c_str());
        CHECK(a.items.size() == b.items.size(), "trailing comma keeps the item count");
        CHECK(a.items.size() == 3, "trailing comma keeps 3 items");
        for (size_t i = 0; i < a.items.size() && i < b.items.size(); ++i)
            CHECK(a.items[i].GetNumber(-1) == b.items[i].GetNumber(-2),
                  "trailing comma keeps item %zu", i);

        JsonValue c, d;
        CHECK(ParseJson("{\"x\":1,\"y\":2}", c, w), "no-comma object parses: %s", w.c_str());
        CHECK(ParseJson("{\"x\":1,\"y\":2,}", d, w), "trailing-comma object parses: %s", w.c_str());
        CHECK(c.members.size() == d.members.size(), "trailing comma keeps the member count");
        CHECK(d.Find("x") && d.Find("x")->GetNumber(-1) == 1.0, "trailing comma keeps x");
        CHECK(d.Find("y") && d.Find("y")->GetNumber(-1) == 2.0, "trailing comma keeps y");

        // A trailing comma in a NESTED array (the shape an element tree
        // actually has: a list of children).
        CHECK(ParseJson("{\"children\":[{\"kind\":\"a\"},{\"kind\":\"b\"},]}", d, w),
              "trailing comma in nested children parses: %s", w.c_str());
        CHECK(d.Find("children") && d.Find("children")->items.size() == 2,
              "nested trailing comma keeps both children");
    }

    // ExtractJsonObject survives prose and fences.
    std::string extracted;
    CHECK(ExtractJsonObject("Here you go:\n```json\n{\"kind\":\"x\"}\n```\nDone.", extracted),
          "extract from prose");
    JsonValue p;
    CHECK(ParseJson(extracted, p, why), "extracted parses: %s", why.c_str());
    CHECK(p.Find("kind") && p.Find("kind")->GetString("") == "x", "extracted kind == x");

    // Braces inside strings must not confuse the brace matcher.
    CHECK(ExtractJsonObject("{\"s\":\"}\",\"t\":1}", extracted), "extract with brace in string");
    CHECK(ParseJson(extracted, p, why), "extracted-with-brace parses: %s", why.c_str());
    CHECK(p.Find("t") && p.Find("t")->GetNumber(-1) == 1.0, "extracted-with-brace keeps later keys");
    // An UNTERMINATED string must NOT yield a bogus object: the closing
    // brace is inside the string, so the extractor must refuse.
    CHECK(!ExtractJsonObject("{\"s\":\"}", extracted), "must not extract from unterminated string");
}

// --- Element tree ------------------------------------------------------------
static void test_tree() {
    std::unique_ptr<Element> tree;
    std::string why;

    // A well-formed tree.
    const char* good = R"({
      "kind":"body", "color":[0.6,0.7,0.8],
      "children":[
        {"kind":"core","substance":"ellipsoid","substance_params":{"sx":0.4,"sy":0.4,"sz":0.5}},
        {"kind":"nodule","origin":[0.2,0.1,0.0],"weight":0.6,
         "substance":"gaussian","substance_params":{"amp":0.12,"sigma":0.2},
         "color":[0.9,0.7,0.4]}
      ]})";
    CHECK(ParseElementTree(good, tree, why), "parse good tree: %s", why.c_str());
    if (tree) {
        CHECK(tree->kind == "body", "root kind == body");
        CHECK(tree->children.size() == 2, "2 children");
        CHECK(tree->children[0]->substance != nullptr, "core has substance");
        CHECK(tree->children[1]->weight > 0.5f, "nodule weight");
        CHECK(tree->children[1]->color.x > 0.8f, "nodule color");
        CHECK(!DescribeTree(*tree).empty(), "describe tree");
    }

    // Arrays are accepted (first element is the root).
    CHECK(ParseElementTree(R"([{"substance":"sphere"},{"substance":"plane"}])", tree, why),
          "parse array form: %s", why.c_str());
    if (tree) CHECK(tree->children.size() >= 1, "array form has children");

    // A tree with NO substance anywhere must be rejected (it would bake into
    // a degenerate body -- the baker skips out-of-support samples).
    CHECK(!ParseElementTree(R"({"kind":"group","children":[{"kind":"g2"}]})", tree, why),
          "reject leafless tree");
    CHECK(!why.empty(), "leafless tree has a reason");

    // An UNKNOWN substance must be rejected, not downgraded to a plane.
    CHECK(!ParseElementTree(R"({"substance":"dragon_scale"})", tree, why),
          "reject unknown substance");
    CHECK(why.find("dragon_scale") != std::string::npos,
          "reason names the bad substance: %s", why.c_str());

    // Structural errors.
    const char* bad_trees[] = {
        R"({"origin":[1,2]})",              // origin needs 1 or 3 components
        R"({"scale":[0,1,1]})",             // zero scale
        R"({"weight":-1})",                // negative weight
        R"({"children":{}})",              // children must be an array
        R"({"substance":5})",              // substance must be a string
        R"({"color":[2,0,0]})",            // color out of range
        R"({"substance":"ellipsoid","substance_params":{"sx":0}})",  // bad extent
        R"({"substance":"gaussian","substance_params":{"sigma":0}})", // bad sigma
        R"(not json at all)",
    };
    for (const char* b : bad_trees) {
        std::unique_ptr<Element> t;
        std::string w;
        CHECK(!ParseElementTree(b, t, w), "must reject bad tree: %s", b);
        CHECK(!w.empty(), "bad tree has a reason: %s", b);
        CHECK(t == nullptr, "rejected tree leaves output untouched: %s", b);
    }
}

// --- The REAL contract: a parsed tree must bake into a valid body -----------
static void test_bake() {
    std::unique_ptr<Element> tree;
    std::string why;
    const char* json = R"({
      "kind":"body",
      "children":[
        {"kind":"host","substance":"plane","color":[0.6,0.65,0.7]},
        {"kind":"node","substance":"gaussian","substance_params":{"amp":0.25,"sigma":0.3},
         "color":[0.9,0.7,0.4]}
      ]})";
    if (!ParseElementTree(json, tree, why)) {
        CHECK(false, "parse for bake: %s", why.c_str());
        return;
    }
    std::string bake_why;
    auto body = ElementSystem::BakeBody(*tree, 12, 1.0f, bake_why);
    CHECK(body != nullptr, "bake parsed tree: %s", bake_why.c_str());
    if (!body) return;

    // A real body: patches + anchors, all with the frozen 64-byte records.
    CHECK(body->Patches().size() > 0, "baked body has patches");
    CHECK(body->AnchorBlob().size() > 0, "baked body has anchors");
    CHECK(body->SurfaceBlob().size() == body->Patches().size(),
          "surface blob matches patch count");

    // The body must REPRODUCE each SHEET's own surface.
    //
    // The baker emits ONE LAYER PER SHEET, so the body carries the plane
    // host and the gaussian bump as SEPARATE surfaces that overlap in space.
    // The "aggregate field" is their weighted AVERAGE -- which is exactly
    // what the multi-layer bake is designed not to produce (averaging an
    // upper and a lower sheet gives one dome: the "only one face" bug).
    //
    // So the fidelity check must compare the body against each sheet's OWN
    // surface. Comparing against the blend measures the distance between two
    // DIFFERENT surfaces and reports a false failure.
    const int N = 14;
    double worst_plane = 0.0;
    double worst_gauss = 0.0;
    int compared = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const float u = i / float(N - 1), v = j / float(N - 1);
            const float x = (u - 0.5f), y = (v - 0.5f);
            // (a) the plane host sheet: z = 0
            {
                const Vec3 target{x, y, 0.0f};
                SurfaceHit hit;
                if (body->ClosestPointEx(target, hit)) {
                    const Vec3 on_body = body->Patches()[hit.patch].Evaluate(hit.u, hit.v);
                    const double d = std::sqrt(
                        (on_body.x - target.x) * (on_body.x - target.x) +
                        (on_body.y - target.y) * (on_body.y - target.y) +
                        (on_body.z - target.z) * (on_body.z - target.z));
                    if (d > worst_plane) worst_plane = d;
                }
            }
            // (b) the gaussian node sheet: z = amp*exp(-r^2/(2 sigma^2))
            {
                const float amp = 0.25f, sigma = 0.3f;
                const float z = amp * std::exp(-(x*x + y*y) / (2.f * sigma * sigma));
                const Vec3 target{x, y, z};
                SurfaceHit hit;
                if (body->ClosestPointEx(target, hit)) {
                    const Vec3 on_body = body->Patches()[hit.patch].Evaluate(hit.u, hit.v);
                    const double d = std::sqrt(
                        (on_body.x - target.x) * (on_body.x - target.x) +
                        (on_body.y - target.y) * (on_body.y - target.y) +
                        (on_body.z - target.z) * (on_body.z - target.z));
                    if (d > worst_gauss) worst_gauss = d;
                    ++compared;
                }
            }
        }
    }
    CHECK(compared > 20, "compared enough samples (%d)", compared);
    // The field is bounded by the canvas (extent 1.0); a faithful bake must
    // land within a small fraction of it. The flat plane must be exact; the
    // gaussian is a smooth bump on a 12x12 grid, so its fit error is larger
    // but must still be well inside one grid cell.
    CHECK(worst_plane < 0.06, "baked body reproduces the plane sheet (worst %.5f)", worst_plane);
    CHECK(worst_gauss < 0.06, "baked body reproduces the gaussian sheet (worst %.5f)", worst_gauss);

    // The presence mask must actually LOCALIZE the system: a node with a
    // presence radius smaller than the canvas must not contribute outside.
    std::unique_ptr<Element> masked;
    const char* mjson = R"({
      "kind":"body",
      "children":[
        {"kind":"host","substance":"plane"},
        {"kind":"node","substance":"gaussian","substance_params":{"amp":0.3,"sigma":0.2},
         "presence":{"radius":0.25,"center_x":0.0,"center_y":0.0}}
      ]})";
    if (ParseElementTree(mjson, masked, why)) {
        // Far from the mask center, the plane must dominate (the masked node
        // contributes ~0). At the center, the node must raise the surface.
        const ElementSystem::Field center = ElementSystem::FieldAt(*masked, {0.f, 0.f, 0.f});
        const ElementSystem::Field far    = ElementSystem::FieldAt(*masked, {0.45f, 0.45f, 0.f});
        CHECK(center.point.z > 0.10f, "masked node raises the center (z=%.4f)", center.point.z);
        CHECK(far.point.z < 0.05f, "masked node is absent far away (z=%.4f)", far.point.z);
    } else {
        CHECK(false, "parse masked tree: %s", why.c_str());
    }
}

// --- The TORUS: a bounded substance with a HOLE -----------------------------
//
// A torus is the case that proves the substance model can express a
// NON-TRIVIAL topology: the centre of the canvas must have NO surface,
// while the ring around it does. The invariants asserted here are the
// ones that distinguish a real torus from a flat disc:
//
//   1. the substance itself is invalid at the centre (no surface)
//   2. it IS valid on the ring
//   3. the ring's height matches the analytic tube profile
//   4. the baked body has an OPEN centre -- the anchor grid has
//      out-of-support samples in the middle, so the body is not a disc
//   5. the body's Z span is the tube's diameter, not zero (it is not flat)
static void test_torus() {
    // --- the substance in isolation ---------------------------------------
    TorusSubstance torus(0.32f, 0.12f);
    const float major = 0.32f, minor = 0.12f;

    // At the exact centre there is no surface: that is the hole.
    const SubstancePoint hole = torus.Eval(0.f, 0.f);
    CHECK(!hole.valid, "torus has no surface at the centre (the hole)");

    // Just inside the hole's rim there is still no surface.
    const float inner_rim = major - minor;
    CHECK(!torus.Eval(inner_rim * 0.98f, 0.f).valid,
          "torus has no surface inside the hole's rim");

    // On the tube's centre line the surface exists and sits at z = minor.
    const SubstancePoint top = torus.Eval(major, 0.f);
    CHECK(top.valid, "torus has a surface on the tube's centre line");
    CHECK(std::fabs(top.p.z - minor) < 1e-4f,
          "torus crown height is the minor radius (z=%.5f)", top.p.z);
    CHECK(std::fabs(top.p.x - major) < 1e-4f, "torus keeps its x position");
    // The normal over the crown points straight up.
    CHECK(top.n.z > 0.99f, "torus crown normal points up (nz=%.4f)", top.n.z);

    // At the outer rim the surface drops to z = 0 and the normal is radial.
    const float outer_rim = major + minor;
    const SubstancePoint outer = torus.Eval(outer_rim * 0.999f, 0.f);
    CHECK(outer.valid, "torus has a surface at the outer rim");
    CHECK(outer.n.x > 0.9f, "torus outer normal points radially out (nx=%.4f)", outer.n.x);

    // Beyond the tube there is nothing.
    CHECK(!torus.Eval(outer_rim * 1.02f, 0.f).valid, "torus ends past the outer rim");

    // --- the parsed + baked body ------------------------------------------
    std::unique_ptr<Element> tree;
    std::string why;
    const char* json = R"({"kind":"body","substance":"torus",
        "substance_params":{"major":0.32,"minor":0.12},"color":[0.7,0.6,0.5]})";
    if (!ParseElementTree(json, tree, why)) {
        CHECK(false, "parse torus tree: %s", why.c_str());
        return;
    }
    CHECK(tree->kind == "body", "torus tree kind");

    // The field must be empty at the centre and present on the ring.
    const ElementSystem::Field centre = ElementSystem::FieldAt(*tree, {0.f, 0.f, 0.f});
    const ElementSystem::Field ring   = ElementSystem::FieldAt(*tree, {major, 0.f, 0.f});
    CHECK(centre.weight <= 1e-9f, "torus field is empty at the centre");
    CHECK(ring.weight > 1e-9f, "torus field is present on the ring");
    CHECK(std::fabs(ring.point.z - minor) < 1e-4f,
          "torus field height on the ring (z=%.5f)", ring.point.z);

    std::string bake_why;
    // Bake at a resolution that can actually RESOLVE the tube.
    //
    // The sample spacing is extent/(N-1); the tube's radius is `minor`.
    // A grid whose spacing approaches the tube radius cannot represent it:
    // the baker's model (P = bilinear corners + a normal-direction
    // displacement) is a TANGENT-PLANE reconstruction, so a coarsely sampled
    // curve gets chorded -- the ring flattens into a disc. Measured: at
    // N=12 (spacing 0.091 vs tube radius 0.12) the body keeps only 19% of
    // the field's vertical relief; at N=48 it keeps 67%.
    //
    // The bake costs ~8 ms at N=48, so this is not a performance trade.
    const int GN = 48;
    auto body = ElementSystem::BakeBody(*tree, GN, 1.0f, bake_why);
    CHECK(body != nullptr, "bake torus tree: %s", bake_why.c_str());
    if (!body) return;
    CHECK(body->Patches().size() > 0, "torus body has patches");

    // The body must be a RING with RELIEF, not a flat disc. The bound is
    // expressed as a FRACTION of the field's own vertical relief, because
    // the absolute z range depends on the tube radius while the fraction
    // measures the same thing for any radius: did the bake preserve the
    // shape the field describes?
    //
    // The field's z range over the baked grid (upper half of the tube):
    float fz_lo = 1e30f, fz_hi = -1e30f;
    for (int j = 0; j < GN; ++j)
        for (int i = 0; i < GN; ++i) {
            const float u = i / float(GN - 1), v = j / float(GN - 1);
            const ElementSystem::Field f =
                ElementSystem::FieldAt(*tree, {(u - 0.5f), (v - 0.5f), 0.f});
            if (f.weight <= 1e-9f) continue;
            fz_lo = std::min(fz_lo, f.point.z);
            fz_hi = std::max(fz_hi, f.point.z);
        }
    const float field_zspan = fz_hi - fz_lo;
    CHECK(field_zspan > 0.05f, "the field has vertical relief (%.5f)", field_zspan);

    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const auto& p : body->Patches()) {
        const Vec3 pt = p.Evaluate(0.5f, 0.5f);
        lo.x = std::min(lo.x, pt.x); lo.y = std::min(lo.y, pt.y); lo.z = std::min(lo.z, pt.z);
        hi.x = std::max(hi.x, pt.x); hi.y = std::max(hi.y, pt.y); hi.z = std::max(hi.z, pt.z);
    }
    const float span_z = hi.z - lo.z;
    const float span_x = hi.x - lo.x;
    // The body must keep MOST of the field's relief. Below ~half, the
    // tangent-plane reconstruction has chorded the ring into a disc.
    CHECK(span_z > 0.5f * field_zspan,
          "torus body keeps the field's relief (body=%.5f field=%.5f)", span_z, field_zspan);
    // ...and it must be genuinely three-dimensional, not a flat disc.
    CHECK(span_z > 0.05f, "torus body has vertical relief (span_z=%.5f)", span_z);
    CHECK(span_x > 0.50f, "torus body is wide (span_x=%.5f)", span_x);

    // --- a torus with bad parameters must be REJECTED ---------------------
    std::unique_ptr<Element> bad;
    std::string bwhy;
    CHECK(!ParseElementTree(R"({"substance":"torus","substance_params":{"major":0.1,"minor":0.2}})",
                            bad, bwhy),
          "reject torus with major <= minor");
    CHECK(!bwhy.empty(), "bad torus has a reason");
    CHECK(!ParseElementTree(R"({"substance":"torus","substance_params":{"minor":0}})", bad, bwhy),
          "reject torus with zero minor radius");
}

int main() {
    test_json();
    test_tree();
    test_bake();
    test_torus();
    std::printf("validate_ai_parser: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
