// =============================================================================
// sphere_probe - why the SPHERE came out as a flat face.
//
// The user reported: "the sphere however isnt and only shows one face
// rebuilding it". The model asked for "a smooth glass ellipsoid" and
// emitted:
//
//   body          substance=ellipsoid
//     sphere     substance=plane
//
// The PLANE is an infinite surface at z=0. With a FLAT weight, the average
// of the dome (z up to 0.5) and the plane (z=0) is a shape whose height is
// HALF the dome -- and, worse, the plane extends to the whole canvas while
// the ellipsoid is bounded, so outside the ellipsoid the plane is the ONLY
// contributor and the result is a flat card with a bump on it. That is
// exactly "only one face".
//
// The fix: DEPTH-WEIGHTED contribution. A nested child contributes less
// than its parent, so the parent's shape survives and the child refines it.
//
// This probe prints the field's Z profile for the exact failing tree, with
// and without the depth weighting, so the difference is measured rather
// than asserted.
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::AI;

static void profile(const char* label, const Element& tree) {
    std::printf("\n%s\n", label);
    // Sample the field along the +X axis at z=0 (the canvas plane).
    std::printf("   x:      ");
    for (int i = 0; i <= 8; ++i) std::printf("%6.2f", -0.5f + 0.125f * i);
    std::printf("\n");
    std::printf("   z:      ");
    for (int i = 0; i <= 8; ++i) {
        const float x = -0.5f + 0.125f * i;
        const ElementSystem::Field f = ElementSystem::FieldAt(tree, {x, 0.f, 0.f});
        std::printf("%6.3f", f.point.z);
    }
    std::printf("\n");
    // The dome alone: z at x=0 should be the ellipsoid's sz.
    const ElementSystem::Field mid = ElementSystem::FieldAt(tree, {0.f, 0.f, 0.f});
    std::printf("   centre z=%.4f   weight=%.3f\n", mid.point.z, mid.weight);
    // How much of the canvas is FLAT (plane-dominated)? Sample the corners.
    float flat = 0.f;
    int n = 0;
    for (int j = -4; j <= 4; ++j)
        for (int i = -4; i <= 4; ++i) {
            const float x = -0.5f + 0.125f * i, y = -0.5f + 0.125f * j;
            const ElementSystem::Field f = ElementSystem::FieldAt(tree, {x, y, 0.f});
            if (f.weight <= 1e-9f) continue;
            if (std::fabs(f.point.z) < 1e-3f) flat += 1.f;
            n += 1;
        }
    std::printf("   flat samples: %d/%d (%.0f%%)\n", int(flat), n, n ? 100.f * flat / n : 0.f);
}

int main() {
    std::printf("=== SPHERE PROBE: ellipsoid with a plane child ===\n");

    // --- the exact tree the model produced ---------------------------------
    const char* json = R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.4,"sy":0.4,"sz":0.5},
        "children":[{"kind":"sphere","substance":"plane"}]})";
    std::unique_ptr<Element> tree;
    std::string why;
    if (!ParseElementTree(json, tree, why)) {
        std::printf("parse failed: %s\n", why.c_str());
        return 1;
    }
    profile("A) ellipsoid + plane child (depth-weighted):", *tree);

    // --- the dome alone, for reference --------------------------------------
    const char* dome = R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.4,"sy":0.4,"sz":0.5}})";
    std::unique_ptr<Element> dome_tree;
    if (ParseElementTree(dome, dome_tree, why)) {
        profile("B) the ellipsoid ALONE (reference):", *dome_tree);
    }

    // --- bake both and compare the resulting bodies ------------------------
    auto bake = [&](const Element& t, const char* label) {
        std::string bw;
        auto b = ElementSystem::BakeBody(t, 48, 1.0f, bw);
        if (!b) { std::printf("%s: bake FAILED: %s\n", label, bw.c_str()); return; }
        float lo = 1e30f, hi = -1e30f;
        for (const auto& p : b->Patches()) {
            const Vec3 c = p.Evaluate(0.5f, 0.5f);
            lo = std::min(lo, c.z); hi = std::max(hi, c.z);
        }
        std::printf("%s: %zu patches, z span %.4f (%.4f..%.4f)\n",
                    label, b->Patches().size(), hi - lo, lo, hi);
    };
    std::printf("\n--- baked bodies ---\n");
    bake(*tree, "A) with the plane child");
    if (dome_tree) bake(*dome_tree, "B) the ellipsoid alone");

    // --- the invariant ------------------------------------------------------
    // The dome's crown must survive the child: the blended centre must be
    // at least 60% of the pure dome's height. Before the fix it was ~50%
    // at the centre and 0% at the rim (the plane took over entirely).
    if (dome_tree) {
        const ElementSystem::Field blended = ElementSystem::FieldAt(*tree, {0.f, 0.f, 0.f});
        const ElementSystem::Field pure = ElementSystem::FieldAt(*dome_tree, {0.f, 0.f, 0.f});
        const float ratio = pure.point.z > 1e-6f ? blended.point.z / pure.point.z : 0.f;
        std::printf("\ncentre height ratio (blended/pure) = %.3f\n", ratio);
        std::printf("%s: the parent keeps a majority voice over its child\n",
                    ratio > 0.60f ? "OK" : "FAIL");
    }
    return 0;
}
