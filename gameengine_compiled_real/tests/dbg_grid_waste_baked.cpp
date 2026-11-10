// =============================================================================
// dbg_grid_waste_baked - MEASURE the waste in the BAKED body (the thing that
// actually costs memory, bandwidth and inertia accuracy).
//
// The previous probe measured the substance's own square grid, which is the
// BEFORE picture. This one measures the baked anchors: how many are
// in-support (a real surface point) vs. the out-of-support placeholders
// (written at the bbox centre).
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace DeliRium;

namespace {

void report(const char* name, const char* json, int N) {
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(json, tree, why)) {
        std::printf("  %-24s parse failed: %s\n", name, why.c_str());
        return;
    }
    ModelOutput out;
    if (!ElementSystem::Bake(*tree, N, 1.0f, out, why)) {
        std::printf("  %-24s bake failed: %s\n", name, why.c_str());
        return;
    }
    auto body = ElementSystem::BakeBody(*tree, N, 1.0f, why);
    if (!body) { std::printf("  %-24s BakeBody failed\n", name); return; }
    const Vec3 bmn = body->BBoxMin(), bsp = body->BBoxSpan();

    // The placeholder is written at the bbox CENTRE: pos = 32767 (the
    // quantised midpoint). An in-support anchor is somewhere else.
    int placeholders = 0, real = 0;
    for (const auto& a : out.anchors) {
        bool is_mid = true;
        for (int c = 0; c < 3; ++c)
            if (a.pos[c] != 32767) { is_mid = false; break; }
        if (is_mid) ++placeholders; else ++real;
    }
    const int total = real + placeholders;
    std::printf("  %-24s anchors %5d  real %5d (%5.1f%%)  "
                "PLACEHOLDERS %5d (%5.1f%%)  surfaces %5zu\n",
                name, total, real, 100.0*real/total,
                placeholders, 100.0*placeholders/total, out.surfaces.size());
}

} // namespace

int main() {
    const int N = 48;
    std::printf("-- the BAKED waste at grid %d --\n\n", N);

    report("ellipsoid",
        R"({"kind":"body","substance":"ellipsoid",
            "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5}})", N);

    report("torus (polar)",
        R"({"kind":"body","substance":"torus",
            "substance_params":{"major":0.32,"minor":0.12}})", N);

    report("thin torus (polar)",
        R"({"kind":"body","substance":"torus",
            "substance_params":{"major":0.40,"minor":0.06}})", N);

    report("closed sphere (2 sheets)",
        R"({"kind":"body","substance":"ellipsoid",
            "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
            "children":[{"kind":"lower","substance":"ellipsoid",
                         "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                         "scale":[1,1,-1],"origin":[0,0,0]}]})", N);

    std::printf("\n-- the geometric check: is the polar torus still a torus? --\n");
    {
        std::unique_ptr<Element> tree;
        std::string why;
        const char* json =
            R"({"kind":"body","substance":"torus",
                "substance_params":{"major":0.32,"minor":0.12}})";
        if (AI::ParseElementTree(json, tree, why)) {
            ModelOutput out;
            if (ElementSystem::Bake(*tree, N, 1.0f, out, why)) {
                auto body = ElementSystem::BakeBody(*tree, N, 1.0f, why);
                const Vec3 bmn = body->BBoxMin(), bsp = body->BBoxSpan();
                float rmin = 1e9f, rmax = -1e9f, zmin = 1e9f, zmax = -1e9f;
                int in_hole = 0;
                for (const auto& a : out.anchors) {
                    bool mid = true;
                    for (int c = 0; c < 3; ++c)
                        if (a.pos[c] != 32767) { mid = false; break; }
                    if (mid) continue;
                    const float x = bmn.x + a.pos[0]*(1.f/65535.f)*bsp.x;
                    const float y = bmn.y + a.pos[1]*(1.f/65535.f)*bsp.y;
                    const float z = bmn.z + a.pos[2]*(1.f/65535.f)*bsp.z;
                    const float r = std::sqrt(x*x+y*y);
                    rmin = std::min(rmin, r); rmax = std::max(rmax, r);
                    zmin = std::min(zmin, z); zmax = std::max(zmax, z);
                    // inside the hole: r < major - minor
                    if (r < 0.32f - 0.12f - 1e-3f) ++in_hole;
                }
                std::printf("  r range: [%.4f, %.4f]  (the tube spans "
                            "[%.4f, %.4f])\n", rmin, rmax,
                            0.32f-0.12f, 0.32f+0.12f);
                std::printf("  z range: [%.4f, %.4f]  (a half tube: [0, %.4f])\n",
                            zmin, zmax, 0.12f);
                std::printf("  anchors INSIDE the hole: %d\n", in_hole);
            }
        }
    }
    return 0;
}
