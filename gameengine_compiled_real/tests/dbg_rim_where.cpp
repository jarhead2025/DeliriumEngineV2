// Diagnostic: WHERE does each sheet's RIM land in world space?
//
// The closed body's two sheets should OVERLAP in an annulus on the seam
// plane (z = 0). This prints the world (x, y) of the rim points each sheet
// produces, so the overlap (or its absence) is visible.
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace DeliRium;

int main() {
    // The ellipsoid alone, and the ellipsoid with a rim.
    EllipsoidSubstance ell(0.5f, 0.5f, 0.5f);
    const float rim = 0.06f;

    std::printf("-- the rim's world points, sampled on a grid --\n");
    std::printf("(the UPPER sheet: scale z = +1; the LOWER sheet: scale z = -1)\n");
    int n_up = 0, n_low = 0, coincident = 0;
    std::vector<std::pair<float,float>> up, low;
    const int N = 48;
    // The sheet's domain: the footprint (0.5) PLUS the rim (6% of 0.5).
    const float hx = 0.5f * (1.f + rim);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float lx = ((i + 0.5f) / N - 0.5f) * (hx * 2.f);
            const float ly = ((j + 0.5f) / N - 0.5f) * (hx * 2.f);
            const SubstancePoint sp = ell.Eval(lx, ly);
            if (!sp.valid) continue;
            // A rim point is one at z == 0 (the seam plane).
            if (std::fabs(sp.p.z) > 1e-6f) continue;
            // UPPER: z = +0 * +1 = 0. LOWER: z = +0 * -1 = -0 = 0.
            // So BOTH land on the seam plane at the SAME (x, y).
            up.push_back({sp.p.x, sp.p.y});
            ++n_up;
        }
    }
    std::printf("rim points found: %d\n", n_up);
    if (n_up) {
        float xmin=1e9f,xmax=-1e9f,ymin=1e9f,ymax=-1e9f;
        for (auto& p : up) {
            xmin=std::min(xmin,p.first); xmax=std::max(xmax,p.first);
            ymin=std::min(ymin,p.second); ymax=std::max(ymax,p.second);
        }
        std::printf("rim extent: x [%+.4f, %+.4f]  y [%+.4f, %+.4f]\n",
                    xmin,xmax,ymin,ymax);
        std::printf("  (the footprint's edge is at radius 0.5; the rim reaches "
                    "0.5 * (1 + %.2f) = %.4f)\n", rim, 0.5f*(1.f+rim));
        std::printf("  a sample of the rim points:\n");
        for (size_t k = 0; k < up.size() && k < 6; ++k)
            std::printf("    (%+.4f, %+.4f)  r=%.4f\n", up[k].first, up[k].second,
                        std::sqrt(up[k].first*up[k].first + up[k].second*up[k].second));
    }

    // Now the REAL question: in the baked body, where do the two sheets'
    // anchors land? The baker samples each sheet on ITS OWN grid -- and the
    // grids may be offset from each other.
    const char* tree_json =
        R"({"kind":"body","substance":"ellipsoid",
            "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
            "children":[{"kind":"lower","substance":"ellipsoid",
                         "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                         "scale":[1,1,-1],"origin":[0,0,0]}]})";
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(tree_json, tree, why)) {
        std::printf("parse failed: %s\n", why.c_str()); return 1;
    }
    ModelOutput out;
    if (!ElementSystem::Bake(*tree, 48, 1.0f, out, why)) {
        std::printf("bake failed: %s\n", why.c_str()); return 1;
    }
    auto body = ElementSystem::BakeBody(*tree, 48, 1.0f, why);
    const Vec3 bmn = body->BBoxMin(), bsp = body->BBoxSpan();
    // layer 0 = the upper sheet's anchors, layer 1 = the lower's
    const size_t per = 48 * 48;
    std::printf("\n-- the BAKED anchors on the seam plane, per layer --\n");
    std::vector<std::pair<float,float>> lay[2];
    for (int layer = 0; layer < 2; ++layer) {
        int n = 0; float xmin=1e9f,xmax=-1e9f;
        for (size_t k = 0; k < per; ++k) {
            const auto& a = out.anchors[layer*per + k];
            const float x = bmn.x + a.pos[0]*(1.f/65535.f)*bsp.x;
            const float y = bmn.y + a.pos[1]*(1.f/65535.f)*bsp.y;
            const float z = bmn.z + a.pos[2]*(1.f/65535.f)*bsp.z;
            if (std::fabs(z) > 1e-3f) continue;
            ++n; xmin=std::min(xmin,x); xmax=std::max(xmax,x);
            lay[layer].push_back({x,y});
        }
        std::printf("layer %d: %d anchors at z~0, x range [%+.4f, %+.4f]\n",
                    layer, n, xmin, xmax);
    }
    // The NEAREST neighbour distance between the two layers' seam anchors.
    // If the rims overlap, this is ~0. If the grids are offset, it is the
    // grid step.
    float best = 1e9f, sum = 0.f; int cnt = 0;
    for (auto& p : lay[0]) {
        float d = 1e9f;
        for (auto& q : lay[1]) {
            const float dx = p.first-q.first, dy = p.second-q.second;
            d = std::min(d, std::sqrt(dx*dx+dy*dy));
        }
        best = std::min(best, d); sum += d; ++cnt;
    }
    std::printf("\nnearest-neighbour distance layer0 -> layer1: "
                "min %.6f, mean %.6f  (n=%d)\n", best, sum/std::max(1,cnt), cnt);
    // The grid step: the sheet's domain is 0.53 half-extent over N-1 cells.
    std::printf("the sheet's grid step is about %.6f\n",
                2.f*0.53f/(48-1));
    return 0;
}
