// Diagnostic: WHY does the ellipsoid's waste rise from 16.3% to 25.5%?
// The domain grew from 1.0x to 1.06x (the rim). Are the extra cells
// VALID rim anchors (good -- the rim is the overlap mechanism) or
// placeholders (bad -- wasted)?
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/AnchorPool.h"
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

using namespace DeliRium;

int main() {
    const char* kEll = R"({"kind":"body","substance":"ellipsoid",
        "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5}})";

    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(kEll, tree, why)) { std::printf("parse: %s\n", why.c_str()); return 1; }

    const int N = 48;
    ModelOutput out;
    if (!ElementSystem::Bake(*tree, N, 1.0f, out, why)) { std::printf("bake: %s\n", why.c_str()); return 1; }

    std::unique_ptr<PSFBody> body = ElementSystem::BakeBody(*tree, N, 1.0f, why);
    if (!body) { std::printf("BakeBody failed\n"); return 1; }
    const Vec3 bmn = body->BBoxMin(), bsp = body->BBoxSpan();

    int placeholder = 0, real = 0;
    // Classify the REAL anchors by their radius: inside the footprint
    // (q < 1) vs on the rim band (q >= 1, still a valid rim point).
    int in_footprint = 0, on_rim = 0;
    float rmax_real = 0.f;
    for (const auto& a : out.anchors) {
        if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767) { ++placeholder; continue; }
        ++real;
        const float x = bmn.x + (a.pos[0] / 65535.0f) * bsp.x;
        const float y = bmn.y + (a.pos[1] / 65535.0f) * bsp.y;
        const float z = bmn.z + (a.pos[2] / 65535.0f) * bsp.z;
        const float q = (x*x)/(0.5f*0.5f) + (y*y)/(0.5f*0.5f);
        if (q < 1.0f) ++in_footprint; else ++on_rim;
        const float r = std::sqrt(x*x + y*y);
        if (r > rmax_real) rmax_real = r;
    }
    std::printf("anchors: %d total, %d real, %d placeholders\n",
                int(out.anchors.size()), real, placeholder);
    std::printf("real anchors: %d inside the footprint (q<1), %d on the RIM band (q>=1)\n",
                in_footprint, on_rim);
    std::printf("max radius among real anchors: %.4f (the footprint's half-extent is 0.5)\n",
                rmax_real);
    std::printf("waste = %.1f%%\n", 100.0f * placeholder / (real + placeholder));
    std::printf("\nbbox: min (%.4f, %.4f, %.4f) span (%.4f, %.4f, %.4f)\n",
                bmn.x, bmn.y, bmn.z, bsp.x, bsp.y, bsp.z);
    return 0;
}
