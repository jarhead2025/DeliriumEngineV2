// Diag 3: instrument the bake. The substance yields 188 wall samples on
// the grid, but the baked body (measured through the .psf file) had
// SIDE-WALL 0. This probe calls ElementSystem::BakeBody -- the exact call
// make_cube_asset uses -- and reports the same distribution cube_diag
// reports, so the dropping stage is between the substance and the records.
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace DeliRium;

int main() {
    auto root = std::make_unique<Element>();
    root->kind = "body";
    root->color = {0.549f, 0.549f, 0.549f};
    root->substance = std::make_unique<BoxSubstance>(0.5f, 0.5f, 0.5f);
    auto lower = std::make_unique<Element>();
    lower->kind = "box_lower";
    lower->substance = std::make_unique<BoxSubstance>(0.5f, 0.5f, 0.5f);
    lower->origin = {0.f, 0.f, 0.f};
    lower->scale  = {1.f, 1.f, -1.f};
    lower->color = {0.549f, 0.549f, 0.549f};
    root->AddChild(std::move(lower));

    std::string why;
    auto body = ElementSystem::BakeBody(*root, 48, 1.0f, why);
    if (!body) {
        std::printf("bake failed: %s\n", why.c_str());
        return 1;
    }
    std::printf("baked: %zu anchors, %zu surfaces\n",
                body->AnchorBlob().size(), body->SurfaceBlob().size());
    std::printf("bbox span (%.4f %.4f %.4f)\n",
                body->BBoxSpan().x, body->BBoxSpan().y, body->BBoxSpan().z);

    // Classify, exactly as cube_diag does -- via the pool (dequantised).
    int top = 0, bottom = 0, wall = 0, interior = 0;
    const size_t n = body->AnchorBlob().size();
    const auto& pool = body->AnchorsConst();
    for (size_t i = 0; i < n; ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        if (p.z > 0.48f) ++top;
        else if (p.z < -0.48f) ++bottom;
        else if (std::fabs(p.x) > 0.48f || std::fabs(p.y) > 0.48f) ++wall;
        else ++interior;
    }
    std::printf("distribution: top %d  bottom %d  WALL %d  interior %d\n",
                top, bottom, wall, interior);
    // ALSO: count wall-ish anchors by the QUANTISED extent -- if the wall
    // points were baked but quantised onto the footprint edge (the box
    // clamp in Eval: p.x = clamped cx), the anchors' x would sit exactly
    // at the edge and my |x|>0.48 test may miss them (0.5 > 0.48 should
    // pass). Print the anchor histogram in x to see.
    int hist[5] = {0, 0, 0, 0, 0};   // x < -0.45, -0.45..-0.05, -0.05..0.05, ...
    for (size_t i = 0; i < n; ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        if (p.x < -0.45f) ++hist[0];
        else if (p.x < -0.05f) ++hist[1];
        else if (p.x < 0.05f) ++hist[2];
        else if (p.x < 0.45f) ++hist[3];
        else ++hist[4];
    }
    std::printf("x histogram: <-0.45 %d | -0.45..-0.05 %d | centre %d | +0.05..+0.45 %d | >+0.45 %d\n",
                hist[0], hist[1], hist[2], hist[3], hist[4]);
    return 0;
}
