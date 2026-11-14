// Diag 4: the cube now has walls, but the render shows (a) the mirrored
// sheet poking out behind instead of sealing, and (b) a dark top rim.
//
// (b) is diagnosable from the records: the wall band's FIRST row is at
// z = 0.48 with a normal that is nearly +Z but leaning outward -- a GRAZING
// normal. The top face's own edge is at z = 0.5 with a pure +Z normal. Two
// surfaces at nearly the same place with different normals = the dark rim.
//
// (a) is a placement question: does the mirrored sheet's wall start at the
// seam (z = 0, descending) or does it re-descend from +0.5? The mirrored
// sheet's local Eval returns z in [0, sz] descending; the scale z = -1
// maps [0, sz] -> [0, -sz]. So its wall spans z in [-sz, 0] -- correct.
// But its TOP FACE (the q <= 1 branch) maps z = +sz -> -sz! The mirrored
// sheet's "top face" is at z = -0.5 -- which is the BOTTOM face, correct
// for the closed body... but its WALL band descends from -0.5 upward? No:
// the wall's z = sz*(1-o) maps to -sz*(1-o) = from -0.5 (o=0) UP to 0
// (o=1). So the mirrored sheet's wall RISES from the bottom face to the
// seam -- correct!
//
// So why does it poke out behind? MEASURE it: render the two sheets
// separately and compare their silhouettes.
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/io/PSFFile.h"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace DeliRium;

static std::unique_ptr<Element> CubeTree(bool mirrored) {
    auto root = std::make_unique<Element>();
    root->kind = "body";
    root->color = {0.549f, 0.549f, 0.549f};
    root->substance = std::make_unique<BoxSubstance>(0.5f, 0.5f, 0.5f);
    if (mirrored) {
        auto lower = std::make_unique<Element>();
        lower->kind = "box_lower";
        lower->substance = std::make_unique<BoxSubstance>(0.5f, 0.5f, 0.5f);
        lower->origin = {0.f, 0.f, 0.f};
        lower->scale  = {1.f, 1.f, -1.f};
        lower->color = {0.549f, 0.549f, 0.549f};
        root->AddChild(std::move(lower));
    }
    return root;
}

int main() {
    for (int mir = 0; mir <= 1; ++mir) {
        auto tree = CubeTree(mir == 1);
        std::string why;
        auto body = ElementSystem::BakeBody(*tree, 48, 1.0f, why);
        if (!body) { std::printf("mir=%d bake failed: %s\n", mir, why.c_str()); return 1; }
        float mn[3] = {1e30f,1e30f,1e30f}, mx[3] = {-1e30f,-1e30f,-1e30f};
        const size_t n = body->AnchorBlob().size();
        const auto& pool = body->AnchorsConst();
        for (size_t i = 0; i < n; ++i) {
            const Vec3 p = pool.Position(uint16_t(i));
            for (int a = 0; a < 3; ++a) {
                const float v = a == 0 ? p.x : (a == 1 ? p.y : p.z);
                if (v < mn[a]) mn[a] = v;
                if (v > mx[a]) mx[a] = v;
            }
        }
        std::printf("mir=%d  anchors %5zu  x[%+.4f,%+.4f] y[%+.4f,%+.4f] z[%+.4f,%+.4f]\n",
                    mir, n, mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);
    }
    return 0;
}
