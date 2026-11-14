// Domain probe: does the baker's sampling ever visit the wall band?
//
// The cube diag showed the bake has ZERO wall anchors: top 2304, bottom
// 2304, SIDE-WALL 0. BoxSubstance::Eval models the walls in the band
// q > 1 (outside the footprint), but BoxSubstance::Domain() returns
// exactly [sx, sy] -- the FOOTPRINT only. The sampling domain must cover
// the wall band or the wall code is dead.
//
// This probe prints, for each substance, the domain the baker actually
// samples and whether that domain reaches the wall band.
#include "delirium/ai/ElementSystem.h"

#include <cstdio>

using namespace DeliRium;

int main() {
    BoxSubstance box(0.5f, 0.5f, 0.5f);
    float hx = 1e30f, hy = 1e30f;
    box.Domain(hx, hy);
    std::printf("BoxSubstance(0.5,0.5,0.5) domain: half_x %.4f half_y %.4f\n", hx, hy);
    std::printf("  the wall band is |x| or |y| in (0.5, 0.5 + wall*reach]\n");
    std::printf("  the domain covers the wall band: %s\n",
                (hx > 0.5001f) ? "YES" : "NO -- the wall code is UNREACHABLE");
    // Eval at a point in the wall band, to show the substance CAN describe
    // it.
    float test_x = 0.55f;                       // outside the footprint
    SubstancePoint sp = box.Eval(test_x, 0.1f);
    std::printf("  Eval(%.2f, 0.10): valid=%d  p=(%.4f, %.4f, %.4f)  n=(%.3f, %.3f, %.3f)\n",
                test_x, int(sp.valid), sp.p.x, sp.p.y, sp.p.z, sp.n.x, sp.n.y, sp.n.z);
    std::printf("    (a valid wall point with a horizontal normal proves the\n");
    std::printf("     substance models the wall -- the DOMAIN is what fails)\n");
    return 0;
}
