// =============================================================================
// dbg_grid_waste - MEASURE how much of each substance's sampling grid lands
// on its actual support.
//
// G21 asks: the grid is a square, so an annular support (the torus) spends
// most of its cells on the hole. Before changing anything, MEASURE the waste
// per substance -- the answer decides whether a support-shaped grid is worth
// an interface change.
//
// The measurement: for each substance, sample the grid the baker would use
// (the sheet's own Domain, clipped to the canvas) and count how many samples
// land on the support (substance->Eval says valid) versus off it.
//
// The off-support samples are NOT free: the baker writes them as
// well-formed anchors at the bbox centre (they are the "out-of-support
// placeholders"), so they cost memory, they cost upload bandwidth, and --
// worse -- they are the 990-of-1024 anchors that made a rod's inertia 25x
// too small (fix: G9).
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace DeliRium;

namespace {

struct Waste { const char* name; int n; int in_support; int wasted; };

void measure(const char* name, const ISubstance& sub, int N, float extent) {
    // The baker's rule: the sheet's own domain, clipped to the canvas.
    float hx = extent * 0.5f, hy = extent * 0.5f;
    sub.Domain(hx, hy);
    if (hx < 1e-6f) hx = extent * 0.5f;
    if (hy < 1e-6f) hy = extent * 0.5f;
    if (hx > extent * 0.5f) hx = extent * 0.5f;
    if (hy > extent * 0.5f) hy = extent * 0.5f;

    int in_support = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float lx = ((i + 0.5f) / N - 0.5f) * (hx * 2.f);
            const float ly = ((j + 0.5f) / N - 0.5f) * (hy * 2.f);
            if (sub.Eval(lx, ly).valid) ++in_support;
        }
    }
    const int total = N * N;
    const int wasted = total - in_support;
    std::printf("  %-26s domain [%+.3f, %+.3f]  in-support %5d / %5d "
                "(%5.1f%%)  WASTED %5d (%5.1f%%)\n",
                name, hx, hy, in_support, total,
                100.0*in_support/total, wasted, 100.0*wasted/total);
}

} // namespace

int main() {
    const int N = 48;
    const float extent = 1.0f;
    std::printf("-- the sampling waste at grid %d, canvas %.1f --\n\n",
                N, extent);

    measure("plane (unbounded)", PlaneSubstance(), N, extent);
    measure("ellipsoid 0.5", EllipsoidSubstance(0.5f, 0.5f, 0.5f), N, extent);
    measure("ellipsoid 0.2", EllipsoidSubstance(0.2f, 0.2f, 0.2f), N, extent);
    measure("rod 0.5 x 0.02", EllipsoidSubstance(0.5f, 0.02f, 0.02f), N, extent);
    measure("torus 0.32/0.12", TorusSubstance(0.32f, 0.12f), N, extent);
    measure("torus 0.40/0.06", TorusSubstance(0.40f, 0.06f), N, extent);
    measure("gaussian", GaussianSystemSubstance(0.2f, 0.35f), N, extent);

    std::printf("\n-- the cost of the waste --\n");
    std::printf("  every wasted sample is written as a WELL-FORMED anchor at\n");
    std::printf("  the bbox centre (the out-of-support placeholder). So the\n");
    std::printf("  waste costs: memory, upload bandwidth, and -- because the\n");
    std::printf("  mass calculation counted them -- a WRONG inertia tensor.\n");
    std::printf("  (That was G9: 970 of 1024 anchors for a rod sat at one\n");
    std::printf("  point, and its inertia tensor was 25x too small.)\n");
    return 0;
}
