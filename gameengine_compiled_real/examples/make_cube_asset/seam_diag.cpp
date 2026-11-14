// SEAM DIAG: measure the torus's equator serration at the record level.
//
// The cube work proved the shading/normals/winding are correct. The
// remaining artifact is the SERRATED TEETH at the seam -- so this names
// the seam's geometry: how the two sheets' rim rows meet, what the seam
// anchors' positions/normals look like, and whether the patches at the
// seam are degenerate (zero-area) or overlapping.
//
// The torus's seam is at z = 0 (the two sheets' rim annulus). The
// serration (from the renders) is a jagged tooth pattern running along
// the rim.
#include "delirium/psf/PSFBody.h"
#include "delirium/ai/ElementSystem.h"
#include <cstdio>
#include <cmath>
#include <memory>

using namespace DeliRium;

int main() {
    // Bake the procedural torus exactly as the workspace does.
    const float major = 0.32f, minor = 0.12f;
    auto root = std::make_unique<Element>();
    root->kind = "torus";
    root->substance = std::make_unique<TorusSubstance>(major, minor);
    auto lower_sheet = std::make_unique<Element>();
    lower_sheet->kind = "torus_lower";
    lower_sheet->substance = std::make_unique<TorusSubstance>(major, minor);
    lower_sheet->origin = {0, 0, 0};
    lower_sheet->scale = {1.f, 1.f, -1.f};
    root->AddChild(std::move(lower_sheet));

    std::string why;
    auto body = ElementSystem::BakeBody(*root, 48, 1.0f, why);
    if (!body) { std::printf("bake failed: %s\n", why.c_str()); return 1; }
    std::printf("baked: %zu anchors, %zu surfaces\n",
                body->AnchorBlob().size(), body->SurfaceBlob().size());

    const size_t n = body->AnchorBlob().size();
    const auto& pool = body->AnchorsConst();
    const auto& SB = body->SurfaceBlob();

    // 1) THE SEAM ANCHORS: how many sit at z ~ 0, and what is their
    //    radial distribution? A serration implies the rim points are
    //    NOT on a clean circle -- or that two rows interleave.
    int seam = 0;
    float rmin = 1e30f, rmax = -1e30f;
    std::vector<float> angles;
    for (size_t i = 0; i < n; ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        if (std::fabs(p.z) > 0.005f) continue;   // only the seam band
        ++seam;
        const float r = std::sqrt(p.x * p.x + p.y * p.y);
        if (r < rmin) rmin = r;
        if (r > rmax) rmax = r;
        angles.push_back(std::atan2(p.y, p.x));
    }
    std::printf("seam anchors (|z| <= 0.005): %d  r range [%.4f, %.4f]\n",
                seam, rmin, rmax);

    // 2) Radial spread of the seam band: a clean seam is ONE circle; a
    //    serrated one is two interleaved radii (upper rim + lower rim
    //    landing at slightly different r).
    // Histogram the radii into 20 bins.
    int bins[20] = {0};
    const float span = rmax - rmin;
    for (float a : angles) { (void)a; }
    std::vector<float> radii;
    for (size_t i = 0; i < n; ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        if (std::fabs(p.z) > 0.005f) continue;
        radii.push_back(std::sqrt(p.x * p.x + p.y * p.y));
    }
    for (float r : radii) {
        int b = int((r - rmin) / (span + 1e-9f) * 20);
        if (b < 0) b = 0; if (b > 19) b = 19;
        ++bins[b];
    }
    std::printf("seam radial histogram (20 bins over %.4f):\n", span);
    for (int b = 0; b < 20; ++b)
        if (bins[b]) std::printf("  r %.4f-%.4f : %d\n",
                                 rmin + span * b / 20, rmin + span * (b + 1) / 20, bins[b]);

    // 3) SEAM PATCHES: how many patches have corners on BOTH sides of
    //    the seam (a degenerate bridge) vs cleanly on one sheet?
    int both = 0, upper = 0, lower = 0;
    for (size_t s = 0; s < SB.size(); ++s) {
        const uint16_t* ai = SB[s].anchor_idx;
        int nu = 0, nl = 0;
        for (int k = 0; k < 4; ++k) {
            const Vec3 p = pool.Position(ai[k]);
            if (p.z >= -1e-4f) ++nu; else ++nl;
        }
        if (nu == 4) ++upper;
        else if (nl == 4) ++lower;
        else ++both;
    }
    std::printf("patches: all-upper %d  all-lower %d  STRADDLING %d\n",
                upper, lower, both);

    // 4) The seam's angular coverage: the torus uses a POLAR grid, so the
    //    angular axis wraps. If the last column's anchors coincide with
    //    the first column's (a duplicated endpoint), the seam has
    //    doubled anchors at one angle -- a tooth source. Count anchors
    //    within one grid-step of another at the same angle.
    // Approximate: count exact-duplicate positions in the seam band.
    int dups = 0;
    for (size_t i = 0; i < radii.size(); ++i)
        for (size_t j = i + 1; j < radii.size(); ++j)
            if (std::fabs(radii[i] - radii[j]) < 1e-5f) ++dups;
    std::printf("seam anchors sharing a radius (within 1e-5): %d\n", dups);
    return 0;
}
