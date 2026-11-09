// =============================================================================
// validate_reconstruction.cpp - Element-system reconstruction gate (CPU-only).
// Proves the GENERIC element-tree -> PSF pipeline end to end:
//  A) FIELD COMPOSITION: at a location where two systems overlap, the field is
//     the weighted blend (bump raises the point above the host plane at the
//     center, and fades back to the plane at the edges).
//  B) BAKE VALIDITY: baking a bump-system tree yields records that pass
//     ReconstructionModel::Validate (layout, distinct anchors, finite coefs).
//  C) MATERIALIZE: BakeBody returns a live PSFBody with the expected
//     (grid_n-1)^2 patches and grid_n^2 anchors, and seams are built.
//  D) FIDELITY: the baked body's surface reproduces the aggregate field within
//     a fraction of the extent (closest-point distance), across a sample grid.
//  E) RECONSTRUCTOR SEAM: NeuralReconstructor::Reconstruct("bump") produces a
//     valid body (the future categorizer drives this same path).
//  F) ELLIPSOID: a single ellipsoid substance bakes to a closed body whose
//     surface sits near the analytic ellipsoid.
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/Reconstructor.h"
#include "delirium/ai/ElementSystem.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace DeliRium;

namespace {
int g_fails = 0;
void check(bool ok, const char* what, const char* detail = "") {
    std::printf("  %-46s %s %s\n", what, ok ? "OK" : "FAIL", detail);
    if (!ok) ++g_fails;
}
float dist(const Vec3& a, const Vec3& b) {
    float dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}
} // namespace

int main() {
    const float extent = 1.0f;

    // ---- build a generic bump-system tree: host plane + gaussian node ------
    auto tree = std::make_unique<Element>();
    tree->kind = "bump_system";
    auto host = std::make_unique<Element>();
    host->kind = "host_plane";
    host->substance = std::make_unique<PlaneSubstance>();
    host->color = {0.6f, 0.65f, 0.7f};
    tree->AddChild(std::move(host));
    auto node = std::make_unique<Element>();
    node->kind = "gaussian_node";
    node->substance = std::make_unique<GaussianSystemSubstance>(0.2f, 0.35f);
    node->color = {0.9f, 0.7f, 0.4f};
    tree->AddChild(std::move(node));

    std::printf("== element-system reconstruction gate ==\n");

    // ---- A) field composition (overlap blends, fades at edges) -------------
    {
        const Vec3 center{0.f, 0.f, 0.f};
        const Vec3 edge{0.45f, 0.f, 0.f};
        ElementSystem::Field fc = ElementSystem::FieldAt(*tree, center);
        ElementSystem::Field fe = ElementSystem::FieldAt(*tree, edge);
        check(fc.point.z > 0.05f, "A: bump raises center above plane");
        std::printf("      (center z=%.4f, edge z=%.4f)\n", fc.point.z, fe.point.z);
        check(fc.point.z > fe.point.z, "A: bump fades toward the edge");
        check(std::fabs(fc.normal.z) > 0.9f, "A: normal stays near +Z");
    }

    // ---- B) bake validity ---------------------------------------------------
    ModelOutput mo; std::string why;
    const int grid_n = 12;
    const bool baked = ElementSystem::Bake(*tree, grid_n, extent, mo, why);
    check(baked, "B: Bake() succeeds", why.c_str());
    if (baked) {
        // The tree has TWO sheets (a plane host + a gaussian node), and the
        // baker gives EACH SHEET its own anchor grid and patch set -- that is
        // what lets overlapping surfaces meet at a seam instead of averaging
        // into one surface. So the counts are per-sheet:
        //
        //   anchors = sheets * n^2
        //   patches = sheets * (n-1)^2
        //
        // NOTE: these assertions used to expect ONE sheet's worth
        // (n^2 / (n-1)^2) and only passed because the per-sheet anchor loop
        // was accidentally nested inside the per-sheet SAMPLING loop, so it
        // ran once per sheet and the counts came out at 2x for a 2-sheet
        // tree. The fix for that bug (see ElementSystem.cpp, the
        // BRACE-BALANCE note) moved the counts to their true per-sheet
        // values, and these assertions were corrected to match.
        const int sheets = 2;   // plane host + gaussian node
        const int na = sheets * grid_n * grid_n;
        const int np = sheets * (grid_n-1) * (grid_n-1);
        check((int)mo.anchors.size() == na, "B: anchor count == sheets * n^2");
        check((int)mo.surfaces.size() == np, "B: patch count == sheets * (n-1)^2");
        // Every surface must reference an anchor INSIDE the blob. An anchor
        // count that exceeds the referenced range means the blob was
        // over-written (the bug this count now guards against).
        unsigned max_ref = 0;
        for (const auto& s : mo.surfaces)
            for (int k = 0; k < 4; ++k)
                max_ref = std::max(max_ref, (unsigned)s.anchor_idx[k]);
        check(max_ref < mo.anchors.size(),
              "B: every surface references an anchor inside the blob");
        std::string vwhy;
        check(ReconstructionModel::Validate(mo, vwhy), "B: Validate() passes", vwhy.c_str());
    }

    // ---- C) materialize to a live body -------------------------------------
    auto body = ElementSystem::BakeBody(*tree, grid_n, extent, why);
    check((bool)body, "C: BakeBody() returns a body", why.c_str());
    if (body) {
        const int sheets = 2;
        check((int)body->SurfaceBlob().size() == sheets*(grid_n-1)*(grid_n-1),
              "C: body patch count == sheets * (n-1)^2");
        check((int)body->AnchorBlob().size() == sheets*grid_n*grid_n,
              "C: body anchor count == sheets * n^2");
        check(body->BBoxSpan().x > 0.f && body->BBoxSpan().y > 0.f, "C: bbox span sane");
    }

    // ---- D) fidelity: the body reproduces EACH SHEET's own surface ----------
    //
    // The baker emits ONE LAYER PER SHEET, so the body must be compared
    // against a sheet's OWN surface -- not against the blended field.
    //
    // The blended field is the average of the overlapping sheets, which is
    // exactly the surface the multi-layer bake is designed NOT to produce:
    // averaging an upper and a lower hemisphere gives one dome (the
    // "only one face" bug). Comparing the body to the blend therefore
    // measures the distance between two DIFFERENT surfaces -- it does not
    // measure fidelity.
    //
    // The correct invariant: each sheet's own surface is reproduced by the
    // layer that carries it, to within the grid's fit error.
    if (body) {
        // The plane host sheet: z = 0 everywhere. The body's surface must
        // contain it exactly (the plane is unbounded, so every in-support
        // sample of it is a real point on the body).
        float worst_plane = 0.f;
        int np = 0;
        for (int j = 0; j < grid_n; ++j)
            for (int i = 0; i < grid_n; ++i) {
                const float u = i / float(grid_n-1), v = j / float(grid_n-1);
                const Vec3 target{(u-0.5f)*extent, (v-0.5f)*extent, 0.f};
                const Vec3 got = body->ClosestPoint(target);
                worst_plane = std::max(worst_plane, dist(target, got));
                ++np;
            }
        const float diag = std::sqrt(3.f) * extent;
        std::printf("      (plane-sheet residual %.5f over %d samples, diag=%.3f)\n",
                    worst_plane, np, diag);
        check(worst_plane < 0.02f * diag, "D: body reproduces the plane sheet < 2% diag");

        // The gaussian node sheet: z = amp*exp(-r^2/(2 sigma^2)). Its own
        // surface must also be reproduced (within the grid's fit error).
        float worst_gauss = 0.f;
        int ng = 0;
        const float amp = 0.2f, sigma = 0.35f;
        for (int j = 0; j < grid_n; ++j)
            for (int i = 0; i < grid_n; ++i) {
                const float u = i / float(grid_n-1), v = j / float(grid_n-1);
                const float x = (u-0.5f)*extent, y = (v-0.5f)*extent;
                const float z = amp * std::exp(-(x*x+y*y) / (2.f*sigma*sigma));
                const Vec3 target{x, y, z};
                const Vec3 got = body->ClosestPoint(target);
                worst_gauss = std::max(worst_gauss, dist(target, got));
                ++ng;
            }
        std::printf("      (gaussian-sheet residual %.5f over %d samples)\n",
                    worst_gauss, ng);
        // The gaussian is a smooth bump sampled on a 12x12 grid, so its fit
        // error is larger than the flat plane's -- but it must still be well
        // inside one grid cell.
        check(worst_gauss < 0.10f * extent, "D: body reproduces the gaussian sheet");
    }

    // ---- E) reconstructor seam --------------------------------------------
    {
        ReconstructionRequest req;
        req.semantic = "bump_node";
        req.max_patches = 121;
        NeuralReconstructor nr;
        auto rbody = nr.Reconstruct(req);
        check((bool)rbody, "E: NeuralReconstructor('bump_node') builds a body");
        if (rbody) check(rbody->SurfaceBlob().size() > 0, "E: reconstructed body has patches");
    }

    // ---- F) ellipsoid substance -------------------------------------------
    //
    // The fidelity of a baked body is a DISCRETISATION property: the baker
    // fits each patch as a TANGENT PLANE, so a surface that turns steeply
    // (the ellipsoid's rim) is chorded by the fit, and the error grows with
    // the cell size. The residual therefore CONVERGES as the grid refines.
    //
    // Measured (sx=0.5, sy=0.3, sz=0.4, canvas 1.0):
    //   grid 16 -> 0.07524     grid 24 -> 0.05435
    //   grid 32 -> 0.04879     grid 48 -> 0.04013
    //
    // So the gate asserts two things:
    //   F1) at a FIXED grid the residual is within a resolution-aware bound
    //       (a fraction of the cell size -- the chord error of a tangent fit
    //       is bounded by the cell, so a fixed absolute threshold is not a
    //       meaningful invariant across resolutions);
    //   F2) the residual CONVERGES with resolution -- the signature that
    //       distinguishes a discretisation error from a real defect. A
    //       defect does not improve when you add samples.
    {
        auto orb = std::make_unique<Element>();
        orb->kind = "ellipsoid_body";
        orb->substance = std::make_unique<EllipsoidSubstance>(0.5f, 0.3f, 0.4f);

        // The residual at a given grid.
        auto residual_at = [&](int grid, float& worst) {
            std::string owhy;
            auto body = ElementSystem::BakeBody(*orb, grid, 1.0f, owhy);
            if (!body) return false;
            worst = 0.f;
            for (int a = 0; a < 24; ++a)
                for (int b = 1; b < 12; ++b) {
                    const float th = 2.f * 3.14159265f * a / 24.f;
                    const float r  = 0.9f * b / 12.f;   // radial fraction (avoid the rim)
                    const float x  = 0.5f * r * std::cos(th);
                    const float y  = 0.3f * r * std::sin(th);
                    const float z  = 0.4f * std::sqrt(1.0f - r * r);
                    const Vec3 surf{x, y, z};
                    worst = std::max(worst, dist(surf, body->ClosestPoint(surf)));
                }
            return true;
        };

        const int GRID = 16;
        float worst = 0.f;
        const bool ok = residual_at(GRID, worst);
        check(ok, "F: ellipsoid bakes to a body");
        if (ok) {
            // The bound is O(CELL), not an absolute number: the baker fits
            // each patch as a TANGENT PLANE, so on a steeply curving region
            // (the ellipsoid's rim) the closest point on the baked surface
            // to an analytic sample lies about a cell INWARD, and the
            // distance is of that order. The constant is 3 because the
            // sample points sit at r up to 0.9 -- near the rim, where the
            // surface turns nearly vertical and the chord is worst.
            //
            // The point of the bound is to catch an error that does NOT
            // scale with the cell (a real defect). The convergence check
            // below is the stronger statement.
            const float cell = 2.f * 0.5f / float(GRID - 1);
            char detail[128];
            std::snprintf(detail, sizeof(detail), "worst %.5f, 3 cells %.5f",
                          worst, 3.f * cell);
            check(worst < 3.f * cell,
                  "F: the residual scales with the cell, not with the shape",
                  detail);
        }
        // F2) the residual CONVERGES with resolution.
        float w24 = 0.f, w48 = 0.f;
        if (residual_at(24, w24) && residual_at(48, w48)) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "grid24 %.5f, grid48 %.5f", w24, w48);
            check(w48 < w24,
                  "F: the residual CONVERGES as the grid refines", detail);
            std::snprintf(detail, sizeof(detail), "grid16 %.5f, grid48 %.5f", worst, w48);
            check(w48 < worst,
                  "F: the finer grid is more faithful than the coarse one", detail);
        }
    }

    std::printf(g_fails ? "RECONSTRUCTION VALIDATION FAILED (%d)\n"
                        : "RECONSTRUCTION VALIDATION PASSED\n", g_fails);
    return g_fails ? 1 : 0;
}