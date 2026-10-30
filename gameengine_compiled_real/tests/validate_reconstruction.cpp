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
        const int np = (grid_n-1)*(grid_n-1), na = grid_n*grid_n;
        check((int)mo.surfaces.size() == np, "B: patch count == (n-1)^2");
        check((int)mo.anchors.size() == na, "B: anchor count == n^2");
        std::string vwhy;
        check(ReconstructionModel::Validate(mo, vwhy), "B: Validate() passes", vwhy.c_str());
    }

    // ---- C) materialize to a live body -------------------------------------
    auto body = ElementSystem::BakeBody(*tree, grid_n, extent, why);
    check((bool)body, "C: BakeBody() returns a body", why.c_str());
    if (body) {
        check((int)body->SurfaceBlob().size() == (grid_n-1)*(grid_n-1), "C: body patch count");
        check((int)body->AnchorBlob().size() == grid_n*grid_n, "C: body anchor count");
        check(body->BBoxSpan().x > 0.f && body->BBoxSpan().y > 0.f, "C: bbox span sane");
    }

    // ---- D) fidelity: body surface ~= aggregate field ----------------------
    if (body) {
        float worst = 0.f; int n = 0;
        for (int j = 0; j < grid_n; ++j)
            for (int i = 0; i < grid_n; ++i) {
                const float u = i / float(grid_n-1), v = j / float(grid_n-1);
                const Vec3 world{(u-0.5f)*extent, (v-0.5f)*extent, 0.f};
                const Vec3 target = ElementSystem::FieldAt(*tree, world).point;
                const Vec3 got = body->ClosestPoint(target);
                worst = std::max(worst, dist(target, got));
                ++n;
            }
        const float diag = std::sqrt(3.f) * extent;
        std::printf("      (worst closest-point residual %.5f over %d samples, diag=%.3f)\n",
                    worst, n, diag);
        check(worst < 0.02f * diag, "D: body reproduces field < 2% diag");
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
    {
        auto orb = std::make_unique<Element>();
        orb->kind = "ellipsoid_body";
        orb->substance = std::make_unique<EllipsoidSubstance>(0.5f, 0.3f, 0.4f);
        std::string owhy;
        auto obody = ElementSystem::BakeBody(*orb, 16, 1.0f, owhy);
        check((bool)obody, "F: ellipsoid bakes to a body", owhy.c_str());
        if (obody) {
            // sample the analytic hemisphere (z = sz*sqrt(1-x^2/sx^2-y^2/sy^2))
            // and check the body is near it
            float worst = 0.f;
            for (int a = 0; a < 24; ++a)
                for (int b = 1; b < 12; ++b) {
                    const float th = 2.f * 3.14159265f * a / 24.f;
                    const float r  = 0.9f * b / 12.f;   // radial fraction (avoid the rim)
                    const float x  = 0.5f * r * std::cos(th);
                    const float y  = 0.3f * r * std::sin(th);
                    const float z  = 0.4f * std::sqrt(1.0f - r * r);
                    const Vec3 surf{x, y, z};
                    worst = std::max(worst, dist(surf, obody->ClosestPoint(surf)));
                }
            std::printf("      (worst ellipsoid residual %.5f)\n", worst);
            check(worst < 0.06f, "F: body near analytic ellipsoid (<0.06)");
        }
    }

    std::printf(g_fails ? "RECONSTRUCTION VALIDATION FAILED (%d)\n"
                        : "RECONSTRUCTION VALIDATION PASSED\n", g_fails);
    return g_fails ? 1 : 0;
}