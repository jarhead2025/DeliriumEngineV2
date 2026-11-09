// =============================================================================
// torus_probe.cpp - DIAGNOSTIC probe for the torus render defects.
//
// Answers four questions with data, not theory:
//   1. CONVENTION: does the baker write anchor_idx in the SAME corner order
//      the renderer reads? (baker writes a00,a10,a11,a01 = B order; the
//      renderer reads convention A: u along A0->A3, v along A0->A1)
//   2. DEBRIS: how many anchors are the out-of-support DEFAULT (all-zero
//      normal -> the kernel normalizes a zero vector -> garbage samples),
//      and do any SURFACES reference them?
//   3. SHREDDING: how many grid cells were SKIPPED because a corner was out
//      of support, and how are they distributed?
//   4. FIDELITY: closest-point distance from the analytic torus surface to
//      the baked body.
//
// A probe, not a gate: it prints the numbers that identify the defect.
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/ai/ElementSystem.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace DeliRium;

int main() {
    const float extent = 1.0f;
    const int grid_n = 48;

    auto tree = std::make_unique<Element>();
    tree->kind = "torus_body";
    tree->substance = std::make_unique<TorusSubstance>(0.32f, 0.12f);
    tree->color = {0.7f, 0.6f, 0.5f};

    ModelOutput mo; std::string why;
    if (!ElementSystem::Bake(*tree, grid_n, extent, mo, why)) {
        std::printf("BAKE FAILED: %s\n", why.c_str());
        return 1;
    }

    std::printf("== torus bake probe (grid_n=%d) ==\n", grid_n);
    std::printf("  anchors=%zu  surfaces=%zu\n", mo.anchors.size(), mo.surfaces.size());

    // --- 1. DEGENERATE ANCHORS: the out-of-support default -------------------
    // The baker writes pos = (32767,32767,32767) and normal = (128,128,255)
    // for an out-of-support sample. The normal is +Z (valid), but the POSITION
    // is the bbox centre -- a point in the middle of the torus's HOLE.
    size_t degenerate = 0;
    std::vector<uint8_t> is_deg(mo.anchors.size(), 0);
    for (size_t i = 0; i < mo.anchors.size(); ++i) {
        const AnchorRecord& a = mo.anchors[i];
        if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767) {
            ++degenerate;
            is_deg[i] = 1;
        }
    }
    std::printf("  out-of-support (degenerate) anchors: %zu / %zu  (%.1f%%)\n",
                degenerate, mo.anchors.size(),
                100.0 * degenerate / (double)mo.anchors.size());

    // Do any SURFACES reference a degenerate anchor?
    size_t refs = 0, patches_with_deg = 0;
    for (const SurfaceRecord& s : mo.surfaces) {
        bool any = false;
        for (int k = 0; k < 4; ++k) {
            const uint16_t a = s.anchor_idx[k];
            if (a < is_deg.size() && is_deg[a]) { ++refs; any = true; }
        }
        if (any) ++patches_with_deg;
    }
    std::printf("  surface corners referencing a degenerate anchor: %zu\n", refs);
    std::printf("  patches containing a degenerate corner: %zu\n", patches_with_deg);

    // --- 2. CONVENTION CHECK -------------------------------------------------
    // The renderer (Patch::EvaluateDerivs) reads:
    //     pl = lerp(P0,P3,u);  pr = lerp(P1,P2,u);  Pb = lerp(pl,pr,v)
    // -> A0=(0,0) A1=(0,1) A2=(1,1) A3=(1,0)
    // The baker writes:
    //     idx = {a00, a10, a11, a01} = {(i,j), (i+1,j), (i+1,j+1), (i,j+1)}
    // -> slot0=(0,0) slot1=(1,0) slot2=(1,1) slot3=(0,1)
    // These DIFFER: slot1 and slot3 are transposed.
    //
    // Detect it WITHOUT relying on the comment: take a patch and check whether
    // the renderer's own Evaluate reproduces the patch's own anchor positions.
    // At (u=0,v=0) the renderer MUST return anchor 0's position (plus h*Nd,
    // and h vanishes at the corners, so exactly anchor 0's position).
    // At (u=1,v=0) it MUST return anchor 3's position (convention A).
    // At (u=0,v=1) it MUST return anchor 1's position (convention A).
    //
    // So: if the renderer's (1,0) corner matches slot 3, the record is in
    // convention A; if it matches slot 1, it is in convention B (transposed).
    std::unique_ptr<PSFBody> body = ElementSystem::BakeBody(*tree, grid_n, extent, why);
    if (!body) { std::printf("BAKEBODY FAILED: %s\n", why.c_str()); return 1; }

    const auto& ab = body->AnchorBlob();
    const auto& sb = body->SurfaceBlob();
    const Vec3 bmin = body->BBoxMin();
    const Vec3 bspan = body->BBoxSpan();
    auto deq = [&](uint16_t ai) {
        const AnchorRecord& r = ab[ai];
        return Vec3{ bmin.x + r.pos[0] / 65535.f * bspan.x,
                     bmin.y + r.pos[1] / 65535.f * bspan.y,
                     bmin.z + r.pos[2] / 65535.f * bspan.z };
    };

    int agree_A = 0, agree_B = 0, examined = 0;
    for (size_t s = 0; s < sb.size(); ++s) {
        if (examined >= 400) break;
        const SurfaceRecord& r = sb[s];
        Patch P(const_cast<SurfaceRecord*>(&r), &body->Anchors(),
                nullptr, 0.f, uint32_t(s));
        Vec3 p00, p10, p01, du, dv;
        P.EvaluateDerivs(0.f, 0.f, p00, du, dv);
        P.EvaluateDerivs(1.f, 0.f, p10, du, dv);
        P.EvaluateDerivs(0.f, 1.f, p01, du, dv);
        const Vec3 a0 = deq(r.anchor_idx[0]);
        const Vec3 a1 = deq(r.anchor_idx[1]);
        const Vec3 a3 = deq(r.anchor_idx[3]);
        auto near = [](const Vec3& x, const Vec3& y) {
            const float dx=x.x-y.x, dy=x.y-y.y, dz=x.z-y.z;
            return std::sqrt(dx*dx+dy*dy+dz*dz);
        };
        if (near(p00, a0) > 1e-3f) continue;      // degenerate corner; skip
        ++examined;
        if (near(p10, a3) < near(p10, a1)) ++agree_A;
        else                               ++agree_B;
    }
    std::printf("  CONVENTION: renderer corner (1,0) matches slot3 in %d, slot1 in %d"
                "  -> %s\n", agree_A, agree_B,
                (agree_A > agree_B) ? "CONVENTION A (renderer & baker agree)"
                                    : "CONVENTION B (BAKER IS TRANSPOSED vs RENDERER)");

    // --- 3. SHREDDING: which grid cells were skipped ------------------------
    // A cell is skipped when ANY of its 4 corners is out of support.
    int skipped = 0, total_cells = 0;
    for (size_t s = 0; s < sb.size(); ++s) {
        ++total_cells;
        (void)s;
    }
    const int npatch = grid_n - 1;
    skipped = npatch * npatch - int(total_cells);
    std::printf("  grid cells %dx%d = %d; patches emitted = %d; SKIPPED = %d (%.1f%%)\n",
                npatch, npatch, npatch * npatch, total_cells, skipped,
                100.0 * skipped / (double)(npatch * npatch));

    // --- 4. FIDELITY: analytic torus vs baked body --------------------------
    // The analytic UPPER tube: r = |xy|, z = +sqrt(minor^2 - (r-major)^2).
    float worst = 0.f; int ns = 0;
    for (int j = 0; j < 40; ++j) {
        for (int i = 0; i < 40; ++i) {
            const float x = (-0.44f + 0.88f * i / 39.f);
            const float y = (-0.44f + 0.88f * j / 39.f);
            const float r = std::sqrt(x*x + y*y);
            const float d = r - 0.32f;
            const float inside = 0.12f*0.12f - d*d;
            if (inside <= 0.f) continue;            // outside the tube
            const Vec3 target{x, y, std::sqrt(inside)};
            const Vec3 got = body->ClosestPoint(target);
            const float dx=target.x-got.x, dy=target.y-got.y, dz=target.z-got.z;
            const float e = std::sqrt(dx*dx+dy*dy+dz*dz);
            if (e > worst) worst = e;
            ++ns;
        }
    }
    std::printf("  fidelity: worst closest-point error %.5f over %d tube samples"
                " (grid step ~%.4f)\n",
                worst, ns, 0.88f / (grid_n - 1));

    // --- 5. IS THE HOLE OPEN? sample the analytic hole axis -----------------
    // Points on the Z axis inside the hole (r < major - minor) must have NO
    // surface nearby: the nearest surface point must be the tube's inner rim.
    for (const Vec3 axis : { Vec3{0,0,0.05f}, Vec3{0,0,0.15f}, Vec3{0,0,0.30f} }) {
        const Vec3 got = body->ClosestPoint(axis);
        const float dx=axis.x-got.x, dy=axis.y-got.y, dz=axis.z-got.z;
        std::printf("  hole axis (0,0,%.2f) -> nearest surface (%.3f,%.3f,%.3f)  dist %.4f\n",
                    axis.z, got.x, got.y, got.z,
                    std::sqrt(dx*dx+dy*dy+dz*dz));
    }
    return 0;
}
