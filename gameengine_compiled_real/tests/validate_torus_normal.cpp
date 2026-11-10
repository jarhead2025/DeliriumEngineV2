// =============================================================================
// validate_torus_normal - GATE for the TORUS SUBSTANCE's normal and closure.
//
// THE BUG THIS EXISTS FOR (found by the user's observation that "only the
// forward-facing faces are displayed" when the camera orbits counter-
// clockwise):
//
//   TorusSubstance::Eval returned the normal (d*x, d*y, z). The TRUE outward
//   normal of a tube at (x, y, z) is (d/r * x, d/r * y, z) -- the raw form
//   over-weights the radial component by a factor of r, so the normal leans
//   toward the radial direction and stops agreeing with the surface.
//
//   MEASURED before the fix:
//     - the substance's normal disagreed with the true outward normal by up
//       to 0.758 in dot product;
//     - the baked anchor normals came out 91.4% ANTIPARALLEL to the view
//       (mean dot -0.7332).
//
//   WHY AN INVERTED NORMAL LOOKS LIKE A MISSING FACE: the march path
//   accumulates the normal over the 3x3x3 neighbourhood of surfels and
//   THEN flips the sum to face the camera. Flipping a WRONG sum does not
//   fix it -- and where the neighbourhood is mixed (near the silhouette,
//   where neighbours are few and grazing) the sum is dominated by the
//   wrong ones. So the surface went dark in bands.
//
//   The second half of the bug: a PSF body is a GRAPH z = f(x,y), so one
//   sheet can only ever describe the UPPER half of the tube. One sheet
//   alone is a half-torus -- an open shell with no underside. The full
//   tube needs TWO sheets (the upper one and a MIRRORED one), exactly like
//   the closed ellipsoid (validate_closed_body pins that rule).
//
// THE GATE asserts:
//   A) the substance's normal IS the true outward normal (worst dot ~ 1.0);
//   B) every sheet's baked normals point OUTWARD FROM ITS OWN HALF -- the
//      upper sheet's up, the lower sheet's down. (A single global "mostly
//      facing the camera" count would flag the CORRECT lower half as
//      broken -- the very mistake the original bug would have hidden.)
//   C) the baked z-span is the FULL tube: 2 * minor.
//
// It also prints the measurements, so a regression shows what changed.
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace DeliRium;

namespace {
int checks = 0, failures = 0;

void check(bool cond, const char* msg) {
    ++checks;
    std::printf("  %s %s\n", cond ? "OK  " : "FAIL", msg);
    if (!cond) ++failures;
}
} // namespace

int main() {
    const float major = 0.32f, minor = 0.12f;
    TorusSubstance torus(major, minor);

    // ---- A) the substance's normal vs the TRUE outward normal --------------
    //
    // The surface is F = (r - major)^2 + z^2 - minor^2 = 0. The gradient in
    // the (r, z) half-plane is (2(r-major), 2z) -- the outward direction of
    // the tube's CROSS-SECTION -- which rotates into 3D as
    // (d/r * x, d/r * y, z).
    double worst = 1.0;
    int n = 0;
    for (int i = 0; i <= 60; ++i) {
        for (int j = 0; j <= 60; ++j) {
            const float x = -0.44f + 0.88f * i / 60.0f;
            const float y = -0.44f + 0.88f * j / 60.0f;
            SubstancePoint sp = torus.Eval(x, y);
            if (!sp.valid) continue;
            const float r = std::sqrt(x * x + y * y);
            if (r < 1e-9f) continue;
            const float d = r - major;
            Vec3 t{ d * x / r, d * y / r, sp.p.z };
            const float l = std::sqrt(t.x*t.x + t.y*t.y + t.z*t.z);
            if (l < 1e-9f) continue;
            t.x /= l; t.y /= l; t.z /= l;
            const float dot = sp.n.x*t.x + sp.n.y*t.y + sp.n.z*t.z;
            if (dot < worst) worst = dot;
            ++n;
        }
    }
    std::printf("substance normal vs TRUE outward: worst dot = %.4f  (n=%d)\n",
                worst, n);
    // 0.9995 rather than 1.0: the normal is quantized to 8 bits per axis
    // before it reaches the record, so a small loss is expected. The BUG's
    // worst dot was 0.242, so the threshold is nowhere near either value.
    check(worst > 0.9995f,
          "the substance's normal IS the true outward normal of the tube");

    // ---- the baked body ----------------------------------------------------
    auto tree = ProceduralElementGenerator::Build("a torus ring");
    if (!tree) { std::printf("FAIL: the generator returned null\n"); return 1; }
    ModelOutput out;
    std::string why;
    if (!ElementSystem::Bake(*tree, 48, 1.0f, out, why)) {
        std::printf("FAIL: bake failed: %s\n", why.c_str());
        return 1;
    }
    std::printf("baked: %zu anchors, %zu surfaces\n",
                out.anchors.size(), out.surfaces.size());
    // A full torus is TWO sheets, so it must have twice the anchors of one.
    check(out.anchors.size() > 4000,
          "the procedural torus bakes TWO sheets (the full tube)");

    auto body = ElementSystem::BakeBody(*tree, 48, 1.0f, why);
    if (!body) { std::printf("FAIL: BakeBody failed: %s\n", why.c_str()); return 1; }
    const Vec3 bmn = body->BBoxMin(), bsp = body->BBoxSpan();

    // ---- B) the per-half normal orientation --------------------------------
    //
    // THE UPPER SHEET's normals point UP, the LOWER SHEET's point DOWN, and
    // both are CORRECT. What must not happen is a normal pointing the wrong
    // way FOR ITS OWN HALF.
    int upper_ok = 0, upper_bad = 0, lower_ok = 0, lower_bad = 0;
    for (const auto& a : out.anchors) {
        const float z = bmn.z + (float)a.pos[2] * (1.0f/65535.0f) * bsp.z;
        Vec3 nrm{ a.normal[0] * (2.f/255.f) - 1.f,
                   a.normal[1] * (2.f/255.f) - 1.f,
                   a.normal[2] * (2.f/255.f) - 1.f };
        const float l = std::sqrt(nrm.x*nrm.x + nrm.y*nrm.y + nrm.z*nrm.z);
        if (l < 1e-9f) continue;
        nrm.z /= l;
        if (z >  0.02f) { if (nrm.z > 0.f) ++upper_ok; else ++upper_bad; }
        if (z < -0.02f) { if (nrm.z < 0.f) ++lower_ok; else ++lower_bad; }
    }
    std::printf("per-half normals: upper %d ok / %d inverted, "
                "lower %d ok / %d inverted\n",
                upper_ok, upper_bad, lower_ok, lower_bad);
    check(upper_bad == 0,
          "the upper sheet's normals all point UP (outward from its half)");
    check(lower_bad == 0,
          "the lower sheet's normals all point DOWN (outward from its half)");

    // ---- C) the tube is FULL: the z-span is 2 * minor ----------------------
    float zmin = 1e9f, zmax = -1e9f;
    for (const auto& a : out.anchors) {
        const float z = bmn.z + (float)a.pos[2] * (1.0f/65535.0f) * bsp.z;
        zmin = std::min(zmin, z); zmax = std::max(zmax, z);
    }
    const float span = zmax - zmin;
    std::printf("anchor z range: [%+.4f, %+.4f]  span %.4f  "
                "(a full tube needs %.4f)\n",
                zmin, zmax, span, 2.0f * minor);
    // The BUG's span was 0.1109 -- exactly half. The tolerance covers the
    // 16-bit quantisation of the positions.
    check(std::fabs(span - 2.0f * minor) < 0.01f,
          "the baked tube's z-span is the FULL 2 * minor");

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
