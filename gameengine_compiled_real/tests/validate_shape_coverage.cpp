// =============================================================================
// validate_shape_coverage - GATE for the MUTUAL-SURROUND DERIVATION over a
// RANGE OF BASIC GEOMETRIC SHAPES.
//
// WHY THIS GATE EXISTS (Josias, 2026-11-11): "lets run a full object
// coverage and test it with basic geometric shapes".
//
// The derivation was developed against ONE shape -- an ellipsoid -- and one
// point cloud. A derivation that works for one convex dome says nothing
// about a shape with a HOLE (the torus), a shape with a hard BOUNDARY (the
// plane, the gaussian bump's footprint), or a shape that is NOT CONVEX.
// Each of those breaks a different assumption in the pipeline:
//
//   - the TORUS's support is an annulus: the cloud has a hole in it, so
//     the triangulations' convex hulls cover the hole, and the cells that
//     span it are not on the surface at all.
//   - the PLANE is unbounded: the "surface" is only as big as the domain
//     we sampled, and its area is the domain's area.
//   - the GAUSSIAN BUMP has a soft boundary: the cloud thins out, so the
//     boundary cells are the ones most likely to be unpaired.
//   - a ROD (a very thin ellipsoid) is the worst case for a point cloud:
//     the cloud is a thin ribbon, so most of its vertices are on the
//     boundary.
//
// WHAT IS ASSERTED, for each shape:
//
//   A) the derivation SUCCEEDS and produces cells
//   B) no DEGENERATE cell (four distinct corners)
//   C) no INSIDE-OUT cell (the winding agrees with the surface normal)
//   D) the cells CONNECT (the mesh is a surface, not a bag of quads)
//   E) the two meshes' areas are each a substantial fraction of the
//      surface's true area -- this is the COVERAGE claim, and it is the
//      property that says the derivation is not throwing geometry away
//   F) the derivation is DETERMINISTIC
//
// THE REFERENCE AREA is computed two ways, and they must agree:
//   - NUMERICALLY: the placed grid's own cell areas, summed. This is what
//     the baker would produce, so it is the honest "did we keep the
//     geometry" comparison.
//   - ANALYTICALLY, where a closed form exists (the torus, the sphere):
//     the upper half of a torus of centre radius R and tube radius r is
//     2*pi^2*R*r; the upper hemisphere of a sphere of radius r is
//     2*pi*r^2. If the numeric reference disagrees with the analytic one,
//     the REFERENCE is wrong -- so this check catches a broken yardstick,
//     not just a broken subject.
// =============================================================================
#include "delirium/ai/MutualMesh.h"
#include "delirium/ai/ElementSystem.h"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace DeliRium;

namespace {

int checks = 0, failures = 0;
std::string current_shape;

void check(bool cond, const char* fmt, ...) {
    ++checks;
    char msg[512];
    va_list ap; va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    std::printf("  %s [%s] %s\n", cond ? "OK  " : "FAIL", current_shape.c_str(), msg);
    if (!cond) ++failures;
}

// Sample a substance into a point cloud -- the "bunch of dots".
//
// THE CLOUD IS SAMPLED FROM THE SUBSTANCE'S OWN DOMAIN, not from a fixed
// square. A torus's domain is the disc of radius major+minor, and its
// support is the annulus inside it -- so the cloud has a hole, which is
// exactly the case this gate exists to test.
std::vector<MeshPoint> SampleCloud(const ISubstance& sub, int N) {
    float hx = 0.5f, hy = 0.5f;
    sub.Domain(hx, hy);
    std::vector<MeshPoint> pts;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float x = ((i + 0.5f) / N - 0.5f) * (hx * 2.f);
            const float y = ((j + 0.5f) / N - 0.5f) * (hy * 2.f);
            const SubstancePoint sp = sub.Eval(x, y);
            if (!sp.valid) continue;
            pts.push_back({ sp.p, sp.n, 1.f });
        }
    return pts;
}

// The area of a quad, split along the diagonal that gives the LARGER total.
// A quad built from a triangle pair has its diagonal on the SHARED edge,
// and splitting along the wrong diagonal measures a much larger area --
// which is how a "176% of the hemisphere" appeared once.
double QuadArea(const MeshQuad& q, const std::vector<MeshPoint>& pts) {
    auto tri = [&](int a, int b, int c) {
        const Vec3& A = pts[a].p; const Vec3& B = pts[b].p; const Vec3& C = pts[c].p;
        const Vec3 e1{ B.x-A.x, B.y-A.y, B.z-A.z };
        const Vec3 e2{ C.x-A.x, C.y-A.y, C.z-A.z };
        Vec3 n{ e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x };
        return 0.5 * std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    };
    const double d02 = tri(q.v[0], q.v[1], q.v[2]) + tri(q.v[0], q.v[2], q.v[3]);
    const double d13 = tri(q.v[0], q.v[1], q.v[3]) + tri(q.v[1], q.v[2], q.v[3]);
    return d02 < d13 ? d02 : d13;
}

// The area of the surface the substance describes, measured on a fine
// PLACED GRID -- the same thing the baker would produce. This is the
// numeric reference the derived mesh is compared against.
//
// THE GRADIENT MUST BE A CENTRAL DIFFERENCE.
//
// THE BUG THIS FIXES (measured 2026-11-11): this used a FORWARD difference
// (f(x+dx) - f(x))/dx. On a dome, the forward difference at a cell's
// CENTRE measures the slope over the cell's FORWARD half, and because the
// slope is steepest at the centre of the footprint the estimate is
// systematically LOW. Measured on the upper hemisphere of radius 0.5:
// the forward difference gave 1.0357 against the true 2*pi*r^2 = 1.5708
// -- a 34% shortfall -- while the central difference gives 1.4628, a 6.9%
// shortfall (which is the expected O(dx^2) truncation error).
//
// The consequence of getting this wrong is worse than a wrong number: the
// reference is the YARDSTICK. A yardstick 34% short makes a mesh that
// covers 101.7% of the true area look like it covers 135% -- and hides
// the very error it was built to catch.
double ReferenceArea(const ISubstance& sub, int N) {
    float hx = 0.5f, hy = 0.5f;
    sub.Domain(hx, hy);
    const float dx = (hx * 2.f) / N, dy = (hy * 2.f) / N;
    // The epsilon of the central difference. It must be much SMALLER than
    // the grid step (so the difference measures the local slope) but much
    // LARGER than the float noise of the evaluation.
    const float e = 1e-5f;
    double area = 0.0;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float x0 = -hx + i * dx, y0 = -hy + j * dy;
            const float xc = x0 + dx*0.5f, yc = y0 + dy*0.5f;
            const SubstancePoint c = sub.Eval(xc, yc);
            if (!c.valid) continue;
            // THE CENTRAL DIFFERENCE, on the substance's own height.
            auto height = [&](float x, float y) -> float {
                const SubstancePoint sp = sub.Eval(x, y);
                return sp.valid ? sp.p.z : c.p.z;
            };
            const float sx = (height(xc+e, yc) - height(xc-e, yc)) / (2.f*e);
            const float sy = (height(xc, yc+e) - height(xc, yc-e)) / (2.f*e);
            // |(-sx,-sy,1) x (0,0,1)| = sqrt(1 + sx^2 + sy^2)
            area += dx * dy * std::sqrt(1.0 + sx*sx + sy*sy);
        }
    return area;
}

// Run the whole battery for one shape and report.
void RunShape(const char* name, const ISubstance& sub, int N,
               double analytic_area, double min_coverage) {
    current_shape = name;
    std::printf("\n--- %s (grid %d) ---\n", name, N);

    const std::vector<MeshPoint> pts = SampleCloud(sub, N);
    check(!pts.empty(), "the cloud is non-empty (%zu points)", pts.size());
    if (pts.empty()) return;

    // The reference area, and -- where a closed form exists -- the
    // agreement between the numeric reference and the analytic one.
    //
    // THE TOLERANCE IS 12%, and it is not slack: the numeric reference is
    // a FIRST-ORDER area element on a finite grid, so it carries an
    // O(dx^2) truncation error. Measured on the upper hemisphere of radius
    // 0.5 at grid 64: the reference is 6.9% below the analytic area. The
    // 12% bound leaves headroom for that without letting a materially
    // wrong yardstick through -- and the check is here to catch a
    // YARDSTICK that is broken, not to pin the reference to a constant.
    const double ref = ReferenceArea(sub, N * 2);
    if (analytic_area > 0.0) {
        const double err = std::fabs(ref - analytic_area) / analytic_area;
        check(err < 0.12,
              "the numeric reference agrees with the analytic area "
              "(ref=%.5f analytic=%.5f err=%.2f%%)", ref, analytic_area, err*100.0);
    }

    MeshDerivation d;
    std::string why;
    const bool ok = DeriveMutualMesh(pts, d, why);
    check(ok, "the derivation succeeded%s%s", ok ? "" : ": ", why.c_str());
    if (!ok) return;

    std::printf("    derived %zu quads (A=%u B=%u avg=%u), %u unpaired; "
                "area_a=%.5f area_b=%.5f (ref %.5f)\n",
                d.quads.size(), d.quads_from_a, d.quads_from_b, d.quads_averaged,
                d.triangles_unpaired, d.area_a, d.area_b, ref);

    // ---- A) it produces cells --------------------------------------------
    check(d.quads.size() > 100, "it produced cells (%zu)", d.quads.size());

    // ---- B) no degenerate cell -------------------------------------------
    int degenerate = 0;
    for (const MeshQuad& q : d.quads) {
        std::set<uint32_t> vs{ q.v[0], q.v[1], q.v[2], q.v[3] };
        if (vs.size() < 4) ++degenerate;
    }
    check(degenerate == 0, "no degenerate cell (%d of %zu)", degenerate, d.quads.size());

    // ---- C) no inside-out cell -------------------------------------------
    // The winding must agree with the SURFACE normal at the cell's centre.
    // A Delaunay does not guarantee this, so it is a real property.
    //
    // THE NORMAL MUST BE THE ENGINE'S OWN. The renderer computes
    // cross(dPdu, dPdv) where dPdu runs A0->A3 and dPdv runs A0->A1
    // (Patch.cpp:180-185) -- NOT the diagonal cross. Testing the diagonal
    // cross here would test a quantity the renderer never computes, and
    // would have reported the mesh as correct while every cell rendered
    // inside-out.
    int inside_out = 0;
    for (const MeshQuad& q : d.quads) {
        const Vec3& A = pts[q.v[0]].p; const Vec3& B = pts[q.v[1]].p;
        const Vec3& C = pts[q.v[2]].p; const Vec3& D = pts[q.v[3]].p;
        // dPdu = (A3 - A0), dPdv = (A1 - A0)
        const Vec3 e1{ D.x-A.x, D.y-A.y, D.z-A.z };
        const Vec3 e2{ B.x-A.x, B.y-A.y, B.z-A.z };
        Vec3 n{ e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x };
        const double len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        if (len < 1e-12) continue;
        n.x /= len; n.y /= len; n.z /= len;
        // The surface normal at the cell's centre, from the cloud's own
        // normals (averaged over the four corners).
        Vec3 sn{ 0,0,0 };
        for (int k = 0; k < 4; ++k) {
            const Vec3& p = pts[q.v[k]].n;
            sn.x += p.x; sn.y += p.y; sn.z += p.z;
        }
        const double sl = std::sqrt(sn.x*sn.x + sn.y*sn.y + sn.z*sn.z);
        if (sl < 1e-12) continue;
        if (n.x*sn.x + n.y*sn.y + n.z*sn.z < 0.0) ++inside_out;
    }
    check(inside_out == 0, "no inside-out cell (%d of %zu)", inside_out, d.quads.size());

    // ---- D) the cells connect --------------------------------------------
    // Count how many cells share each edge. An edge shared by exactly two
    // cells is an INTERIOR edge; by one is a BOUNDARY edge; by three or
    // more is an OVERLAP -- the two meshes' cells crossing.
    std::map<std::pair<uint32_t,uint32_t>, int> edge_use;
    for (const MeshQuad& q : d.quads)
        for (int k = 0; k < 4; ++k) {
            uint32_t a = q.v[k], b = q.v[(k+1)%4];
            if (a > b) std::swap(a, b);
            edge_use[{a,b}]++;
        }
    int interior = 0, boundary = 0, overlap = 0;
    for (const auto& kv : edge_use) {
        if (kv.second == 1) ++boundary;
        else if (kv.second == 2) ++interior;
        else ++overlap;
    }
    // A connected surface has nearly as many interior edges as cells.
    check(interior > (int)(d.quads.size() * 0.7),
          "the cells form a surface (%d interior edges for %zu cells, "
          "%d boundary, %d overlapping)", interior, d.quads.size(), boundary, overlap);

    // ---- E) coverage ------------------------------------------------------
    // Each mesh covers the whole surface ON ITS OWN; the two OVERLAP, so
    // their union describes it twice. That is the point of the design.
    const double cov_a = ref > 0 ? d.area_a / ref : 0.0;
    const double cov_b = ref > 0 ? d.area_b / ref : 0.0;
    check(cov_a >= min_coverage,
          "mesh A covers the surface (%.1f%% of %.5f, need %.0f%%)",
          cov_a*100.0, ref, min_coverage*100.0);
    check(cov_b >= min_coverage,
          "mesh B covers the surface (%.1f%% of %.5f, need %.0f%%)",
          cov_b*100.0, ref, min_coverage*100.0);
    // And the two must actually OVERLAP -- that is what gives the
    // "average them" step something to average.
    check(d.quads_averaged > 0,
          "the two meshes overlap (%u averaged cells)", d.quads_averaged);

    // ---- F) deterministic -------------------------------------------------
    MeshDerivation d2;
    std::string why2;
    const bool ok2 = DeriveMutualMesh(pts, d2, why2);
    bool same = ok2 && d2.quads.size() == d.quads.size();
    if (same)
        for (size_t i = 0; i < d.quads.size(); ++i)
            for (int k = 0; k < 4; ++k)
                if (d.quads[i].v[k] != d2.quads[i].v[k]) { same = false; break; }
    check(same, "the derivation is deterministic");
}

} // namespace

int main() {
    std::printf("=== validate_shape_coverage: the mutual-surround derivation "
                "over basic geometric shapes ===\n");

    // THE SPHERE. The upper hemisphere of radius 0.5: the shape the
    // derivation was developed on, and the one with a closed form to
    // check the yardstick against (2*pi*r^2 = 1.5707963).
    RunShape("sphere", EllipsoidSubstance(0.5f, 0.5f, 0.5f), 32,
             2.0 * 3.14159265358979 * 0.25, 0.80);

    // THE ELLIPSOID -- prolate: the same dome, stretched. The area has no
    // closed form, so the numeric reference stands alone.
    RunShape("ellipsoid-prolate", EllipsoidSubstance(0.5f, 0.25f, 0.35f), 32,
             0.0, 0.80);

    // THE ROD -- a very thin ellipsoid. The worst case for a point cloud:
    // nearly every vertex is on the boundary, so nearly every cell is a
    // boundary cell.
    RunShape("rod", EllipsoidSubstance(0.5f, 0.03f, 0.03f), 32,
             0.0, 0.55);

    // THE TORUS -- the shape with a HOLE. Its support is an annulus, so
    // the cloud has a hole and the triangulations' convex hulls cover it.
    //
    // THE ANALYTIC AREA IS THE UPPER HALF ONLY. The full torus's surface
    // area is 4*pi^2*R*r, but TorusSubstance models only the UPPER half of
    // the tube (z = +sqrt(...)), so the surface being measured is HALF
    // that: 2*pi^2*R*r.
    //
    // (My first attempt compared against the FULL torus's area and the
    // reference came out 27.66% short -- which was the YARDSTICK being
    // wrong, not the subject. The gate checks the reference against the
    // analytic value precisely so a broken yardstick is caught.)
    {
        const float R = 0.32f, r = 0.12f;
        RunShape("torus", TorusSubstance(R, r), 48,
                 2.0 * 3.14159265358979 * 3.14159265358979 * R * r, 0.45);
    }

    // THE PLANE -- unbounded, and its "surface" is only as big as the
    // domain we chose. The simplest possible case: if the derivation
    // cannot handle a flat sheet, nothing else means anything.
    RunShape("plane", PlaneSubstance(), 32, 0.0, 0.70);

    // THE GAUSSIAN BUMP -- a soft boundary on a flat host. The cloud thins
    // toward the edge, so the boundary cells are the hard ones.
    RunShape("gaussian-bump", GaussianSystemSubstance(0.25f, 0.30f), 32,
             0.0, 0.70);

    // THE HEIGHTFIELD -- the generic leaf: an arbitrary analytic surface.
    // A saddle, which is NON-CONVEX in a way the others are not.
    RunShape("heightfield-saddle", HeightfieldSubstance([](float x, float y) {
        return 0.20f * (x*x - y*y);
    }), 32, 0.0, 0.70);

    std::printf("\n=== %d checks, %d failures ===\n", checks, failures);
    if (failures) {
        std::printf("SHAPE COVERAGE GATE FAILED\n");
        return 1;
    }
    std::printf("SHAPE COVERAGE GATE PASSED\n");
    return 0;
}
