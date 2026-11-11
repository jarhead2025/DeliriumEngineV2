// =============================================================================
// validate_mutual_mesh - GATE for the MUTUAL-SURROUND MESH DERIVATION.
//
// THE IDEA (Josias, 2026-11-10): derive the mesh cells from the POINT SET.
// Split the dots into two sets; use each set to determine the other's
// surround; the triangles are a convenient stacking device; the cells are
// overlapped and averaged.
//
// THE GATE asserts the properties that make the derivation BETTER than a
// placed grid, not just that it runs:
//
//   A) the derivation PRODUCES cells (and a lot of them)
//   B) every cell's corners are DISTINCT points (no degenerate cell)
//   C) every cell's winding agrees with the surface normal (no inside-out
//      cell) -- this is the property a Delaunay does NOT give you for free
//   D) the cells CONNECT: consecutive cells share edges, so the mesh is a
//      surface and not a bag of disconnected quads
//   E) the two sets' triangulations really are DIFFERENT (so the "mutual"
//      is doing something) and their cells OVERLAP (so the averaging has
//      something to average)
//   F) the derivation is DETERMINISTIC (same cloud -> same cells)
//
// It also compares against the PLACED GRID: the derived mesh must cover
// the same surface (its cells' areas must sum to roughly the surface's
// area), which is the property that says the derivation is not throwing
// geometry away.
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
void check(bool cond, const char* fmt, ...) {
    ++checks;
    char msg[512];
    va_list ap; va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    std::printf("  %s %s\n", cond ? "OK  " : "FAIL", msg);
    if (!cond) ++failures;
}

// Sample a substance into a point cloud -- the "bunch of dots". A simple
// GRID of samples is fine here: the point SET is what the derivation
// consumes, and how it was sampled does not change what the derivation
// does with it. (The baker's own sampling would give the same cloud shape.)
std::vector<MeshPoint> SampleCloud(const ISubstance& sub, int N, float hx, float hy) {
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

double quad_area(const MeshQuad& q, const std::vector<MeshPoint>& pts) {
    // The two triangles of the quad.
    auto tri = [&](int a, int b, int c) {
        const Vec3& A = pts[a].p; const Vec3& B = pts[b].p; const Vec3& C = pts[c].p;
        const Vec3 e1{ B.x-A.x, B.y-A.y, B.z-A.z };
        const Vec3 e2{ C.x-A.x, C.y-A.y, C.z-A.z };
        Vec3 n{ e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x };
        return 0.5 * std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    };
    return tri(q.v[0], q.v[1], q.v[2]) + tri(q.v[0], q.v[2], q.v[3]);
}

} // namespace

int main() {
    const int N = 32;
    EllipsoidSubstance ell(0.5f, 0.5f, 0.5f);
    std::vector<MeshPoint> pts = SampleCloud(ell, N, 0.5f, 0.5f);
    std::printf("sampled %zu points\n", pts.size());

    MeshDerivation d;
    std::string why;
    const bool ok = DeriveMutualMesh(pts, d, why);
    std::printf("DIAG: set A -> %u triangles, %u pairable edges; "
                "set B -> %u triangles, %u pairable edges\n",
                d.triangles_a, d.pairable_edges_a,
                d.triangles_b, d.pairable_edges_b);
    if (!ok) {
        std::printf("FAIL: the derivation failed: %s\n", why.c_str());
        return 1;
    }
    std::printf("derived: %zu quads (A=%u B=%u averaged=%u), "
                "%u triangles unpaired\n",
                d.quads.size(), d.quads_from_a, d.quads_from_b,
                d.quads_averaged, d.triangles_unpaired);

    // ---- A) it produces cells --------------------------------------------
    //
    // AND the triangulation itself produced triangles. This is a SEPARATE
    // assertion from "there are quads", because the bug that cost the most
    // time was in the CAVITY BOUNDARY: the boundary edges were counted
    // wrongly, so NO edge was ever re-triangulated, the triangle list
    // emptied on the first insertion, and the whole thing returned zero
    // cells -- SILENTLY, with no error.
    check(d.triangles_a > 0 && d.triangles_b > 0,
          "BOTH triangulations produced triangles (A=%u B=%u)",
          d.triangles_a, d.triangles_b);
    check(d.pairable_edges_a > 0 && d.pairable_edges_b > 0,
          "both triangulations have PAIRABLE edges (A=%u B=%u)",
          d.pairable_edges_a, d.pairable_edges_b);
    check(d.quads.size() > 100,
          "the derivation PRODUCES cells (%zu)", d.quads.size());

    // ---- B) the corners are distinct -------------------------------------
    int degenerate = 0;
    for (const MeshQuad& q : d.quads) {
        std::set<uint32_t> s(q.v, q.v + 4);
        if (s.size() != 4) ++degenerate;
    }
    std::printf("degenerate cells (repeated corners): %d\n", degenerate);
    check(degenerate == 0, "every cell has FOUR DISTINCT corners (%d bad)",
          degenerate);

    // ---- C) the winding agrees with the surface normal --------------------
    //
    // A Delaunay triangulation has NO notion of "which side is up", so
    // this must be FIXED, not assumed. A cell whose normal disagrees with
    // the surface's is INSIDE-OUT and would render as a hole.
    int inside_out = 0;
    float worst_dot = 1.f;
    for (const MeshQuad& q : d.quads) {
        const Vec3& P0 = pts[q.v[0]].p;
        const Vec3& P1 = pts[q.v[1]].p;
        const Vec3& P2 = pts[q.v[2]].p;
        const Vec3& P3 = pts[q.v[3]].p;
        const Vec3 e1{ P2.x-P0.x, P2.y-P0.y, P2.z-P0.z };
        const Vec3 e2{ P3.x-P1.x, P3.y-P1.y, P3.z-P1.z };
        Vec3 nrm{ e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x };
        const float l = std::sqrt(nrm.x*nrm.x + nrm.y*nrm.y + nrm.z*nrm.z);
        if (l < 1e-12f) { ++inside_out; continue; }
        nrm.x /= l; nrm.y /= l; nrm.z /= l;
        Vec3 ref{};
        for (int k = 0; k < 4; ++k) {
            ref.x += pts[q.v[k]].n.x; ref.y += pts[q.v[k]].n.y;
            ref.z += pts[q.v[k]].n.z;
        }
        const float rl = std::sqrt(ref.x*ref.x + ref.y*ref.y + ref.z*ref.z);
        if (rl < 1e-12f) continue;
        const float dot = (nrm.x*ref.x + nrm.y*ref.y + nrm.z*ref.z) / rl;
        if (dot < worst_dot) worst_dot = dot;
        if (dot <= 0.f) ++inside_out;
    }
    std::printf("inside-out cells: %d, worst normal dot: %.4f\n",
                inside_out, worst_dot);
    check(inside_out == 0,
          "every cell's winding agrees with the surface normal (%d bad)",
          inside_out);

    // ---- D) the cells CONNECT --------------------------------------------
    //
    // A bag of disconnected quads is not a surface. Count the edges that
    // are shared by exactly two cells -- a connected mesh has many.
    std::map<std::pair<uint32_t,uint32_t>, int> edge_count;
    for (const MeshQuad& q : d.quads)
        for (int e = 0; e < 4; ++e) {
            uint32_t u = q.v[e], w = q.v[(e+1)%4];
            if (u > w) std::swap(u, w);
            edge_count[{u, w}]++;
        }
    int shared = 0, boundary = 0;
    for (auto& [k, c] : edge_count) {
        if (c == 2) ++shared;
        else if (c == 1) ++boundary;
    }
    std::printf("edges: %zu shared by 2 cells, %d on the boundary\n",
                edge_count.size(), boundary);
    std::printf("  shared %d, boundary %d, cells %zu\n", shared, boundary,
                d.quads.size());
    check(shared > (int)d.quads.size() / 2,
          "the cells CONNECT (%d shared edges for %zu cells)",
          shared, d.quads.size());

    // ---- E) the two sets are different, and their cells OVERLAP ---------
    check(d.quads_from_a > 0 && d.quads_from_b > 0,
          "BOTH sets contributed cells (A=%u B=%u)",
          d.quads_from_a, d.quads_from_b);
    check(d.quads_averaged > 0,
          "some cells were OVERLAPPED and AVERAGED (%u)", d.quads_averaged);

    // ---- F) deterministic --------------------------------------------------
    MeshDerivation d2;
    std::string why2;
    if (!DeriveMutualMesh(pts, d2, why2)) {
        check(false, "the derivation is deterministic (the second run failed)");
    } else {
        bool same = d2.quads.size() == d.quads.size();
        if (same)
            for (size_t i = 0; i < d.quads.size(); ++i)
                for (int k = 0; k < 4; ++k)
                    if (d.quads[i].v[k] != d2.quads[i].v[k]) { same = false; break; }
        check(same, "the derivation is DETERMINISTIC (same cells twice)");
    }

    // ---- the derived mesh COVERS the surface -----------------------------
    //
    // The property that says the derivation is not throwing geometry away:
    // the cells' areas must sum to roughly the surface's area.
    //
    // THE MEASUREMENT IS PER MESH, not over the merged set. The two
    // meshes OVERLAP -- that is the point of the idea -- so summing the
    // merged cells' areas double-counts the overlap and reports ~187% of
    // the surface. Each mesh on its own covers the surface ONCE.
    //
    // So the measure is: the area of ONE mesh's cells. The merged set's
    // area is reported too, but as the OVERLAP factor, not as coverage.
    //
    // The ellipsoid's upper hemisphere has area 2*pi*r^2 for r=0.5.
    const double expect = 2.0 * 3.14159265358979 * 0.25;   // a hemisphere

    // Re-derive each mesh's cells separately, so the areas are per-mesh.
    // The derivation reports how many cells came from each set, but not
    // WHICH -- so the area is measured by re-running the pairing on each
    // set's triangulation. That is what the counts are for: the cells are
    // in `d.quads` in the order A's then B's.
    double area_a = 0, area_b = 0;
    size_t k = 0;
    for (uint32_t i = 0; i < d.quads_from_a && k < d.quads.size(); ++i, ++k)
        area_a += quad_area(d.quads[k], pts);
    for (uint32_t i = 0; i < d.quads_from_b && k < d.quads.size(); ++i, ++k)
        area_b += quad_area(d.quads[k], pts);
    double area = area_a + area_b;
    std::printf("mesh A's area: %.4f  mesh B's area: %.4f  "
                "(the hemisphere's is %.4f)\n",
                area_a, area_b, expect);
    std::printf("the merged set's area: %.4f (%.1f%% -- the OVERLAP factor)\n",
                area, 100.0 * area / expect);
    check(area_a > expect * 0.80 && area_a < expect * 1.20,
          "mesh A COVERS the surface (%.1f%% of the hemisphere)",
          100.0 * area_a / expect);
    check(area_b > expect * 0.80 && area_b < expect * 1.20,
          "mesh B COVERS the surface (%.1f%% of the hemisphere)",
          100.0 * area_b / expect);
    // The two meshes MUST overlap -- that is what there is to average.
    check(area > expect * 1.20,
          "the two meshes OVERLAP (the merged area is %.1f%% of one mesh's)",
          100.0 * area / expect);

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
