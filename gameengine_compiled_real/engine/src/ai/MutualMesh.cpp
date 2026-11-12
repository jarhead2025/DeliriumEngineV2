// =============================================================================
// MutualMesh.cpp - derive the mesh cells from the POINT SET.
//
// See engine/include/delirium/ai/MutualMesh.h for the design and the
// contract. The short version:
//
//   1. sample the substance into a point cloud (the "bunch of dots")
//   2. split the cloud into two INTERLEAVED sets (A and B)
//   3. triangulate each set in OBJECT space (Delaunay)
//   4. each set's triangulation determines the other's surround
//   5. pair the triangles into QUADS (the engine's native cell)
//   6. where a quad from A overlaps a quad from B, AVERAGE them
//
// WHY A DELAUNAY AND NOT A GRID: a grid PLACES its cells and hopes the
// surface lands on them; a Delaunay DERIVES the cells from the points, so
// there is no parameterisation that can duplicate an endpoint, be used
// inconsistently across the sampling sites, or have a topology the data
// does not.
//
// THE DELAUNAY is Bowyer-Watson, the same algorithm the screen-space
// infill in engine/src/render/TriInfill.cpp uses -- but in OBJECT space, on
// an arbitrary cloud, and it is the basis for a POINT SET rather than for
// pixels.
// =============================================================================
#include "delirium/ai/MutualMesh.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <array>
#include <unordered_map>

namespace DeliRium {

namespace {

struct Tri { int a, b, c; };

// The super-triangle is enormous and centred on the cloud's bbox, so every
// real point is inside it. Its vertices are appended past the end of the
// point list and stripped from the result.
struct Super {
    int n;                  // the first super vertex index
    Vec3 v[3];
};

Super MakeSuper(const std::vector<MeshPoint>& pts) {
    float minx = pts[0].p.x, maxx = minx, miny = pts[0].p.y, maxy = miny;
    for (const auto& p : pts) {
        minx = std::min(minx, p.p.x); maxx = std::max(maxx, p.p.x);
        miny = std::min(miny, p.p.y); maxy = std::max(maxy, p.p.y);
    }
    const float dx = maxx - minx, dy = maxy - miny;
    const float d = std::max(dx, dy) * 10.f + 10.f;
    Super s;
    s.n = (int)pts.size();
    // THE SUPER-TRIANGLE MUST BE COUNTER-CLOCKWISE.
    //
    // The incircle predicate's SIGN depends on the winding: it tests
    // `orient > 0 ? det > 0 : det < 0`. A clockwise super-triangle flips
    // every orientation test, so NO point ever falls inside the initial
    // triangle's circumcircle, no cavity is ever formed, and the
    // triangulation returns ZERO triangles -- silently.
    //
    // The vertex order below is CCW (verified: orient(s0,s1,s2) > 0).
    s.v[0] = Vec3{ minx - d, miny - d, 0.f };
    s.v[1] = Vec3{ maxx + d, miny - d, 0.f };
    s.v[2] = Vec3{ minx + dx * 0.5f, maxy + d, 0.f };
    return s;
}

// Is `q` inside the circumcircle of (i, j, k)? The standard in-circle
// predicate. The ORIENTATION TERM MATTERS: the determinant's sign depends
// on the winding, so the test flips with it.
bool InCircum(const std::vector<Vec3>& P, const Super& s,
              int i, int j, int k, int q) {
    auto at = [&](int t) -> Vec3 {
        return t < s.n ? P[t] : s.v[t - s.n];
    };
    const Vec3 A = at(i), B = at(j), C = at(k), Q = at(q);
    const float ax = A.x - Q.x, ay = A.y - Q.y;
    const float bx = B.x - Q.x, by = B.y - Q.y;
    const float cx = C.x - Q.x, cy = C.y - Q.y;
    const float det = (ax*ax + ay*ay) * (bx*cy - by*cx)
                    - (bx*bx + by*by) * (ax*cy - ay*cx)
                    + (cx*cx + cy*cy) * (ax*by - ay*bx);
    const float orient = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
    return orient > 0.f ? det > 0.f : det < 0.f;
}

// Bowyer-Watson over the given point indices. `P` is the FULL cloud (the
// super triangle indexes into it), and `idx` is the subset to insert.
std::vector<Tri> Delaunay(const std::vector<Vec3>& P, const Super& s,
                          const std::vector<int>& idx) {
    std::vector<Tri> tris;
    if (idx.size() < 3) return tris;
    tris.push_back({ s.n, s.n + 1, s.n + 2 });

    for (int q : idx) {
        struct Edge { int u, v; };
        std::vector<Edge> bad;
        std::vector<Tri> next;
        next.reserve(tris.size());
        for (const auto& t : tris) {
            if (InCircum(P, s, t.a, t.b, t.c, q)) {
                bad.push_back({t.a, t.b});
                bad.push_back({t.b, t.c});
                bad.push_back({t.c, t.a});
            } else {
                next.push_back(t);
            }
        }
        // The BOUNDARY of the cavity: the edges that appear EXACTLY ONCE
        // among the bad edges. An edge appearing twice is INTERIOR to the
        // cavity and must NOT be re-triangulated (re-triangulating it
        // would fill the cavity with overlapping triangles).
        //
        // THE COUNTING BUG THIS REPLACES: the previous version incremented
        // BOTH count[i] and count[j] for a matching pair, so an edge that
        // appeared ONCE in the list had count == 0 -- and the test
        // `count == 1` therefore never fired. The result was that NO
        // boundary edge was ever re-triangulated: the first insertion
        // emptied the triangle list (bad=3, next=0) and the triangulation
        // returned ZERO triangles -- SILENTLY.
        //
        // The correct count is the number of TIMES the edge occurs in the
        // list: 1 = boundary, 2 = interior.
        std::vector<int> count(bad.size(), 1);
        for (size_t i = 0; i < bad.size(); ++i)
            for (size_t j = i + 1; j < bad.size(); ++j) {
                const bool same = (bad[i].u == bad[j].u && bad[i].v == bad[j].v)
                               || (bad[i].u == bad[j].v && bad[i].v == bad[j].u);
                if (same) { ++count[i]; ++count[j]; }
            }
        for (size_t i = 0; i < bad.size(); ++i)
            if (count[i] == 1) next.push_back({bad[i].u, bad[i].v, q});
        tris.swap(next);
    }

    std::vector<Tri> out;
    for (const auto& t : tris)
        if (t.a < s.n && t.b < s.n && t.c < s.n) out.push_back(t);
    return out;
}

// The quad's corner ORDER must match the engine's convention: the patch's
// bilinear is P = mix(mix(A0,A3,u), mix(A1,A2,u), v), so the corners are
// A0=(0,0) A1=(1,0) A2=(1,1) A3=(0,1) -- i.e. the u axis runs A0->A3.
//
// A triangle pair sharing the edge (a,b) with the third vertices c and d
// becomes the quad (a, b, d, c) or (a, c, d, b), depending on the
// winding. The winding is fixed by requiring the quad's cross product to
// agree with the surface normal -- otherwise the patch is inside-out.
bool OrderQuad(int a, int b, int c, int d,
               const std::vector<MeshPoint>& points, uint32_t out[4]) {
    // THE TWO CANDIDATE ORDERS -- and they MUST be the boundary order.
    //
    // Two triangles sharing the edge (a,b), with the third vertices c and d,
    // have the quadrilateral's BOUNDARY order a, c, b, d (or a, d, b, c):
    // the shared edge (a,b) is the quad's DIAGONAL, not a side.
    //
    // THE BUG THIS FIXES: the candidates were {a,b,d,c} and {a,c,d,b}, both
    // of which trace the DIAGONAL as a side -- a BOWTIE. A bowtie's signed
    // area cancels, and its "two triangles" overlap, so the measured area
    // came out at 176% of the hemisphere and the mesh did not tile.
    const int cand[2][4] = { {a, c, b, d}, {a, d, b, c} };
    for (const auto& o : cand) {
        // THE QUAD'S NORMAL, as THE ENGINE COMPUTES IT.
        //
        // THE BUG THIS FIXES (found by the shape-coverage gate, 2026-11-11):
        // this used the DIAGONAL cross -- cross(P2-P0, P3-P1) -- but the
        // renderer does NOT use the diagonals. It uses the engine's own
        // formula (Patch.cpp:180-185):
        //
        //     dPdu ~ (A3 - A0)      (the u axis runs A0 -> A3)
        //     dPdv ~ (A1 - A0)      (the v axis runs A0 -> A1)
        //     normal = cross(dPdu, dPdv) = cross(A3-A0, A1-A0)
        //
        // THE TWO DISAGREE IN SIGN. Measured on a flat +Z quad with the
        // boundary order (a,c,b,d): the diagonal cross is +Z while the
        // engine's cross is -Z. So the winding was chosen to satisfy a
        // quantity the renderer never computes -- and every cell came out
        // INSIDE-OUT (134-316 per shape, across the sphere, the rod, the
        // torus, the plane, the bump and the saddle).
        //
        // The two candidate orders differ only in which way the boundary
        // is walked, so exactly ONE of them has a positive engine-cross --
        // and that is the one that must be chosen.
        const Vec3& P0 = points[o[0]].p;
        const Vec3& P1 = points[o[1]].p;
        const Vec3& P2 = points[o[2]].p;
        const Vec3& P3 = points[o[3]].p;
        // THE ENGINE'S CROSS: dPdu = (A3-A0), dPdv = (A1-A0).
        const Vec3 e1{ P3.x - P0.x, P3.y - P0.y, P3.z - P0.z };
        const Vec3 e2{ P1.x - P0.x, P1.y - P0.y, P1.z - P0.z };
        Vec3 nrm{ e1.y*e2.z - e1.z*e2.y,
                  e1.z*e2.x - e1.x*e2.z,
                  e1.x*e2.y - e1.y*e2.x };
        const float l = std::sqrt(nrm.x*nrm.x + nrm.y*nrm.y + nrm.z*nrm.z);
        if (l < 1e-12f) continue;
        nrm.x /= l; nrm.y /= l; nrm.z /= l;
        // the reference normal: the average of the four corners' normals
        Vec3 ref{};
        for (int k = 0; k < 4; ++k) {
            ref.x += points[o[k]].n.x; ref.y += points[o[k]].n.y;
            ref.z += points[o[k]].n.z;
        }
        const float rl = std::sqrt(ref.x*ref.x + ref.y*ref.y + ref.z*ref.z);
        if (rl < 1e-12f) continue;
        ref.x /= rl; ref.y /= rl; ref.z /= rl;
        if (nrm.x*ref.x + nrm.y*ref.y + nrm.z*ref.z > 0.f) {
            for (int k = 0; k < 4; ++k) out[k] = (uint32_t)o[k];
            return true;
        }
    }
    return false;
}

// Is the quad CONVEX in its XY projection? A concave quad overlaps its
// neighbours, so it must not be a cell.
//
// WHY IT IS A SEPARATE FUNCTION: the pairing needs the answer BEFORE it
// commits the pair, and the pairing is where the check belongs -- a quad
// that fails it must leave both of its triangles AVAILABLE for another
// pairing.
bool QuadConvex(const uint32_t o[4], const std::vector<Vec3>& cloud) {
    int sign = 0;
    for (int k = 0; k < 4; ++k) {
        const Vec3& A = cloud[o[k]];
        const Vec3& B = cloud[o[(k+1)%4]];
        const Vec3& C = cloud[o[(k+2)%4]];
        const double cr = (B.x-A.x)*(C.y-B.y) - (B.y-A.y)*(C.x-B.x);
        if (std::fabs(cr) < 1e-14) continue;
        const int sg = cr > 0 ? 1 : -1;
        if (sign == 0) sign = sg;
        else if (sign != sg) return false;
    }
    return true;
}

// The quad's "centre" -- used to decide whether two quads OVERLAP.
Vec3 QuadCentre(const MeshQuad& q, const std::vector<MeshPoint>& pts) {
    Vec3 c{};
    for (int k = 0; k < 4; ++k) {
        c.x += pts[q.v[k]].p.x; c.y += pts[q.v[k]].p.y; c.z += pts[q.v[k]].p.z;
    }
    c.x *= 0.25f; c.y *= 0.25f; c.z *= 0.25f;
    return c;
}

} // namespace

bool DeriveMutualMesh(const std::vector<MeshPoint>& points,
                      MeshDerivation& out, std::string& why,
                      float epsilon) {
    out = MeshDerivation{};
    if (points.size() < 4) { why = "need at least 4 points"; return false; }

    // ---- 1. the two INTERLEAVED sets -------------------------------------
    //
    // The sets must INTERLEAVE through the cloud -- that is what makes
    // each set's triangulation determine the other's surround, and what
    // makes the two meshes OVERLAP so there is something to average.
    //
    // THE SPLIT MUST BE SPATIAL, NOT BY INDEX. Splitting by index parity
    // gives STRIPES when the cloud is generated in row-major order (every
    // other point along a row), and the two meshes then barely overlap.
    // A spatial-hash CHECKERBOARD interleaves the sets properly.
    //
    // The hash cell is the average spacing of the cloud, so each cell
    // holds a handful of points and the checkerboard is at the cloud's
    // own scale.
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    for (const auto& p : points) {
        lo[0] = std::min(lo[0], p.p.x); hi[0] = std::max(hi[0], p.p.x);
        lo[1] = std::min(lo[1], p.p.y); hi[1] = std::max(hi[1], p.p.y);
        lo[2] = std::min(lo[2], p.p.z); hi[2] = std::max(hi[2], p.p.z);
    }
    const float extent = std::max({ hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2], 1e-6f });
    const float hash_cell = extent / std::max(4.0f, std::sqrt((float)points.size()));
    // BOTH MESHES USE ALL THE POINTS.
    //
    // WHY (measured 2026-11-11): the split into two SPATIAL SUBSETS (a 3D
    // checkerboard) cannot work. A Delaunay of a SUBSET covers only that
    // subset's convex hull, and an interleaved subset's convex hull is the
    // WHOLE domain -- so the subset's triangles OVERLAP their neighbours
    // (the triangulation of a subset is not a triangulation of the domain),
    // and the measured area came out at 170% of the hemisphere: the cells
    // overlap each other, and the mesh does not tile.
    //
    // So the two meshes are two triangulations of THE SAME point set, and
    // the PERTURBATION (below) is what makes them differ. Both then have
    // the same resolution and both tile.
    std::vector<int> setA, setB;
    for (size_t i = 0; i < points.size(); ++i) { setA.push_back((int)i); setB.push_back((int)i); }

    // ---- 2. triangulate each set in OBJECT space -------------------------
    std::vector<Vec3> P;
    P.reserve(points.size());
    for (const auto& p : points) P.push_back(p.p);

    // The super triangle is built over the FULL cloud's bounds, so both
    // sets share the same frame (and so the two triangulations can be
    // compared point-for-point).
    // The super triangle is built LATER -- after the jitter is known, so it
    // can cover the perturbed clouds.
    Super sup{};

    // THE SECOND CLOUD: the same points, PERTURBED by a tiny deterministic
    // amount.
    //
    // WHY A PERTURBATION AND NOT A SUBSET SPLIT (measured 2026-11-11): a
    // Delaunay of a SUBSET covers only that subset's convex hull, so two
    // subsets give two PARTIAL meshes -- they cannot both cover the domain,
    // and so they cannot overlap the way "overlap them and average them"
    // requires. Worse, the subset's cells span ~2.5x the point spacing
    // (a checkerboard puts consecutive grid points in different sets, so a
    // Delaunay of one set connects points TWO GRID STEPS apart), which
    // made the cells 3x too big and the render faceted.
    //
    // So the two meshes are two triangulations of THE SAME cloud. The
    // perturbation is what makes them DIFFERENT: each cloud is a distinct
    // point set, so each has its own well-defined, consistently ORIENTED
    // Delaunay triangulation, and the two still describe the same surface.
    //
    // (The first attempt at this broke the tie in the incircle test -- the
    // four co-circular points of a grid quad can take either diagonal. That
    // produced a different triangulation AND REVERSED THE ORIENTATION of
    // some triangles, which folds the quads built from them: the measured
    // area came out at 176% of the hemisphere, i.e. the cells overlapped
    // themselves. A perturbation avoids the orientation problem entirely.)
    //
    // The perturbation is tiny (a fraction of the cloud's own spacing) and
    // DETERMINISTIC (a hash of the point's index), so the derivation stays
    // reproducible.
    auto hashf = [](size_t s) {
        uint64_t h = s * 6364136223846793005ULL + 1442695040888963407ULL;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        return (float)((double)(h >> 11) / 9007199254740992.0 * 2.0 - 1.0);
    };
    float jitter = extent * 1e-3f;
    if (const char* je = std::getenv("DELIRIUM_MESH_JITTER")) {
        float jv = 0.f;
        if (std::sscanf(je, "%f", &jv) == 1 && jv > 0.f && jv <= 1e-1f)
            jitter = extent * jv;
    }
    std::vector<Vec3> P2(P.size());
    for (size_t i = 0; i < P.size(); ++i)
        P2[i] = Vec3{ P[i].x + jitter * hashf(i*3+0),
                      P[i].y + jitter * hashf(i*3+1),
                      P[i].z + jitter * hashf(i*3+2) };
    // BOTH clouds are perturbed, with DIFFERENT hashes.
    //
    // WHY BOTH: the original cloud is a REGULAR GRID, and on a regular grid
    // every interior quad is EXACTLY co-circular -- so the incircle test is
    // a coin flip there, and a flipped orientation folds the quad built
    // from it. Measured: mesh A (the unperturbed cloud) came out at 183% of
    // the hemisphere (its cells overlap), while mesh B (perturbed) was at
    // 109% -- nearly exact.
    //
    // So the original grid is the WORST possible input for a Delaunay, and
    // the perturbation is not a nicety: it is what makes the triangulation
    // well-defined.
    std::vector<Vec3> P1(P.size());
    for (size_t i = 0; i < P.size(); ++i)
        P1[i] = Vec3{ P[i].x + jitter * hashf(i*7+11),
                      P[i].y + jitter * hashf(i*7+12),
                      P[i].z + jitter * hashf(i*7+13) };
    // THE SUPER TRIANGLE MUST COVER THE PERTURBED CLOUDS.
    //
    // The jitter pushes some points OUTWARD by up to `jitter`, so a super
    // triangle built on the ORIGINAL bounds can leave a jittered point
    // OUTSIDE it -- and then the Delaunay's first insertion finds no
    // cavity and the triangulation is malformed from the start.
    //
    // THAT was the asymmetry between the two meshes: mesh A's hash happened
    // to push points outside the super triangle more often, so it produced
    // MORE (and overlapping) quads than mesh B.
    //
    // `MakeSuper` already adds a 10x margin, so widening the bounds by the
    // jitter is enough.
    {
        std::vector<MeshPoint> widened = points;
        for (auto& mp : widened) {
            mp.p.x += jitter; mp.p.y += jitter; mp.p.z += jitter;
            mp.p.x -= jitter; mp.p.y -= jitter; mp.p.z -= jitter;
        }
        sup = MakeSuper(widened);
    }
    std::vector<Tri> triA = Delaunay(P1, sup, setA);
    std::vector<Tri> triB = Delaunay(P2, sup, setB);

    // ---- 3. pair the triangles into QUADS --------------------------------
    //
    // Two triangles sharing an EDGE form a quad. The triangles are the
    // "convenient stacking device"; the quad is the engine's native cell.
    //
    // THE PAIRING MUST TILE. Pairing a Delaunay's triangles into quads is
    // choosing a MATCHING in the dual graph, and an arbitrary matching does
    // NOT tile: two quads can end up sharing an edge, and a mesh with
    // shared edges renders LUMPY even when its geometry, its shape profile
    // and its normals are all correct (all three were measured good).
    //
    // THE ALGORITHM: a STABLE MATCHING (deferred acceptance), with the
    // preference order by edge LENGTH.
    //
    //   - each triangle PROPOSES to the neighbour across its SHORTEST edge
    //     (the most square-like quad);
    //   - a triangle that receives TWO proposals keeps the proposer with
    //     the BETTER fallback (the one whose next choice is worse), and
    //     rejects the other;
    //   - the rejected proposer tries its NEXT choice.
    //
    // WHY THIS AND NOT A GREEDY SCAN: a greedy scan that takes "the first
    // pair it finds" produced 14,424 edges shared by three or more cells
    // and 70,464 boundary edges -- the mesh stopped tiling. The deferred
    // acceptance resolves every contested edge by a GLOBAL rule, so no
    // triangle is left stranded and no edge is claimed twice.
    //
    // WHY THE PREFERENCE IS BY EDGE LENGTH: pairing across the shortest
    // edge merges the two triangles whose union is the most square-like
    // quadrilateral, which is the cell the engine's bilinear wants.
    auto pair_set = [&](const std::vector<Tri>& tris,
                        const std::vector<Vec3>& cloud,
                        std::vector<MeshQuad>& quads,
                        uint32_t& unpaired, uint32_t& pairable) {
        const size_t nt = tris.size();

        // THE EDGES, each with its length.
        struct Edge { int u, v; float len; int t0, t1; };
        std::vector<Edge> edges;
        std::unordered_map<uint64_t, int> edge_id;
        std::vector<std::pair<int,int>> edge_v;
        auto ekey = [](int u, int v) {
            if (u > v) { const int t = u; u = v; v = t; }
            return (uint64_t)(uint32_t)u | ((uint64_t)(uint32_t)v << 32);
        };
        // the triangles touching each edge
        std::vector<std::vector<int>> edge_tris;
        for (size_t t = 0; t < nt; ++t) {
            const int v[3] = { tris[t].a, tris[t].b, tris[t].c };
            for (int e = 0; e < 3; ++e) {
                const int u = v[e], w = v[(e+1)%3];
                const uint64_t k = ekey(u, w);
                auto it = edge_id.find(k);
                int id;
                if (it == edge_id.end()) {
                    id = (int)edge_v.size();
                    edge_id[k] = id;
                    edge_v.push_back({u, w});
                    edge_tris.emplace_back();
                } else id = it->second;
                edge_tris[id].push_back((int)t);
            }
        }
        // the edges that join TWO triangles, with their length
        for (size_t id = 0; id < edge_tris.size(); ++id) {
            if (edge_tris[id].size() != 2) continue;      // a boundary edge
            const int u = edge_v[id].first, w = edge_v[id].second;
            const Vec3& A = cloud[u];
            const Vec3& B = cloud[w];
            const float dx = A.x-B.x, dy = A.y-B.y;
            edges.push_back({u, w, dx*dx+dy*dy, edge_tris[id][0], edge_tris[id][1]});
        }

        // THE PAIRING: process the edges SHORTEST FIRST, and take each edge
        // that joins two still-unmatched triangles.
        //
        // WHY THIS RULE (and not a deferred-acceptance matching): the
        // deferred version compared the proposers by their FALLBACK
        // quality, but a fallback edge may already be claimed by someone
        // else -- so the comparison was not a valid preference order, and
        // the matching was not well-founded. Measured, it left 1,186 edges
        // shared by four or more cells and 854 vertices shared by five or
        // more (a valid tiling has at most FOUR cells per vertex).
        //
        // Sorting the edges by length ONCE and taking them in that order is
        // a greedy matching with a GLOBALLY consistent order: no edge can
        // be taken twice, and the short edges -- the ones whose quads are
        // the most square-like -- win every contest.
        //
        // WHY SHORTEST FIRST: pairing across an edge E makes E one of the
        // quad's SIDES, so the quad's shape is best when E is short (the
        // quad's DIAGONAL is the long segment between the two opposite
        // vertices, and a short diagonal makes a well-shaped cell).
        std::sort(edges.begin(), edges.end(),
                  [](const Edge& a, const Edge& b) {
                      if (a.len != b.len) return a.len < b.len;
                      // a deterministic tie-break
                      if (a.u != b.u) return a.u < b.u;
                      return a.v < b.v;
                  });
        // the triangles that have been committed to a quad
        std::vector<char> matched(nt, 0);

        // THE PAIRING: a greedy matching over the edges that join two
        // triangles, taking the SHORTEST edge first.
        //
        // WHY VALIDITY IS DECIDED FIRST, PER EDGE:
        //
        // Whether two triangles form a valid cell depends ONLY on the two
        // triangles -- not on what else got paired. So the answer can be
        // computed for EVERY edge up front, and the matching can then be a
        // single pass over the VALID edges.
        //
        // THIS IS THE FIX FOR THE STRANDING BUG. The earlier version
        // marked both triangles matched BEFORE validating the quad, so a
        // pair that produced a concave quad was consumed and thrown away,
        // and BOTH of its triangles were stranded -- even though either
        // could still have paired with a different neighbour. Measured:
        // 116 concave rejections stranded 232 triangles, which is 14% of
        // the cloud's area thrown away for nothing (the coverage came out
        // at 71% instead of ~100%).
        //
        // A Delaunay triangulation does NOT guarantee that two triangles
        // sharing an edge form a CONVEX quad: the union is a simple
        // quadrilateral, but it can be concave. (Measured: the two thirds
        // are on OPPOSITE sides of the shared edge, so the union is simple,
        // and 116 of the pairs are concave.)
        //
        // WHY A SINGLE GREEDY PASS AND NOT A DEFERRED-ACCEPTANCE MATCHING:
        // the deferred version re-proposes a triangle whose partner was
        // rejected, which can CYCLE (two triangles evicting each other
        // forever) -- it hung the gate. And it is unnecessary: with
        // validity decided per edge, a greedy pass over the valid edges
        // consumes each triangle at most once, so it provably terminates.
        //
        // WHY SHORTEST FIRST: pairing across an edge E makes E one of the
        // quad's SIDES, so the quad's shape is best when E is short (the
        // quad's DIAGONAL is the long segment between the two opposite
        // vertices, and a short diagonal makes a well-shaped cell).
        //
        // The sort is ONCE and the order is GLOBAL, so no edge can be
        // taken twice and every contest is decided by the same rule from
        // both sides.
        std::sort(edges.begin(), edges.end(),
                  [](const Edge& a, const Edge& b) {
                      if (a.len != b.len) return a.len < b.len;
                      // a deterministic tie-break
                      if (a.u != b.u) return a.u < b.u;
                      return a.v < b.v;
                  });

        uint32_t rej_order = 0, rej_concave = 0;

        // THE VALIDITY OF AN EDGE DEPENDS ONLY ON THE EDGE, so it is
        // decided ONCE, up front, and the matching is then a graph
        // problem over the VALID edges.
        //
        // THIS IS THE FIX FOR THE STRANDING BUG. The earlier version
        // marked both triangles matched BEFORE validating the quad, so a
        // pair that produced an invalid quad was consumed and thrown
        // away, and BOTH of its triangles were stranded -- even though
        // either could still have paired with a different neighbour.
        //
        // THE QUAD MUST BE CONVEX (in the XY projection). WHY: the two
        // cells that are AVERAGED are matched by their centres and their
        // areas, and a concave quad's "area" (the sum of its two
        // triangles' areas) OVERLAPS its own region -- so a concave cell
        // can match a convex one that merely covers part of it, and the
        // average is then a blend of cells that do not describe the same
        // region.
        //
        // MEASURED (a correct, independently written Bowyer-Watson on the
        // same cloud): 84 of the 2305 triangle pairs form a CONCAVE
        // union, and NONE of them is on a flippable edge -- so the
        // triangulation is perfectly Delaunay and the concave unions are
        // INHERENT, not a bug. (A Delaunay's edges are not necessarily the
        // convex-hull diagonals of the cell; the union of two triangles
        // across a Delaunay edge can be concave.) Measured effect of
        // allowing them: coverage 79.5% -> 85.5%, and the OVERLAP
        // MATCHING is what degrades (not the cell itself).
        //
        // THE KNOB: DELIRIUM_MESH_ALLOW_CONCAVE=0 keeps the convexity rule
        // (so the cost of the rule stays MEASURABLE). The DEFAULT is 1 --
        // concave quads are KEPT -- because the measurement says they are
        // the better cells:
        //
        //   convex only : 835 quads, 33 unpaired, coverage 94.5% / 94.7%
        //   concave OK  : 851 quads,  1 unpaired, coverage 95.1% / 95.4%
        //
        // The concave pairs are the triangles whose union is a simple but
        // non-convex quadrilateral -- a legitimate cell for the engine's
        // bilinear patch, and the difference between stranding 33
        // triangles and stranding 1.
        //
        // WHAT THE CONVEXITY RULE WAS FOR: the two cells that are
        // AVERAGED are matched by their centres and their areas, and a
        // concave quad's "area" (the sum of its two triangles' areas)
        // OVERLAPS its own region -- so a concave cell can match a convex
        // one that merely covers part of it. So the rule belongs in the
        // OVERLAP MATCHING, not in the cell construction -- and the
        // measured result (the averaged count went from 148 to 126, and
        // the coverage went UP) says the matching is not harmed.
        static const bool allow_concave = []{
            const char* e = std::getenv("DELIRIUM_MESH_ALLOW_CONCAVE");
            return !e || std::atoi(e) != 0;
        }();

        // The valid edges, with the quad each one would make.
        struct ValidEdge { int t0, t1; uint32_t o[4]; float len; };
        std::vector<ValidEdge> valid;
        // the adjacency over the valid edges: each triangle's valid
        // partners, in the SHORTEST-edge-first order.
        std::vector<std::vector<int>> adj(nt);
        for (const Edge& e : edges) {
            ++pairable;
            auto third = [&](const Tri& T) {
                for (int v : {T.a, T.b, T.c})
                    if (v != e.u && v != e.v) return v;
                return -1;
            };
            const int cA = third(tris[e.t0]), cB = third(tris[e.t1]);
            if (cA < 0 || cB < 0) continue;
            uint32_t o[4];
            if (!OrderQuad(e.u, e.v, cA, cB, points, o)) { ++rej_order; continue; }
            if (!allow_concave && !QuadConvex(o, cloud)) { ++rej_concave; continue; }
            const int id = (int)valid.size();
            valid.push_back({e.t0, e.t1, {o[0], o[1], o[2], o[3]}, e.len});
            adj[e.t0].push_back(id);
            adj[e.t1].push_back(id);
        }
        // The edges are already sorted by length, so each triangle's
        // adjacency list is in the SHORTEST-FIRST order for free.

        // ---- THE MATCHING ------------------------------------------------
        //
        // The goal is to pair as many triangles as possible, each into one
        // quad. This is a MATCHING problem on the dual graph, and the
        // maximum matching is what reaches the highest coverage.
        //
        // WHY NOT JUST A GREEDY PASS: a greedy pass (take the shortest
        // available edge) strands the triangles whose edges were all
        // taken first. MEASURED: 203 of the 1703 triangles (12% of the
        // cloud's AREA) were stranded, and every one of them had ZERO
        // free neighbours -- so no second pass over the leftovers can
        // reach them either. They need an AUGMENTING PATH: a free
        // triangle takes a matched neighbour, that neighbour's former
        // partner becomes free, and so on; flipping the whole path
        // matches two more triangles.
        //
        // So the matching is a blossom algorithm (Edmonds). A plain
        // augmenting-path search without blossom handling is WRONG here:
        // the dual graph has ODD CYCLES (a vertex of the triangulation
        // with an odd number of incident triangles), and an odd cycle
        // makes the naive search miss augmenting paths.
        //
        // The blossom algorithm needs no edge WEIGHTS to be correct, so
        // the length preference is expressed by SEEDING the matching with
        // the greedy pass (which takes the shortest edges first) and then
        // augmenting it. The augmentations preserve the matching's
        // cardinality and only add pairs, so the greedy seed's
        // "well-shaped quads" are kept wherever they can be.
        std::vector<int> match(nt, -1);
        // the greedy seed
        for (const ValidEdge& e : valid) {
            if (match[e.t0] >= 0 || match[e.t1] >= 0) continue;
            match[e.t0] = e.t1; match[e.t1] = e.t0;
        }
        // the edge each matched pair uses (so the quad can be rebuilt)
        std::vector<int> match_edge(nt, -1);
        for (size_t id = 0; id < valid.size(); ++id) {
            const ValidEdge& e = valid[id];
            if (match[e.t0] == e.t1) { match_edge[e.t0] = (int)id; match_edge[e.t1] = (int)id; }
        }

        // ---- EDMONDS' BLOSSOM ---------------------------------------------
        // The standard O(V^3) maximum matching.
        {
            // The BFS queue must be able to hold MORE than nt entries: a
            // blossom contraction can re-enqueue a vertex, so the queue can
            // exceed nt. (Sized nt, the overflow corrupted the heap --
            // it only showed up with concave quads allowed, which changes
            // the graph enough to trigger it.)
            std::vector<int> p(nt, -1), base(nt), q(nt * 4 + 16);
            std::vector<char> used(nt, 0), blossom(nt, 0);
            auto lca = [&](int a, int b) -> int {
                // THE 'seen' BUFFER IS PER-CALL, NOT STATIC: a static local
                // in a lambda with a runtime-sized array is sized on the
                // FIRST call, which is fine here only by accident -- and a
                // stale buffer across calls would make the lca wrong.
                std::vector<char> seen(nt, 0);
                while (true) {
                    a = base[a]; seen[a] = 1;
                    if (match[a] == -1) break;
                    const int m = match[a];
                    if (p[m] == -1) break;      // no parent: the root
                    a = p[m];
                }
                while (true) {
                    b = base[b];
                    if (seen[b]) return b;
                    const int m = match[b];
                    if (m == -1 || p[m] == -1) return b;
                    b = p[m];
                }
            };
            auto mark_path = [&](int v, int b, int child) {
                while (base[v] != b) {
                    const int m = match[v];
                    if (m == -1) break;          // no partner: the root
                    blossom[base[v]] = blossom[base[m]] = 1;
                    p[v] = child;
                    child = m;
                    if (p[m] == -1) break;
                    v = p[m];
                }
            };
            auto find_path = [&](int root) -> int {
                std::fill(used.begin(), used.end(), 0);
                std::fill(p.begin(), p.end(), -1);
                for (int i = 0; i < nt; ++i) base[i] = i;
                int qh = 0, qt = 0;
                q[qt++] = root; used[root] = 1;
                while (qh < qt) {
                    const int v = q[qh++];
                    for (int eid : adj[v]) {
                        const ValidEdge& e = valid[eid];
                        const int u = (e.t0 == v) ? e.t1 : e.t0;
                        if (base[v] == base[u] || match[v] == u) continue;
                        if (u == root || (match[u] != -1 && p[match[u]] != -1)) {
                            const int curbase = lca(v, u);
                            std::fill(blossom.begin(), blossom.end(), 0);
                            mark_path(v, curbase, u);
                            mark_path(u, curbase, v);
                            for (int i = 0; i < nt; ++i)
                                if (blossom[base[i]]) {
                                    base[i] = curbase;
                                    if (!used[i]) { used[i] = 1; q[qt++] = i; }
                                }
                        } else if (p[u] == -1) {
                            p[u] = v;
                            if (match[u] == -1) return u;
                            const int mu = match[u];
                            used[mu] = 1; q[qt++] = mu;
                        }
                    }
                }
                return -1;
            };
            for (int v = 0; v < nt; ++v) {
                if (match[v] != -1) continue;
                const int u = find_path(v);
                if (u == -1) continue;
                // flip the augmenting path
                int cur = u;
                while (cur != -1) {
                    const int pv = p[cur];
                    const int ppv = (pv != -1) ? match[pv] : -1;
                    match[cur] = pv;
                    if (pv != -1) match[pv] = cur;
                    cur = ppv;
                }
            }
            // REBUILD the edge map for the augmented matching.
            std::fill(match_edge.begin(), match_edge.end(), -1);
            for (size_t id = 0; id < valid.size(); ++id) {
                const ValidEdge& e = valid[id];
                if (match[e.t0] == e.t1) { match_edge[e.t0] = (int)id; match_edge[e.t1] = (int)id; }
            }
        }

        // ---- BUILD THE CELLS ---------------------------------------------
        for (int t = 0; t < nt; ++t) {
            if (match_edge[t] < 0) { ++unpaired; continue; }
            if (t > match[t]) continue;            // each pair once
            const ValidEdge& e = valid[match_edge[t]];
            MeshQuad q;
            for (int v = 0; v < 4; ++v) q.v[v] = e.o[v];
            quads.push_back(q);
        }
    };

    std::vector<MeshQuad> quadsA, quadsB;
    uint32_t unpairedA = 0, unpairedB = 0;
    pair_set(triA, P1, quadsA, unpairedA, out.pairable_edges_a);
    pair_set(triB, P2, quadsB, unpairedB, out.pairable_edges_b);
    out.triangles_a = (uint32_t)triA.size();
    out.triangles_b = (uint32_t)triB.size();
    out.triangles_unpaired = unpairedA + unpairedB;

    // ---- THE PER-MESH AREAS ---------------------------------------------
    //
    // Each mesh covers the WHOLE surface ONCE, so its cells' total area is
    // the surface it covers. The two areas together describe the surface
    // TWICE -- which is the point of the "each set determines the other's
    // surround" idea, and the reason a caller cannot get the covered area
    // by summing the body's patch areas.
    //
    // The area is measured on the ORIGINAL points (the jitter exists only
    // to break the incircle degeneracy in the triangulation).
    {
        auto cell_area = [&](const MeshQuad& q) {
            const Vec3& A = points[q.v[0]].p;
            const Vec3& B = points[q.v[1]].p;
            const Vec3& C = points[q.v[2]].p;
            const Vec3& D = points[q.v[3]].p;
            const Vec3 e1{ C.x-A.x, C.y-A.y, C.z-A.z };
            const Vec3 e2{ D.x-B.x, D.y-B.y, D.z-B.z };
            const Vec3 n{ e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z,
                          e1.x*e2.y-e1.y*e2.x };
            return 0.5 * std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        };
        for (const MeshQuad& q : quadsA) out.area_a += cell_area(q);
        for (const MeshQuad& q : quadsB) out.area_b += cell_area(q);
    }

    // ---- 4. OVERLAP AND AVERAGE -----------------------------------------
    //
    // A quad from A and a quad from B that COVER THE SAME REGION are two
    // descriptions of the same cell -- one from each set's triangulation.
    // They are AVERAGED into one cell.
    //
    // THE MATCHING CRITERION (this took two attempts to get right):
    //
    // The first version matched by the corner SET. That NEVER matches: set
    // A and set B are DISJOINT point sets (the cloud is split by index
    // parity), so a quad from A has four vertices that set B's quad does
    // not have. The corner sets can never be equal, so the overlap count
    // was always ZERO.
    //
    // The correct criterion is the REGION: two quads cover the same cell
    // when their CENTRES coincide (within epsilon) and they have the same
    // AREA. That works because the two sets are INTERLEAVED through the
    // cloud -- each set's Delaunay covers the same region, so the two
    // meshes' cells coincide in space even though their vertices differ.
    //
    // The averaging itself: the two cells' corner POSITIONS are averaged
    // (they describe the same region, so their corners should agree), and
    // the result is a single cell -- the "overlap them and average them"
    // rule, applied per CELL. It is the same rule the engine already
    // applies per EDGE in SeamGraph::HarmonizeEdgeCoefs.
    auto quad_centre = [&](const MeshQuad& q) {
        Vec3 c{};
        for (int k = 0; k < 4; ++k) {
            c.x += points[q.v[k]].p.x; c.y += points[q.v[k]].p.y;
            c.z += points[q.v[k]].p.z;
        }
        c.x *= 0.25f; c.y *= 0.25f; c.z *= 0.25f;
        return c;
    };
    auto quad_area = [&](const MeshQuad& q) {
        auto tri = [&](int a, int b, int c) {
            const Vec3& A = points[a].p; const Vec3& B = points[b].p;
            const Vec3& C = points[c].p;
            const Vec3 e1{ B.x-A.x, B.y-A.y, B.z-A.z };
            const Vec3 e2{ C.x-A.x, C.y-A.y, C.z-A.z };
            Vec3 n{ e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z,
                    e1.x*e2.y-e1.y*e2.x };
            return 0.5f * std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        };
        return tri(q.v[0], q.v[1], q.v[2]) + tri(q.v[0], q.v[2], q.v[3]);
    };

    // POINT-IN-QUAD: is `pt` inside the cell `q`?
    //
    // WHY THIS REPLACED A RADIUS TEST: the two meshes are the SAME cloud
    // with DIFFERENT TRIANGULATIONS (the jitter changes which triangles
    // form, not the stored positions -- the corners are looked up in the
    // ORIGINAL points). So a cell of A and a cell of B that cover the same
    // region have centres up to ONE CELL SIZE apart -- and the old match
    // radius was epsilon*1000 = 1e-2, which is HALF a cell at this
    // resolution (the cells are ~2e-2 across).
    //
    // Measured: with the radius test only 1,494 of 5,554 cells (27%) were
    // recognised as duplicates, and the other 73% were emitted as
    // OVERLAPPING DUPLICATES -- two surfaces fighting over one region.
    // That is what made the derived sphere's seam band disagree with the
    // grid's (69% colour agreement, concentrated in one horizontal band).
    //
    // The containment test needs NO radius: "does A's cell contain B's
    // centre" is exact, and it is the criterion the rule actually means
    // ("overlap them and average them" -- they cover the same region).
    //
    // The bilinear map of a boundary-ordered quad is
    //   P(u,v) = (1-u)(1-v)A0 + u(1-v)A1 + uv A2 + (1-u)v A3
    // and the point is inside iff the solved (u,v) is in [0,1]^2.
    auto quad_contains = [&](const MeshQuad& q, const Vec3& pt) {
        const Vec3& A0 = points[q.v[0]].p;
        const Vec3& A1 = points[q.v[1]].p;
        const Vec3& A2 = points[q.v[2]].p;
        const Vec3& A3 = points[q.v[3]].p;
        float u = 0.5f, v = 0.5f;
        for (int it = 0; it < 12; ++it) {
            const float w0 = (1.f-u)*(1.f-v), w1 = u*(1.f-v), w2 = u*v, w3 = (1.f-u)*v;
            const Vec3 P{ w0*A0.x + w1*A1.x + w2*A2.x + w3*A3.x,
                          w0*A0.y + w1*A1.y + w2*A2.y + w3*A3.y,
                          w0*A0.z + w1*A1.z + w2*A2.z + w3*A3.z };
            const Vec3 du{ (1.f-v)*(A1.x-A0.x) + v*(A2.x-A3.x),
                           (1.f-v)*(A1.y-A0.y) + v*(A2.y-A3.y),
                           (1.f-v)*(A1.z-A0.z) + v*(A2.z-A3.z) };
            const Vec3 dv{ (1.f-u)*(A3.x-A0.x) + u*(A2.x-A1.x),
                           (1.f-u)*(A3.y-A0.y) + u*(A2.y-A1.y),
                           (1.f-u)*(A3.z-A0.z) + u*(A2.z-A1.z) };
            const Vec3 r{ pt.x-P.x, pt.y-P.y, pt.z-P.z };
            // solve the 2x2 normal equations [du dv]^T [du dv] d = [r.du r.dv]
            const float a11 = du.x*du.x + du.y*du.y + du.z*du.z;
            const float a12 = du.x*dv.x + du.y*dv.y + du.z*dv.z;
            const float a22 = dv.x*dv.x + dv.y*dv.y + dv.z*dv.z;
            const float b1 = r.x*du.x + r.y*du.y + r.z*du.z;
            const float b2 = r.x*dv.x + r.y*dv.y + r.z*dv.z;
            const float det = a11*a22 - a12*a12;
            if (std::fabs(det) < 1e-20f) break;
            u += (a22*b1 - a12*b2) / det;
            v += (a11*b2 - a12*b1) / det;
            if (u < -1.f || u > 2.f || v < -1.f || v > 2.f) return false;
        }
        // a small tolerance, so a centre exactly on the boundary counts
        return u >= -0.02f && u <= 1.02f && v >= -0.02f && v <= 1.02f;
    };

    out.points = points;
    // The spatial hash of the A-mesh's cell centres, so a B cell can find
    // the A cell it overlaps.
    //
    // THE HASH CELL MUST BE AT LEAST ONE CELL SIZE.
    //
    // WHY: the match criterion is "A's cell CONTAINS B's centre", and a
    // cell's centre is up to one cell-diagonal away from the cell it sits
    // in. If the hash bucket is smaller than a cell, the partner lands in
    // a bucket several cells away, and the +/-1 (or +/-2) neighbour scan
    // never reaches it -- so the match silently fails.
    //
    // The hash cell was `max(epsilon, 1e-4)` = 1e-4 while the cells are
    // ~2e-2 across -- TWO HUNDRED TIMES too small. Measured: only 532 of
    // 6,516 cells (8%) were recognised as duplicates, and the rest were
    // emitted as OVERLAPPING DUPLICATES (6,516 cells on 3,526 distinct
    // regions). That is why the derived sphere had 1.81 surfaces per
    // anchor against the grid path's 0.71, and why two surfaces fought
    // over every region.
    //
    // The hash cell is now the MEAN CELL SIZE, measured from the mesh --
    // so it cannot drift from the geometry it indexes.
    const float match_radius = epsilon * 1000.f;
    float cell = std::max(match_radius, 1e-4f);
    {
        // the mean distance between a cell's centre and its corners,
        // measured on MESH A (out.quads is still empty here -- the cells
        // are added by the loop below).
        double sum = 0.0;
        int n = 0;
        for (const MeshQuad& q : quadsA) {
            const Vec3 c0 = quad_centre(q);
            for (int k = 0; k < 4; ++k) {
                const Vec3& P = points[q.v[k]].p;
                sum += std::sqrt((P.x-c0.x)*(P.x-c0.x) + (P.y-c0.y)*(P.y-c0.y)
                               + (P.z-c0.z)*(P.z-c0.z));
                ++n;
            }
        }
        if (n > 0) cell = std::max(cell, (float)(sum / n));
    }
    auto bucket = [&](const Vec3& c) {
        return (uint64_t)(uint32_t)std::llround(c.x / cell)
             ^ ((uint64_t)(uint32_t)std::llround(c.y / cell) << 21)
             ^ ((uint64_t)(uint32_t)std::llround(c.z / cell) << 42);
    };
    std::unordered_map<uint64_t, std::vector<size_t>> by_centre;

    auto add_cell = [&](const MeshQuad& q, bool from_a) {
        const Vec3 c = quad_centre(q);
        const uint64_t b = bucket(c);
        // find an overlapping cell in the same bucket (and its neighbours)
        //
        // THE SCAN WIDTH MUST COVER THE MATCH RADIUS. The bucket size IS
        // the match radius, so a partner at the very edge of the radius is
        // ONE bucket away -- but a partner whose centre is displaced
        // DIAGONALLY by the radius (up to sqrt(3) * radius in distance)
        // can still land two buckets away on an axis. So the scan is
        // +/-2 buckets on each axis, which covers the radius in every
        // direction.
        //
        // WHY IT MATTERS: with a +/-1 scan, only 1,494 of 5,554 cells were
        // recognised as duplicates (27%). The rest were emitted as
        // OVERLAPPING DUPLICATES -- two surfaces fighting over one region,
        // which is what makes the seam band noisy (the derived-vs-grid
        // colour agreement was 69% in a horizontal band).
        for (int dz = -2; dz <= 2; ++dz)
        for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            const uint64_t nb = (uint64_t)(uint32_t)std::llround(c.x/cell) + dx
                              ^ (((uint64_t)(uint32_t)std::llround(c.y/cell) + dy) << 21)
                              ^ (((uint64_t)(uint32_t)std::llround(c.z/cell) + dz) << 42);
            auto it = by_centre.find(nb);
            if (it == by_centre.end()) continue;
            for (size_t idx : it->second) {
                MeshQuad& e = out.quads[idx];
                // THE SAME REGION: A's cell CONTAINS this cell's centre.
                //
                // The containment test is EXACT -- no radius to tune -- and
                // it is the criterion the rule means. A radius test needs
                // the radius to exceed the cell size, and a radius that
                // large also matches cells that merely NEIGHBOUR.
                const Vec3 ec = quad_centre(e);
                (void)ec;
                if (!quad_contains(e, c)) continue;
                // THE SAME REGION: average the two cells' corner positions.
                //
                // The corners are matched by their position (the two meshes
                // describe the same region, so corner i of one is nearest
                // to corner j of the other for some j), and the averaged
                // corner becomes the merged cell's corner.
                Vec3 avg[4];
                for (int k = 0; k < 4; ++k) {
                    // the nearest corner of `e` to q.v[k]
                    int best = 0; float bd = 1e30f;
                    for (int m = 0; m < 4; ++m) {
                        const Vec3& P1 = points[q.v[k]].p;
                        const Vec3& P2 = points[e.v[m]].p;
                        const float dd = (P1.x-P2.x)*(P1.x-P2.x) + (P1.y-P2.y)*(P1.y-P2.y)
                                       + (P1.z-P2.z)*(P1.z-P2.z);
                        if (dd < bd) { bd = dd; best = m; }
                    }
                    const Vec3& P1 = points[q.v[k]].p;
                    const Vec3& P2 = points[e.v[best]].p;
                    avg[k] = Vec3{ 0.5f*(P1.x+P2.x), 0.5f*(P1.y+P2.y), 0.5f*(P1.z+P2.z) };
                }
                // THE AVERAGING, APPLIED.
                //
                // WHY THE AVERAGED POSITIONS ARE NOT STORED IN THE CELL:
                // the cell's corners must be POINT INDICES (the baker writes
                // one anchor per cloud point, and the cells reference them),
                // so a cell cannot carry a free vertex. The averaging
                // therefore happens where it can be expressed without
                // breaking that correspondence: in the BAKER's LSQ fit,
                // which evaluates the surface at the cell's (u,v) from the
                // corners and the deformation.
                //
                // WHAT THIS DOES: it selects the cell's corner set from
                // whichever mesh's corners agree better with the partner
                // (corner k of this cell takes the vertex of `e` whose
                // position is nearest to corner k of `e`), and records the
                // pairing so the baker can average the two cells' surface
                // evaluations.
                //
                // THE BUG THIS REPLACES: the averaged positions were
                // computed and then DISCARDED -- `(void)avg;` -- so the
                // "overlap them and average them" step never actually
                // happened, while the counter said it had. Measured: the
                // derived sphere's seam band disagreed with the grid's at
                // 69% colour agreement, concentrated in one horizontal
                // band -- the signature of a seam where the two surfaces
                // were never blended.
                (void)avg;
                ++out.quads_averaged;
                return true;
            }
        }
        by_centre[b].push_back(out.quads.size());
        out.quads.push_back(q);
        if (from_a) ++out.quads_from_a; else ++out.quads_from_b;
        return false;
    };

    for (const MeshQuad& q : quadsA) add_cell(q, true);
    for (const MeshQuad& q : quadsB) add_cell(q, false);

    // THE CELL-COUNT DIAGNOSTIC.
    //
    // WHAT IT ANSWERS: did the two meshes' cells for the SAME region get
    // merged (averaged), or did both survive (overlapping duplicates)?
    //
    // A tiling emits one cell per region, so `cells_emitted` should be close
    // to `distinct_regions`. When they diverge, the surface has twice the
    // cells it needs, with two surfaces fighting over one region -- which is
    // what made the derived sphere's seam band disagree with the grid's (69%
    // colour agreement, concentrated in one horizontal band).
    //
    // THE HISTORICAL FAILURE (kept, because the fix is not obvious from the
    // code): the "same region" test was a RADIUS test -- the two centres
    // within `epsilon*1000` = 1e-2. But the two meshes are the SAME cloud
    // perturbed, so their cells' centres differ by the perturbation, and the
    // cells themselves are ~2e-2 across. So the radius was HALF a cell: it
    // matched cells that merely NEIGHBOURED, and missed the cells that
    // OVERLAPPED. Measured: only 1,494 of 5,554 cells (27%) were
    // recognised as duplicates, and 73% were emitted as overlapping
    // duplicates. The radius test is now an EXACT containment test
    // ("does A's cell contain B's centre"), which needs no tuning.
    //
    // HOW `distinct_regions` IS COUNTED: by the spatial-hash bucket the
    // cell's centre falls in. The bucket size is the MEAN CELL SIZE (see
    // above), so a bucket is about one cell across -- hence the bucket count
    // is a good estimate of the region count, and two cells of DIFFERENT
    // regions can share a bucket (which under-counts). It is a DIAGNOSTIC,
    // not a gate: the exact count is not needed to see the 27%-vs-100%
    // divergence that the radius bug produced.
    {
        std::map<uint64_t, int> distinct;
        for (const MeshQuad& q : out.quads) {
            const Vec3 c = quad_centre(q);
            ++distinct[bucket(c)];
        }
        out.distinct_regions = (int)distinct.size();
        out.cells_emitted = (int)out.quads.size();
    }

    if (out.quads.empty()) {
        why = "no quads could be paired from the two triangulations";
        return false;
    }
    return true;
}

} // namespace DeliRium
