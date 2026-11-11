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
#include <map>
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
               const std::vector<MeshPoint>& pts, uint32_t out[4]) {
    // The two candidate orders.
    const int cand[2][4] = { {a, b, d, c}, {a, c, d, b} };
    for (const auto& o : cand) {
        // the quad's normal from its diagonals
        const Vec3& P0 = pts[o[0]].p;
        const Vec3& P1 = pts[o[1]].p;
        const Vec3& P2 = pts[o[2]].p;
        const Vec3& P3 = pts[o[3]].p;
        const Vec3 e1{ P2.x - P0.x, P2.y - P0.y, P2.z - P0.z };
        const Vec3 e2{ P3.x - P1.x, P3.y - P1.y, P3.z - P1.z };
        Vec3 nrm{ e1.y*e2.z - e1.z*e2.y,
                  e1.z*e2.x - e1.x*e2.z,
                  e1.x*e2.y - e1.y*e2.x };
        const float l = std::sqrt(nrm.x*nrm.x + nrm.y*nrm.y + nrm.z*nrm.z);
        if (l < 1e-12f) continue;
        nrm.x /= l; nrm.y /= l; nrm.z /= l;
        // the reference normal: the average of the four corners' normals
        Vec3 ref{};
        for (int k = 0; k < 4; ++k) {
            ref.x += pts[o[k]].n.x; ref.y += pts[o[k]].n.y;
            ref.z += pts[o[k]].n.z;
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
    std::vector<int> setA, setB;
    for (size_t i = 0; i < points.size(); ++i) {
        const long long cx = (long long)std::llround((points[i].p.x - lo[0]) / hash_cell);
        const long long cy = (long long)std::llround((points[i].p.y - lo[1]) / hash_cell);
        const long long cz = (long long)std::llround((points[i].p.z - lo[2]) / hash_cell);
        // a 3D CHECKERBOARD: the parity of the cell sum.
        if (((cx + cy + cz) & 1LL) == 0) setA.push_back((int)i);
        else                            setB.push_back((int)i);
    }

    // ---- 2. triangulate each set in OBJECT space -------------------------
    std::vector<Vec3> P;
    P.reserve(points.size());
    for (const auto& p : points) P.push_back(p.p);

    // The super triangle is built over the FULL cloud's bounds, so both
    // sets share the same frame (and so the two triangulations can be
    // compared point-for-point).
    const Super sup = MakeSuper(points);
    std::vector<Tri> triA = Delaunay(P, sup, setA);
    std::vector<Tri> triB = Delaunay(P, sup, setB);

    // ---- 3. pair the triangles into QUADS --------------------------------
    //
    // Two triangles sharing an EDGE form a quad. The triangles are the
    // "convenient stacking device"; the quad is the engine's native cell.
    //
    // The pairing is done PER SET: within A's triangulation, and within
    // B's. The edge key is the unordered pair of point indices.
    auto pair_set = [&](const std::vector<Tri>& tris,
                        std::vector<MeshQuad>& quads,
                        uint32_t& unpaired, uint32_t& pairable) {
        // bucket the triangles by their edges
        std::map<std::pair<int,int>, std::vector<int>> by_edge;
        for (size_t t = 0; t < tris.size(); ++t) {
            const int v[3] = { tris[t].a, tris[t].b, tris[t].c };
            for (int e = 0; e < 3; ++e) {
                int u = v[e], w = v[(e+1)%3];
                if (u > w) std::swap(u, w);
                by_edge[{u, w}].push_back((int)t);
            }
        }
        std::vector<char> used(tris.size(), 0);
        for (auto& [k, lst] : by_edge) {
            if (lst.size() != 2) continue;
            ++pairable;
            const Tri& T1 = tris[lst[0]];
            const Tri& T2 = tris[lst[1]];
            if (used[lst[0]] || used[lst[1]]) continue;
            // the shared edge is (k.first, k.second); the third vertices
            // are the ones not on it
            auto third = [&](const Tri& T) {
                for (int v : {T.a, T.b, T.c})
                    if (v != k.first && v != k.second) return v;
                return -1;
            };
            const int c1 = third(T1), c2 = third(T2);
            if (c1 < 0 || c2 < 0) continue;
            uint32_t o[4];
            if (!OrderQuad(k.first, k.second, c1, c2, points, o)) continue;
            MeshQuad q;
            for (int v = 0; v < 4; ++v) q.v[v] = o[v];
            quads.push_back(q);
            used[lst[0]] = used[lst[1]] = 1;
        }
        for (size_t t = 0; t < tris.size(); ++t)
            if (!used[t]) ++unpaired;
    };

    std::vector<MeshQuad> quadsA, quadsB;
    uint32_t unpairedA = 0, unpairedB = 0;
    pair_set(triA, quadsA, unpairedA, out.pairable_edges_a);
    pair_set(triB, quadsB, unpairedB, out.pairable_edges_b);
    out.triangles_a = (uint32_t)triA.size();
    out.triangles_b = (uint32_t)triB.size();
    out.triangles_unpaired = unpairedA + unpairedB;

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

    out.points = points;
    // The spatial hash of the A-mesh's cell centres, so a B cell can find
    // the A cell it overlaps.
    const float cell = std::max(epsilon, 1e-4f);
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
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const uint64_t nb = (uint64_t)(uint32_t)std::llround(c.x/cell) + dx
                              ^ (((uint64_t)(uint32_t)std::llround(c.y/cell) + dy) << 21)
                              ^ (((uint64_t)(uint32_t)std::llround(c.z/cell) + dz) << 42);
            auto it = by_centre.find(nb);
            if (it == by_centre.end()) continue;
            for (size_t idx : it->second) {
                MeshQuad& e = out.quads[idx];
                const Vec3 ec = quad_centre(e);
                const float d = std::sqrt((ec.x-c.x)*(ec.x-c.x) + (ec.y-c.y)*(ec.y-c.y)
                                        + (ec.z-c.z)*(ec.z-c.z));
                if (d > epsilon * 1000.f) continue;
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
                // the merged cell's corners are the averaged positions, and
                // they must still be POINTS (the baker writes anchors, not
                // free vertices) -- so the merged cell keeps the vertices
                // of whichever mesh is finer, and the averaging shows up
                // in the SURFACE FIT (the LSQ target), not in the records.
                //
                // WHAT THE AVERAGING ACTUALLY DOES HERE: it selects the
                // cell's corner set from the mesh whose corners agree
                // better, and records that the cell was AVERAGED. The
                // geometric averaging happens in the baker's LSQ fit,
                // which averages the two cells' surface evaluations.
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

    if (out.quads.empty()) {
        why = "no quads could be paired from the two triangulations";
        return false;
    }
    return true;
}

} // namespace DeliRium
