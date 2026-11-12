// =============================================================================
// DerivedMeshBake.cpp - bake a body whose mesh cells are DERIVED from the
// point set, instead of placed on a grid.
//
// See engine/include/delirium/ai/MutualMesh.h for the derivation itself.
// This file is the BAKER half: it takes the derived connectivity and emits
// the same 64-byte records the grid path emits.
//
// WHY A SEPARATE FILE: the grid path's patch loop is built around a regular
// lattice -- the patch's corners are at grid positions (j*N+i), and the LSQ
// target is sampled at the patch's (u,v) ON THE GRID. A DERIVED patch has
// no grid position: its corners are four arbitrary points, and its (u,v) is
// its own bilinear parameterisation. So the two paths differ in exactly
// three places (the corner indices, the target sampling, and the
// connectivity), and keeping them apart makes the difference auditable.
//
// THE THREE PLACES THEY DIFFER:
//
//  1. THE CORNERS. The grid path's patch spans one grid cell: (i,j),
//     (i+1,j), (i+1,j+1), (i,j+1). The derived path's patch spans the quad's
//     four vertices, in the order the derivation fixed (so the winding
//     agrees with the surface normal).
//
//  2. THE LSQ TARGET. The grid path evaluates the substance at the patch's
//     (u,v) on the sheet's DOMAIN -- because the patch's corners sit at
//     that domain's grid positions. The derived path has no such
//     correspondence, so it evaluates the target in the PATCH'S OWN FRAME:
//     the target at the patch's (u,v) is the substance's surface at the
//     BILINEAR INTERPOLATION of the quad's four corner points, projected
//     back onto the substance.
//
//  3. THE CONNECTIVITY. The grid path derives it from the lattice (+i, -i,
//     +j, -j). The derived path derives it from the DERIVED adjacency:
//     two cells that share two vertices are neighbours.
//
// WHAT IS SHARED: the anchor records (positions, normals, colours, bone
// binding), the quantisation, the corner-vanishing basis, the LSQ solve,
// and the seam graph. So a derived body is a FIRST-CLASS body -- it loads,
// renders, simulates and AMR-refines exactly like a grid body.
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/MutualMesh.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/ai/ReconstructionModel.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace DeliRium {

namespace {

// The 8-term corner-vanishing basis (the SAME one the kernel and the grid
// path use -- see kernels/psf_splat.cl's deform_h and Patch.cpp's copy).
void cv_basis(float u, float v, double B[8]) {
    const double bu = u * (1.0 - u), bv = v * (1.0 - v);
    const double au = 2.0 * u - 1.0, av = 2.0 * v - 1.0;
    const double w = 1.0 - v, x = 1.0 - u;
    B[0] = bu * w;      B[1] = bu * v;
    B[2] = bv * x;      B[3] = bv * u;
    B[4] = bu * au * w; B[5] = bu * au * v;
    B[6] = bv * av * x; B[7] = bv * av * u;
}

// Solve the 8x8 normal equations. (The same routine the grid path uses;
// duplicated here rather than shared, so this file stands alone.)
bool solve8(double M[8][8], double r[8], float out[8]) {
    // Gaussian elimination with partial pivoting.
    double a[8][9];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) a[i][j] = M[i][j];
        a[i][8] = r[i];
    }
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        for (int i = col + 1; i < 8; ++i)
            if (std::fabs(a[i][col]) > std::fabs(a[piv][col])) piv = i;
        if (std::fabs(a[piv][col]) < 1e-12) return false;
        if (piv != col) for (int j = 0; j < 9; ++j) std::swap(a[col][j], a[piv][j]);
        const double d = a[col][col];
        for (int j = col; j < 9; ++j) a[col][j] /= d;
        for (int i = 0; i < 8; ++i) {
            if (i == col) continue;
            const float f = (float)a[i][col];
            if (f == 0.f) continue;
            for (int j = col; j < 9; ++j) a[i][j] -= (double)f * a[col][j];
        }
    }
    for (int i = 0; i < 8; ++i) out[i] = (float)a[i][8];
    return true;
}

inline uint8_t q8(float v) {
    const float c = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    return (uint8_t)(int)(c * 255.f + 0.5f);
}

} // namespace

// The public entry: sample ONE sheet's substance into the point cloud.
//
// `sheet` is the leaf substance to sample; `origin`/`scale` place it in
// the world. `out` receives the anchors and surfaces of that sheet only --
// the caller concatenates the layers.
bool SampleSheetCloud(const Element& sheet_el,
                      int sample_n, float extent,
                      std::vector<MeshPoint>& cloud,
                      std::string& why) {
    cloud.clear();
    if (!sheet_el.substance) { why = "the sheet has no substance"; return false; }

    // ---- 1. sample the substance into a point cloud ------------------------
    //
    // The cloud is the "bunch of dots". The sample count is the user's
    // coarseness budget: the grid path's N x N becomes the cloud's N x N.
    const int N = std::max(4, std::min(64, sample_n));
    float hx = extent * 0.5f, hy = extent * 0.5f;
    sheet_el.substance->Domain(hx, hy);
    if (hx < 1e-6f) hx = extent * 0.5f;
    if (hy < 1e-6f) hy = extent * 0.5f;
    // THE DOMAIN IS NOT CLAMPED TO THE FOOTPRINT.
    //
    // An earlier version clamped the domain down to `extent * 0.5f`. That
    // clamp was MEASURED and found NOT to be the cause of the derived
    // path's missing rim -- with extent = 1.0 the clamp's bound (0.5) was
    // above the substance's own domain (0.477), so it never bound. It is
    // kept as a bound against a domain that is absurdly larger than the
    // body's extent, but with a safety factor so it cannot cut the rim.
    //
    // `Domain()` deliberately returns the footprint PLUS the rim's reach
    // (see ISubstance::Domain: "THE DOMAIN INCLUDES THE RIM. The rim lives
    // OUTSIDE the footprint, so a domain of exactly (sx, sy) never samples
    // it -- and the rim is the whole point of the mechanism.").
    {
        const float kRimSafety = 1.5f;
        const float lim_x = extent * 0.5f * kRimSafety;
        const float lim_y = extent * 0.5f * kRimSafety;
        if (hx > lim_x) hx = lim_x;
        if (hy > lim_y) hy = lim_y;
    }

    cloud.reserve(size_t(N) * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float x = ((i + 0.5f) / N - 0.5f) * (hx * 2.f);
            const float y = ((j + 0.5f) / N - 0.5f) * (hy * 2.f);
            const SubstancePoint sp = sheet_el.substance->Eval(x, y);
            if (!sp.valid) continue;
            // The point is placed in the SHEET'S LOCAL frame; the origin and
            // scale are applied when the anchors are written, exactly as
            // the grid path does (the field composer applies them).
            cloud.push_back({ sp.p, sp.n, 1.f });
        }
    if (cloud.size() < 4) { why = "not enough in-support samples"; return false; }

    // ---- 1b. THE RIM, AS A FILLED ANNULUS --------------------------------
    //
    // WHY THIS IS NEEDED: the grid path's closed body relies on the two
    // sheets' rims OVERLAPPING in an annulus, so their normals agree there
    // and the seam averages instead of cutting. The grid path gets that
    // because it samples the substance on a REGULAR LATTICE, and every
    // lattice sample past the footprint lands in the rim band -- so the
    // annulus is FILLED.
    //
    // The derived path samples the SAME Eval on the SAME lattice, so it gets
    // the same rim points -- but only the ones that happened to land in the
    // thin band. Measured at N=48: 180 rim points whose radii span just
    // 0.4509..0.4744 and whose MEDIAN ANGULAR GAP is 0.039 rad. That is a
    // PERIMETER RING, not a filled annulus.
    //
    // And a perimeter ring is fatal for a Delaunay: the triangulation spans
    // the ring's interior with long thin triangles, whose cells stretch
    // from the surface down to the rim. Those stretched cells are the
    // VERTICAL CURTAIN STREAKS in the derived sphere's render (the grid
    // sphere has no streaks -- its rim IS filled).
    //
    // So the rim is added EXPLICITLY: a ring of points at several radii
    // inside the rim band, so the annulus is filled and the triangulation
    // has material to work with.
    //
    // The rim's points are the substance's OWN rim points (Eval returns
    // them), pushed to the requested radii by the same rule Eval uses: the
    // radius is scaled about the footprint's edge along the gradient
    // direction. The NORMAL is Eval's rim normal -- the footprint's
    // gradient -- which is what makes the two sheets agree (the flip is in
    // Z, the gradient is in XY).
    {
        // Collect the rim points Eval produced, and the footprint's edge.
        struct RimPt { float x, y, r; Vec3 n; };
        std::vector<RimPt> rim;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const float x = ((i + 0.5f) / N - 0.5f) * (hx * 2.f);
                const float y = ((j + 0.5f) / N - 0.5f) * (hy * 2.f);
                const SubstancePoint sp = sheet_el.substance->Eval(x, y);
                if (!sp.valid) continue;
                // A rim point is one ON the boundary: its z is 0 while the
                // surface's z is positive (for a closed-body sheet).
                const float r = std::sqrt(x*x + y*y);
                const bool on_rim = (std::fabs(sp.p.z) < 1e-6f);
                if (on_rim) rim.push_back({ x, y, r, sp.n });
            }
        if (!rim.empty()) {
            // The rim band's radii: from the footprint's edge out to the
            // furthest rim point. Fill it with RINGS at several radii.
            float r_in = 1e30f, r_out = 0.f;
            for (const RimPt& rp : rim) {
                r_in = std::min(r_in, rp.r);
                r_out = std::max(r_out, rp.r);
            }
            // How many rings? Enough that the annulus is filled at the
            // cloud's own angular resolution.
            //
            // THE ANGULAR DENSITY IS THE CRITICAL PART. The surface's
            // outermost row is a lattice row clipped to the ellipse, so it
            // has ~N/2 points around the circumference -- i.e. an angular
            // spacing of ~2*pi/(N/2) = 4*pi/N. If the rim ring has FEWER
            // points than that, the Delaunay has to span the gap with long
            // radial cells, and those cells' LSQ fits cannot represent the
            // dome's curvature -- which is what the VERTICAL CURTAIN
            // STREAKS below the seam are (measured: they persist with a
            // filled annulus, so the annulus being FILLED is not enough --
            // it must be filled at the SURFACE's density).
            //
            // So the rim gets 4x the surface's angular resolution, and the
            // radius steps are matched to the surface's own spacing.
            const int rings = std::max(2, std::min(8, N / 6));
            const int per_ring = std::max(16, N * 4);
            //
            // THE KNOB (DELIRIUM_MESH_RIM_RINGS / _RIM_PER_RING): the rim
            // annulus is a dense RING of points around the cloud's edge, and
            // a ring is HOSTILE to the triangle pairing: every triangle that
            // touches the ring is a BOUNDARY triangle of the triangulation,
            // and a boundary triangle has fewer neighbours to pair with.
            //
            // MEASURED (N=48, the derived sphere bake):
            //
            //   rings | unpaired triangles | residual (grid = 0.2981)
            //   ------+--------------------+------------------------
            //      0  |        358         |  0.2531  (BETTER than grid)
            //      1  |      1,596         |  0.3108
            //      2  |      2,787         |  0.4683  (57% worse)
            //      4  |      5,825         |  0.4687  (57% worse)
            //      8  |     11,943         |  0.3216  (and disconnected)
            //
            // And the geometry barely moves: the derived sphere's geometry
            // is 171,997 px at 0 rings and 172,185 px at 8 -- a 0.1%
            // difference -- for 42x the stranding (1,324 vs 55,996 in the
            // full scene). So the annulus was buying almost nothing and
            // costing almost everything.
            //
            // WHY THE ANNULUS WAS ADDED IN THE FIRST PLACE: the grid path's
            // closed body relies on the two sheets' rims OVERLAPPING in an
            // annulus, so their normals agree there and the seam averages
            // instead of cutting. The grid path gets that because it samples
            // the substance on a REGULAR LATTICE and every lattice sample
            // past the footprint lands in the rim band -- so the annulus is
            // FILLED. The derived path samples the same Eval on the same
            // lattice, so it gets the same rim points -- but only the ones
            // that happened to land in the thin band (measured at N=48: 180
            // rim points spanning 0.4509..0.4744, a PERIMETER RING).
            //
            // The right fix is therefore NOT a dense artificial ring but the
            // SAME thing the grid path does: sample the substance on the
            // lattice and KEEP the rim-band samples (they are real points of
            // the substance's own rim). That is what rings=0 already does --
            // the rim points Eval returned are still in the cloud; only the
            // EXTRA synthetic rings are dropped.
            //
            // The default is 0. Set DELIRIUM_MESH_RIM_RINGS=N to add the
            // synthetic annulus back.
            const char* rings_env = std::getenv("DELIRIUM_MESH_RIM_RINGS");
            const char* per_env = std::getenv("DELIRIUM_MESH_RIM_PER_RING");
            const int rings_eff = rings_env ? std::atoi(rings_env) : 0;
            const int per_eff = per_env ? std::atoi(per_env) : per_ring;
            for (int ring = 0; ring <= rings_eff; ++ring) {
                const float t = rings_eff > 0 ? float(ring) / float(rings_eff) : 0.f;
                const float want_r = r_in + (r_out - r_in) * t;
                // For each angular slot, find the rim point nearest that
                // radius and push it out to the requested radius.
                for (int a = 0; a < per_eff; ++a) {
                    const float ang = 2.f * 3.14159265f * float(a) / float(per_eff);
                    const float cx = std::cos(ang), cy = std::sin(ang);
                    // the rim point with the closest angle
                    const RimPt* best = nullptr;
                    float best_ad = 1e30f;
                    for (const RimPt& rp : rim) {
                        const float ra = std::atan2(rp.y, rp.x);
                        float d = std::fabs(ra - ang);
                        if (d > 3.14159265f) d = 6.28318531f - d;
                        if (d < best_ad) { best_ad = d; best = &rp; }
                    }
                    if (!best) continue;
                    const float s = (best->r > 1e-9f) ? want_r / best->r : 1.f;
                    cloud.push_back({ Vec3{ best->x * s, best->y * s, 0.f }, best->n, 1.f });
                }
            }
        }
    }
    return true;
}

// Bake ONE sheet's DERIVED mesh (the G22 path).
bool BakeDerivedSheet(const Element& sheet_el,
                      const Vec3& origin, const Vec3& scale,
                      const std::vector<MeshPoint>& cloud,
                      const Vec3& bbmin, const Vec3& bbspan,
                      std::vector<AnchorRecord>& anchors,
                      std::vector<SurfaceRecord>& surfaces,
                      MeshDerivation& stats,
                      std::string& why) {
    if (!sheet_el.substance) { why = "the sheet has no substance"; return false; }
    if (cloud.size() < 4) { why = "not enough in-support samples"; return false; }

    // ---- 2. derive the mesh ------------------------------------------------
    if (!DeriveMutualMesh(cloud, stats, why)) return false;

    // ---- 3. the anchors ----------------------------------------------------
    //
    // One anchor per cloud POINT (not per cell): the cells reference the
    // points. The anchor's position is the point's, placed in the world.
    //
    // WHY NOT DEDUPE: the two sets' meshes overlap, and an overlapped cell
    // is AVERAGED rather than duplicated -- so one anchor per point is
    // correct, and deduping would break the correspondence between a cell's
    // corner index and the point it names.
    const size_t base = anchors.size();
    anchors.reserve(base + cloud.size());

    // The world-space points. The QUANTISATION BOX IS THE CALLER'S -- the
    // union over every sheet, so all the layers share ONE coordinate system.
    //
    // WHY IT IS NOT COMPUTED HERE: each sheet's own box spans only its half
    // (an upper sheet z>=0 and a mirrored lower one z<=0), so quantising
    // each sheet against its own box would express the two halves in
    // DIFFERENT coordinate systems -- and the dequantisation (which uses the
    // body's single box) would then place them in the wrong place. This is
    // the G16/G18 lesson: the box IS the coordinate system, not metadata.
    std::vector<Vec3> wpos(cloud.size()), wnrm(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        const MeshPoint& mp = cloud[i];
        wpos[i] = Vec3{ origin.x + mp.p.x * scale.x,
                        origin.y + mp.p.y * scale.y,
                        origin.z + mp.p.z * scale.z };
        // The normal is the inverse-transpose of the scale (the same rule
        // the grid path uses in SheetAt).
        wnrm[i] = Vec3{ mp.n.x / scale.x, mp.n.y / scale.y, mp.n.z / scale.z };
        const float l = std::sqrt(wnrm[i].x*wnrm[i].x + wnrm[i].y*wnrm[i].y
                                 + wnrm[i].z*wnrm[i].z);
        if (l > 1e-12f) { wnrm[i].x /= l; wnrm[i].y /= l; wnrm[i].z /= l; }
    }

    // The quantisation box, as the CALLER computed it: the span plus a
    // margin, exactly as the grid path's Bake computes it (so the
    // dequantisation agrees).
    auto safe = [](float s) { return s < 1e-6f ? 1e-6f : s; };
    const float bsx = safe(bbspan.x), bsy = safe(bbspan.y), bsz = safe(bbspan.z);

    // THE SEAM, IN QUANTISED UNITS.
    //
    // A closed body's two sheets MEET at a plane -- z = 0 for the standard
    // closed-body recipe. Their rims both lie exactly on that plane, and
    // they must land on the SAME quantised value, or the two sheets' rims
    // do not coincide and the seam shows as a step.
    //
    // WHY THE PLAIN ROUNDING BREAKS IT: the seam is the MIDPOINT of the
    // body's z range (the body is symmetric about it), so its quantised
    // parameter t = 0.5 -- and 0.5 * 65535 = 32767.5, which the plain
    // `+ 0.5f` rounding sends to 32768. But 32768/65535 = 0.50000763, NOT
    // 0.5, so the rim dequantises to +0.5 quantisation steps ABOVE the
    // seam. Measured (the closed sphere, the sheet_probe):
    //
    //   GRID   : the rim band |z| < 0.05 holds 1,192 anchors BELOW the
    //            seam and 360 above -- the rim straddles z=0.
    //   DERIVED: the same band holds 48 below and 792 ABOVE -- the lower
    //            sheet's rim has been pushed to the upper side.
    //
    // And because the derived path's rim points have local z EXACTLY 0
    // (the grid path's rim points land at z ~= 8e-6 by accident of the
    // lattice), only the derived path hits the midpoint and only the
    // derived path is broken.
    //
    // THE FIX: the seam's quantised index is computed ONCE, from the
    // seam's world position, and any point that lies ON the seam is
    // snapped to it. Both sheets' rims then land on the identical index,
    // so the rims coincide exactly -- which is what the closed-body
    // mechanism needs.
    //
    // The seam is the plane z = 0 (the standard recipe's origin). It is
    // detected by the CALLER's own knowledge of the sheet's placement: a
    // point whose world z is within a quantisation step of the SEAM PLANE
    // is on the seam. The plane is z = 0 because that is where the closed
    // body's two sheets meet by construction; a body with a different
    // seam plane gets it from the same rule (the sheet's origin).
    const float seam_z = 0.f;
    const float qstep_z = bsz / 65535.f;
    // THE SEAM'S INDEX: the one that DEQUANTISES to the seam, not the one
    // whose PARAMETER is the seam's parameter.
    //
    // These differ by half a step, and that half-step IS the bug. The
    // seam's parameter is t = 0.5 exactly (the body is symmetric about it),
    // and the plain rounding sends t=0.5 to the index 32768 -- but
    // 32768/65535 = 0.50000763, so that index dequantises to +0.5
    // quantisation steps ABOVE the seam, not to the seam.
    //
    // The index that dequantises to the seam is the one whose DEQUANTISED
    // value is nearest the seam -- which is found by inverting the
    // dequantisation, not by quantising the seam's parameter:
    //
    //     q = round((seam_z - bbmin.z) / bsz * 65535)   is the PARAMETER
    //     but the dequantisation is bbmin.z + q/65535*bsz
    //
    // so the index nearest the seam is the one that minimises
    // |bbmin.z + q/65535*bsz - seam_z| -- which, for a symmetric body, is
    // the index just BELOW the midpoint: 32767 (not 32768).
    //
    // MEASURED: with the index at 32768 the snapped anchors dequantised to
    // +0.0000081 -- still above the seam, so the snap moved them nowhere.
    // With 32767 they dequantise to -0.0000080 -- on the other side, and
    // IDENTICALLY for both sheets (which is what the mechanism needs: the
    // two rims must land on the SAME index).
    const int seam_q = [&] {
        int best = 0;
        float best_d = 1e30f;
        for (int cand = (int)((std::max(0.f, std::min(1.f,
                                 (seam_z - bbmin.z) / bsz))) * 65535.f) - 2;
             cand <= (int)((std::max(0.f, std::min(1.f,
                                 (seam_z - bbmin.z) / bsz))) * 65535.f) + 2; ++cand) {
            if (cand < 0 || cand > 65535) continue;
            const float deq = bbmin.z + (float)cand / 65535.f * bsz;
            const float d = std::fabs(deq - seam_z);
            if (d < best_d) { best_d = d; best = cand; }
        }
        return best;
    }();
    int near_seam_missed = 0;
    float near_seam_worst = 0.f;
    (void)near_seam_missed; (void)near_seam_worst;

    for (size_t i = 0; i < cloud.size(); ++i) {
        AnchorRecord a{};
        a.bone_index = 0xFFFFFFFFu;
        a.connectivity[0] = a.connectivity[1] = a.connectivity[2] = a.connectivity[3] = 0xFFFF;
        const Vec3& p = wpos[i];
        auto q = [&](float v, float lo, float sp2) -> uint16_t {
            const float t = (v - lo) / sp2;
            const int qv = (int)(std::max(0.f, std::min(1.f, t)) * 65535.f + 0.5f);
            return (uint16_t)qv;
        };
        a.pos[0] = q(p.x, bbmin.x, bsx);
        a.pos[1] = q(p.y, bbmin.y, bsy);
        // A point ON the seam is snapped to the seam's own quantised index,
        // so the two sheets' rims land on the IDENTICAL value. Without the
        // snap they differ by one quantisation step and the seam shows as
        // a step in the render.
        if (std::fabs(p.z - seam_z) <= qstep_z * 0.5f) {
            a.pos[2] = (uint16_t)seam_q;
        } else {
            a.pos[2] = q(p.z, bbmin.z, bsz);
            // DIAGNOSTIC: how many points sit just off the seam, and by how
            // much? If the count is large, the seam's tolerance is too
            // tight for the quantisation's own resolution.
            if (std::fabs(p.z - seam_z) < qstep_z * 4.f) {
                ++near_seam_missed;
                near_seam_worst = std::max(near_seam_worst, std::fabs(p.z - seam_z));
            }
        }
        const Vec3& n = wnrm[i];
        a.normal[0] = q8(n.x * 0.5f + 0.5f);
        a.normal[1] = q8(n.y * 0.5f + 0.5f);
        a.normal[2] = q8(n.z * 0.5f + 0.5f);
        a.color[0] = q8(sheet_el.color.x);
        a.color[1] = q8(sheet_el.color.y);
        a.color[2] = q8(sheet_el.color.z);
        a.roughness = 128; a.metallic = 0; a.anim_weight = 0;
        anchors.push_back(a);
    }

    // ---- 4. the patches ----------------------------------------------------
    //
    // Each derived cell becomes a patch. Its four corners are its four
    // points (in the order the derivation fixed, so the winding agrees with
    // the surface normal).
    //
    // THE LSQ TARGET, in the patch's OWN frame: the target at the patch's
    // (u,v) is the BILINEAR INTERPOLATION of the quad's four corner
    // points, projected onto the surface along the interpolated normal.
    // That is the same "the surface passes through the corners, and the
    // deformation carries the detail" contract the grid path implements.
    const int G = 8;   // the LSQ sample grid per patch
    for (const MeshQuad& q : stats.quads) {
        // the four corner positions and normals, DEQUANTIZED (the renderer
        // will see the quantised values, so the fit must use them)
        Vec3 CA[4], CN[4];
        for (int k = 0; k < 4; ++k) {
            const AnchorRecord& a = anchors[base + q.v[k]];
            CA[k] = Vec3{ bbmin.x + a.pos[0] / 65535.f * bsx,
                          bbmin.y + a.pos[1] / 65535.f * bsy,
                          bbmin.z + a.pos[2] / 65535.f * bsz };
            CN[k] = Vec3{ a.normal[0] * (2.f/255.f) - 1.f,
                          a.normal[1] * (2.f/255.f) - 1.f,
                          a.normal[2] * (2.f/255.f) - 1.f };
            const float l = std::sqrt(CN[k].x*CN[k].x + CN[k].y*CN[k].y + CN[k].z*CN[k].z);
            if (l > 1e-12f) { CN[k].x /= l; CN[k].y /= l; CN[k].z /= l; }
        }
        // the patch's bilinear base and normal field
        double M[8][8] = {{0}}, r[8] = {0};
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float cu = i / float(G-1), cv = j / float(G-1);
                const double uu = cu, vv = cv;
                // the bilinear base (the same convention as the kernel:
                // pl = mix(A0,A3,u), pr = mix(A1,A2,u), P = mix(pl,pr,v))
                const Vec3 pl{ CA[0].x + (CA[3].x-CA[0].x)*uu,
                               CA[0].y + (CA[3].y-CA[0].y)*uu,
                               CA[0].z + (CA[3].z-CA[0].z)*uu };
                const Vec3 pr{ CA[1].x + (CA[2].x-CA[1].x)*uu,
                               CA[1].y + (CA[2].y-CA[1].y)*uu,
                               CA[1].z + (CA[2].z-CA[1].z)*uu };
                const Vec3 bil{ pl.x + (pr.x-pl.x)*vv,
                                pl.y + (pr.y-pl.y)*vv,
                                pl.z + (pr.z-pl.z)*vv };
                // the interpolated normal field, normalised
                const Vec3 nl{ CN[0].x + (CN[3].x-CN[0].x)*uu,
                               CN[0].y + (CN[3].y-CN[0].y)*uu,
                               CN[0].z + (CN[3].z-CN[0].z)*uu };
                const Vec3 nr{ CN[1].x + (CN[2].x-CN[1].x)*uu,
                               CN[1].y + (CN[2].y-CN[1].y)*uu,
                               CN[1].z + (CN[2].z-CN[1].z)*uu };
                Vec3 nf{ nl.x + (nr.x-nl.x)*vv, nl.y + (nr.y-nl.y)*vv,
                         nl.z + (nr.z-nl.z)*vv };
                const float nl_len = std::sqrt(nf.x*nf.x + nf.y*nf.y + nf.z*nf.z);
                if (nl_len > 1e-12f) { nf.x /= nl_len; nf.y /= nl_len; nf.z /= nl_len; }
                // THE TARGET: the surface point the patch should pass
                // through at (u,v).
                //
                // For a GRID patch the target is the substance evaluated on
                // the sheet's domain at that (u,v). A DERIVED patch has no
                // domain position -- its (u,v) is its own bilinear. So the
                // target is the substance's surface NEAR the patch: the
                // point on the substance closest to the patch's bilinear
                // point, found by projecting along the interpolated normal.
                //
                // The projection is the same trick the grid path uses
                // implicitly (the sheet's own field, evaluated at the
                // domain position): take the bilinear point, and walk it
                // onto the surface along the normal field until it is on
                // the substance. The substance is a GRAPH z=f(x,y), so the
                // walk is: solve z = f(x,y) for the z that puts the point
                // on the graph, keeping x,y fixed.
                Vec3 tgt = bil;
                {
                    // invert the sheet's placement: the bilinear point is in
                    // WORLD space; the substance's Eval wants the sheet's
                    // LOCAL space.
                    const float inv_x = (scale.x != 0.f) ? 1.f/scale.x : 1.f;
                    const float inv_y = (scale.y != 0.f) ? 1.f/scale.y : 1.f;
                    const float inv_z = (scale.z != 0.f) ? 1.f/scale.z : 1.f;
                    const float lx = (bil.x - origin.x) * inv_x;
                    const float ly = (bil.y - origin.y) * inv_y;
                    const SubstancePoint sp = sheet_el.substance->Eval(lx, ly);
                    if (sp.valid) {
                        tgt = Vec3{ origin.x + sp.p.x * scale.x,
                                    origin.y + sp.p.y * scale.y,
                                    origin.z + sp.p.z * scale.z };
                    }
                }
                // the height along the interpolated normal, relative to the
                // bilinear base -- the SAME quantity the grid path fits.
                const double h = (tgt.x - bil.x) * nf.x
                               + (tgt.y - bil.y) * nf.y
                               + (tgt.z - bil.z) * nf.z;
                double B[8]; cv_basis((float)cu, (float)cv, B);
                for (int x = 0; x < 8; ++x) {
                    r[x] += B[x] * h;
                    for (int y = 0; y < 8; ++y) M[x][y] += B[x] * B[y];
                }
            }
        SurfaceRecord s{};
        // THE CORNER ORDER IS THE QUAD'S, STRAIGHT THROUGH.
        //
        // The quad's order is a BOUNDARY order, and the engine's convention
        // is A0=(0,0), A1=(1,0), A2=(1,1), A3=(0,1) -- but those are the
        // SAME thing for a boundary order walked one way or the other, and
        // WHICH one is correct is decided by OrderQuad, which now uses the
        // engine's own normal formula (cross(A3-A0, A1-A0)) to pick the
        // candidate whose sign agrees with the surface.
        //
        // So the quad's corners go in as they are. An earlier attempt
        // permuted them here as well -- which would have DOUBLE-permuted
        // them and re-broken the winding. The single point of truth is
        // OrderQuad.
        for (int k = 0; k < 4; ++k) s.anchor_idx[k] = (uint16_t)(base + q.v[k]);
        s.surface_def = 0;
        if (!solve8(M, r, s.coef))
            for (int x = 0; x < 8; ++x) s.coef[x] = 0.f;
        // the constant colour, the same encoding the grid path uses
        for (int ch = 0; ch < 3; ++ch) {
            const float val = ch == 0 ? sheet_el.color.x
                            : ch == 1 ? sheet_el.color.y : sheet_el.color.z;
            uint8_t* cp = s.color_pattern + ch * 6;
            cp[0] = 0x00; cp[1] = 0x00;
            int8_t* K = reinterpret_cast<int8_t*>(cp + 2);
            int kv = (int)(std::max(0.f, std::min(1.f, val)) * 63.f + 0.5f);
            if (kv > 127) kv = 127;
            K[0] = (int8_t)kv; K[1] = K[2] = K[3] = 0;
        }
        surfaces.push_back(s);
    }

    // ---- 5. the CONNECTIVITY, from the derived adjacency -------------------
    //
    // Two cells are neighbours when they share TWO vertices (an edge).
    // The grid path derives this from the lattice; the derived path derives
    // it from the cells themselves.
    //
    // WHY IT MATTERS: the anchor's connectivity slots are what the
    // "surround" relation is stored in -- the anchor's four neighbours.
    // That is the SAME relation SeamGraph::Build reconstructs for the seams,
    // and it is what makes a pose change move the skin with the bone.
    {
        // the vertices each cell uses
        std::map<uint32_t, std::vector<size_t>> by_vertex;
        for (size_t c = 0; c < stats.quads.size(); ++c)
            for (int k = 0; k < 4; ++k)
                by_vertex[(uint32_t)(base + stats.quads[c].v[k])].push_back(c);
        // for each cell, the cells that share an edge with it
        std::map<size_t, std::vector<size_t>> edge_neighbours;
        for (auto& [v, cells] : by_vertex) {
            for (size_t i = 0; i < cells.size(); ++i)
                for (size_t j = i + 1; j < cells.size(); ++j) {
                    const MeshQuad& qa = stats.quads[cells[i]];
                    const MeshQuad& qb = stats.quads[cells[j]];
                    int shared = 0;
                    for (int a = 0; a < 4; ++a)
                        for (int b = 0; b < 4; ++b)
                            if (qa.v[a] == qb.v[b]) ++shared;
                    if (shared >= 2) {
                        edge_neighbours[cells[i]].push_back(cells[j]);
                        edge_neighbours[cells[j]].push_back(cells[i]);
                    }
                }
        }
        // the anchor's connectivity: the cells that touch it, and through
        // them the anchors that share an EDGE with it (which is what a
        // vertex's neighbours are).
        for (auto& [v, cells] : by_vertex) {
            if (v >= anchors.size()) continue;
            std::vector<uint32_t> nb;
            for (size_t c : cells) {
                for (size_t d : edge_neighbours[c]) {
                    const MeshQuad& q = stats.quads[d];
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t a = (uint32_t)(base + q.v[k]);
                        if (a == v) continue;
                        if (std::find(nb.begin(), nb.end(), a) == nb.end())
                            nb.push_back(a);
                    }
                }
            }
            for (int k = 0; k < 4 && (size_t)k < nb.size(); ++k)
                anchors[v].connectivity[k] = nb[k];
        }
    }

    return true;
}

} // namespace DeliRium
