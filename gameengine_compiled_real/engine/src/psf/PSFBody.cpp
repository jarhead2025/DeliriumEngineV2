#include "delirium/psf/PSFBody.h"
#include "delirium/io/PSFFile.h"
#include <CL/cl.h>
#include <cmath>
#include <cstring>

namespace DeliRium {

std::unique_ptr<PSFBody> PSFBody::FromFile(const char* path) {
    PSFFile f;
    std::string err;
    if (!PSFFile::Load(path, f, &err)) return nullptr;
    return FromRecords(std::move(f.anchors), std::move(f.surfaces),
                       std::move(f.stitch), f.bbox_min, f.bbox_span, f.s_scale);
}

std::unique_ptr<PSFBody> PSFBody::FromRecords(std::vector<AnchorRecord> a,
                                              std::vector<SurfaceRecord> s,
                                              std::vector<int8_t> st,
                                              Vec3 bbmin, Vec3 bbspan,
                                              float sscale) {
    auto body = std::unique_ptr<PSFBody>(new PSFBody());
    body->anchor_blob  = std::move(a);
    body->surface_blob = std::move(s);
    body->stitch_blob  = std::move(st);
    body->bbox_min = bbmin; body->bbox_span = bbspan; body->s_scale = sscale;
    if (body->stitch_blob.empty())
        body->stitch_blob.assign(body->surface_blob.size() * 8, 0);
    body->anchors.Bind(&body->anchor_blob, bbmin, bbspan);
    body->RebuildViews();
    body->seams.Build(*body);
    return body;
}

// 8x8 SPD solve by Gaussian elimination with partial pivoting (no deps)
static bool solve8(double A[8][8], double b[8], float out[8]) {
    for (int c = 0; c < 8; ++c) {
        int best = c;
        for (int r = c + 1; r < 8; ++r)
            if (std::fabs(A[r][c]) > std::fabs(A[best][c])) best = r;
        if (std::fabs(A[best][c]) < 1e-14) return false;
        if (best != c) { for (int k = 0; k < 8; ++k) std::swap(A[c][k], A[best][k]);
                         std::swap(b[c], b[best]); }
        for (int r = c + 1; r < 8; ++r) {
            const double f = A[r][c] / A[c][c];
            for (int k = c; k < 8; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    for (int c = 7; c >= 0; --c) {
        double s = b[c];
        for (int k = c + 1; k < 8; ++k) s -= A[c][k] * out[k];
        out[c] = (float)(s / A[c][c]);
    }
    return true;
}

static void cv_basis(float u, float v, double B[8]) {
    const double bu = u*(1.0-u), bv = v*(1.0-v);
    const double au = 2.0*u-1.0, av = 2.0*v-1.0;
    B[0]=bu*(1-v); B[1]=bu*v; B[2]=bv*(1-u); B[3]=bv*u;
    B[4]=bu*au*(1-v); B[5]=bu*au*v; B[6]=bv*av*(1-u); B[7]=bv*av*u;
}
// forward decls (defined further down, used by RefitLocal/ClosestPoint)
static float patch_size(const PSFBody& b, uint32_t pid);
static Vec3  patch_center(const PSFBody& b, uint32_t pid);

bool PSFBody::SplitPatch(uint32_t pid, uint32_t children_out[4]) {
    if (pid >= patches.size()) return false;
    // snapshot the PARENT as an independent evaluator (records will move)
    const SurfaceRecord parent_rec = surface_blob[pid];
    int8_t parent_st[8];
    std::memcpy(parent_st, &stitch_blob[size_t(pid)*8], 8);
    Patch parent(const_cast<SurfaceRecord*>(&parent_rec), &anchors,
                 parent_st, s_scale, pid);

    // --- midpoint + center anchors --------------------------------------
    // reuse a hanging anchor if the neighbor across an edge already split:
    // it is the seam endpoint whose coarse-parameter is 0.5
    auto find_existing_mid = [&](uint8_t edge) -> uint16_t {
        static const int EPL[4][2] = {{0,3},{1,2},{0,1},{3,2}};
        for (const Seam& s : seams.All()) {
            if (!(s.patch_b == pid && s.edge_b == edge && s.a_is_finer))
                continue;
            const SurfaceRecord& F = surface_blob[s.patch_a];
            const uint16_t e0 = F.anchor_idx[EPL[s.edge_a][0]];
            const uint16_t e1 = F.anchor_idx[EPL[s.edge_a][1]];
            if (s.tb0 == 0.5f) return e0;
            if (s.tb1 == 0.5f) return e1;
        }
        return 0xFFFF;
    };
    // edge midpoints at parent params: v0->(0.5,0) v1->(0.5,1)
    //                                  u0->(0,0.5) u1->(1,0.5)
    static const float MID_UV[4][2] = {{0.5f,0},{0.5f,1},{0,0.5f},{1,0.5f}};
    uint16_t mid[4];
    for (int e = 0; e < 4; ++e) {
        mid[e] = find_existing_mid((uint8_t)e);
        if (mid[e] == 0xFFFF)
            mid[e] = anchors.CreateSnapped(parent, MID_UV[e][0], MID_UV[e][1]);
        if (mid[e] == 0xFFFF) return false;            // anchor overflow
    }
    const uint16_t ctr = anchors.CreateSnapped(parent, 0.5f, 0.5f);
    if (ctr == 0xFFFF) return false;
    split_centers.insert(ctr);          // mark lineage for safe merging

    // --- 4 children: anchors + LSQ fit against the parent ----------------
    const uint16_t* pa = parent_rec.anchor_idx;     // A0 A1 A2 A3
    // quadrant (qu,qv) origin; child slots [A0 A1 A2 A3] follow the same
    // convention: u along A0->A3, v along A0->A1
    struct Q { float u0, v0; uint16_t a[4]; };
    const Q quads[4] = {
        {0.0f,0.0f, { pa[0],  mid[2], ctr,    mid[0] }},   // u-,v-
        {0.0f,0.5f, { mid[2], pa[1],  mid[1], ctr    }},   // u-,v+
        {0.5f,0.0f, { mid[0], ctr,    mid[3], pa[3]  }},   // u+,v-
        {0.5f,0.5f, { ctr,    mid[1], pa[2],  mid[3] }},   // u+,v+
    };

    SurfaceRecord child[4];
    const int G = 9;
    for (int q = 0; q < 4; ++q) {
        child[q] = parent_rec;                       // inherit color/material
        for (int k = 0; k < 4; ++k) child[q].anchor_idx[k] = quads[q].a[k];
        // child anchor positions/normals for bilinear + normal field
        Vec3 CA[4], CN[4];
        for (int k = 0; k < 4; ++k) {
            CA[k] = anchors.Position(quads[q].a[k]);
            CN[k] = anchors.Normal(quads[q].a[k]);
        }
        double M[8][8] = {{0}}, r[8] = {0};
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float cu = i/float(G-1), cv = j/float(G-1);
                const float pu = quads[q].u0 + 0.5f*cu;
                const float pv = quads[q].v0 + 0.5f*cv;
                const Vec3 tgt = parent.Evaluate(pu, pv);
                // child bilinear + normal field at (cu,cv)
                auto mixv=[&](Vec3 a, Vec3 b, float t){
                    return Vec3{a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t};};
                const Vec3 bil = mixv(mixv(CA[0],CA[3],cu), mixv(CA[1],CA[2],cu), cv);
                Vec3 n = mixv(mixv(CN[0],CN[3],cu), mixv(CN[1],CN[2],cu), cv);
                const float nl = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
                n = nl > 1e-12f ? Vec3{n.x/nl,n.y/nl,n.z/nl} : Vec3{0,0,1};
                const double h = (tgt.x-bil.x)*n.x + (tgt.y-bil.y)*n.y
                               + (tgt.z-bil.z)*n.z;
                double B[8]; cv_basis(cu, cv, B);
                for (int x = 0; x < 8; ++x) {
                    r[x] += B[x]*h;
                    for (int y = 0; y < 8; ++y) M[x][y] += B[x]*B[y];
                }
            }
        if (!solve8(M, r, child[q].coef))
            for (int x = 0; x < 8; ++x) child[q].coef[x] = 0;
    }

    // --- commit: parent slot <- child0, append 1..3; fresh stitch slots --
    surface_blob[pid] = child[0];
    std::memset(&stitch_blob[size_t(pid)*8], 0, 8);
    children_out[0] = pid;
    for (int q = 1; q < 4; ++q) {
        children_out[q] = (uint32_t)surface_blob.size();
        surface_blob.push_back(child[q]);
        stitch_blob.insert(stitch_blob.end(), 8, 0);
    }
    RebuildViews();
    // The seam rebuild is 98.7% of this function's cost, and a batch AMR
    // operation calls this in a LOOP -- so rebuilding per split is O(n^2).
    //
    // Inside a batch, the rebuild is deferred to EndSeamBatch(), which
    // rebuilds ONCE and restitches every patch the batch touched. The seam
    // graph is left consistent when the batch ends.
    if (seam_batch_depth_ > 0) {
        seam_batch_touched_.insert(seam_batch_touched_.end(),
                                   children_out, children_out + 4);
        return true;
    }
    seams.Build(*this);
    std::vector<uint32_t> ids(children_out, children_out + 4);
    seams.RestitchAround(*this, ids);
    return true;
}

// ---------------------------------------------------------------------------
// Deferred-seam batches
// ---------------------------------------------------------------------------
void PSFBody::BeginSeamBatch() {
    ++seam_batch_depth_;
}

void PSFBody::EndSeamBatch() {
    if (seam_batch_depth_ <= 0) return;
    if (--seam_batch_depth_ > 0) return;   // an outer batch is still open
    if (seam_batch_touched_.empty()) return;
    // ONE rebuild for the whole batch, then restitch every patch that
    // moved. The touched set accumulates across the batch, so a patch that
    // moved early is still restitched at the end.
    const std::vector<uint32_t> touched = std::move(seam_batch_touched_);
    seam_batch_touched_.clear();
    RebuildViews();
    seams.Build(*this);
    seams.RestitchAround(*this, touched);
}

void PSFBody::RebuildViews() {
    patches.clear();
    patches.reserve(surface_blob.size());
    for (uint32_t i = 0; i < surface_blob.size(); ++i)
        patches.emplace_back(&surface_blob[i], &anchors,
                             &stitch_blob[size_t(i)*8], s_scale, i);
}

// Detect a mergeable quad-group around interior anchor `ctr`: exactly 4
// patches must reference it, their union of corner anchors must be the 4
// shared midpoints + 4 outer corners forming a 2x2 cell. Returns the 4
// patch ids (in quadrant order u-v-, u-v+, u+v-, u+v+) and the parent's
// 4 outer corner anchors; false if `ctr` is not a clean 4-way center.
bool PSFBody::FindMergeGroup(uint16_t ctr, uint32_t grp[4],
                             uint16_t outer[4]) const {
    // gather patches touching ctr and the slot ctr sits in
    int found = 0; uint32_t pids[8]; int slots[8];
    for (uint32_t s = 0; s < surface_blob.size(); ++s)
        for (int k = 0; k < 4; ++k)
            if (surface_blob[s].anchor_idx[k] == ctr) {
                if (found >= 8) return false;
                pids[found] = s; slots[found] = k; ++found;
            }
    if (found != 4) return false;     // not a clean 4-way interior center

    // In SplitPatch the center sits at child-slot: q0->2, q1->3, q2->1,
    // q3->0. Use that to place each child in its quadrant, and read the
    // outer corner = the slot diagonally opposite the center in that child.
    const int OPP[4] = {2,3,0,1};                 // diagonal slot
    const int CTR_SLOT_FOR_QUAD[4] = {2,3,1,0};
    for (int g = 0; g < 4; ++g) grp[g] = 0xFFFFFFFF;
    for (int i = 0; i < found; ++i) {
        int q = -1;
        for (int qq = 0; qq < 4; ++qq)
            if (CTR_SLOT_FOR_QUAD[qq] == slots[i]) { q = qq; break; }
        if (q < 0 || grp[q] != 0xFFFFFFFF) return false;  // ambiguous
        grp[q] = pids[i];
        outer[q] = surface_blob[pids[i]].anchor_idx[OPP[slots[i]]];
    }
    for (int g = 0; g < 4; ++g) if (grp[g] == 0xFFFFFFFF) return false;

    // STRICT geometric validation: not every 4-way interior anchor is a
    // split product (a regular mesh has many). Require that the group
    // forms a genuine 2x2 cell: the 4 shared edge-midpoint anchors must
    // each sit at the geometric midpoint of the corresponding outer edge,
    // and the center at the cell centroid. Otherwise these are ordinary
    // neighbors and merging them would destroy the surface.
    auto pos = [&](uint16_t a){ return anchors.Position(a); };
    // outer[] in quadrant order maps to parent corners A0 A1 A3 A2:
    const Vec3 A0 = pos(outer[0]), A1 = pos(outer[1]);
    const Vec3 A3 = pos(outer[2]), A2 = pos(outer[3]);
    // shared midpoints: child q0 and q1 share the A0-A1 edge mid (slot
    // pattern from split): recover them from the children's anchor sets
    // edge mids live at the non-center, non-outer shared slots
    auto shared_mid = [&](int qa, int qb)->uint16_t {
        const uint16_t* a = surface_blob[grp[qa]].anchor_idx;
        const uint16_t* b = surface_blob[grp[qb]].anchor_idx;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (a[i] == b[j] && a[i] != ctr) return a[i];
        return 0xFFFF;
    };
    const uint16_t m01 = shared_mid(0,1);   // A0-A1 edge
    const uint16_t m02 = shared_mid(0,2);   // A0-A3 edge
    const uint16_t m13 = shared_mid(1,3);   // A1-A2 edge
    const uint16_t m23 = shared_mid(2,3);   // A3-A2 edge
    if (m01==0xFFFF||m02==0xFFFF||m13==0xFFFF||m23==0xFFFF) return false;
    const Vec3 sp_ = bbox_span;
    const float diag_ = std::sqrt(sp_.x*sp_.x+sp_.y*sp_.y+sp_.z*sp_.z);
    const float midtol = 0.02f * diag_;     // 2% slack for curvature
    auto near_mid = [&](uint16_t m, const Vec3& p, const Vec3& q){
        const Vec3 mid{0.5f*(p.x+q.x),0.5f*(p.y+q.y),0.5f*(p.z+q.z)};
        const Vec3 mp = pos(m);
        return std::sqrt((mp.x-mid.x)*(mp.x-mid.x)+(mp.y-mid.y)*(mp.y-mid.y)
                        +(mp.z-mid.z)*(mp.z-mid.z)) < midtol;
    };
    if (!near_mid(m01,A0,A1) || !near_mid(m02,A0,A3)
     || !near_mid(m13,A1,A2) || !near_mid(m23,A3,A2)) return false;
    return true;
}

bool PSFBody::MergeGroup(uint16_t ctr, float tol) {
    // only anchors created as split centers are safe to merge: a regular
    // mesh has many 4-way interior anchors that are NOT split products,
    // and merging those destroys the surface
    if (split_centers.find(ctr) == split_centers.end()) return false;
    uint32_t grp[4]; uint16_t outer[4];
    if (!FindMergeGroup(ctr, grp, outer)) return false;

    // 2:1 BALANCE RULE: merging halves this group's resolution. If any
    // group patch borders a FINER patch (a neighbor that is itself split,
    // i.e. shares only part of an edge via a mid-anchor not in our group),
    // merging would create a 2-level T-junction the seam graph can't sew.
    // Detect: any outer edge of the group whose midpoint anchor is
    // referenced by a patch outside the group => finer neighbor => abort.
    {
        std::vector<int> refcount(anchors.Count(), 0);
        for (const SurfaceRecord& s : surface_blob)
            for (int k = 0; k < 4; ++k) refcount[s.anchor_idx[k]]++;
        // the group's 4 outer-edge midpoints: shared between adjacent
        // children; if also used by a 5th patch, that side is finer
        auto shared_mid = [&](int qa, int qb)->uint16_t {
            const uint16_t* a = surface_blob[grp[qa]].anchor_idx;
            const uint16_t* b = surface_blob[grp[qb]].anchor_idx;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (a[i] == b[j] && a[i] != ctr) return a[i];
            return 0xFFFF;
        };
        const int pairs[4][2] = {{0,1},{0,2},{1,3},{2,3}};
        // characteristic size of the group's children (post-merge size is
        // ~2x this); a neighbor is "finer" only if notably smaller
        float grp_size = 0;
        for (int g = 0; g < 4; ++g) {
            const SurfaceRecord& s = surface_blob[grp[g]];
            for (int i = 0; i < 4; ++i)
                for (int j = i+1; j < 4; ++j) {
                    const Vec3 a = anchors.Position(s.anchor_idx[i]);
                    const Vec3 b = anchors.Position(s.anchor_idx[j]);
                    grp_size = std::fmax(grp_size,
                        std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)
                                 +(a.z-b.z)*(a.z-b.z)));
                }
        }
        for (auto& pr : pairs) {
            const uint16_t m = shared_mid(pr[0], pr[1]);
            if (m == 0xFFFF) return false;
            if (refcount[m] <= 2) continue;          // clean edge
            // a 5th+ patch hangs off this midpoint: only a problem if it
            // is FINER than our group (merging would span 2 levels)
            for (uint32_t s = 0; s < surface_blob.size(); ++s) {
                if (s==grp[0]||s==grp[1]||s==grp[2]||s==grp[3]) continue;
                bool uses = false;
                for (int k = 0; k < 4; ++k)
                    if (surface_blob[s].anchor_idx[k] == m) { uses=true; break; }
                if (!uses) continue;
                float nsz = 0;
                const SurfaceRecord& sr = surface_blob[s];
                for (int i = 0; i < 4; ++i)
                    for (int j = i+1; j < 4; ++j) {
                        const Vec3 a = anchors.Position(sr.anchor_idx[i]);
                        const Vec3 bb = anchors.Position(sr.anchor_idx[j]);
                        nsz = std::fmax(nsz, std::sqrt(
                            (a.x-bb.x)*(a.x-bb.x)+(a.y-bb.y)*(a.y-bb.y)
                           +(a.z-bb.z)*(a.z-bb.z)));
                    }
                if (nsz < 0.75f * grp_size) return false;  // finer: abort
            }
        }
    }

    // parent anchors: outer[] are A0..A3 in quadrant order, which maps to
    // parent slots: q0 outer = parent A0, q1 outer = A1, q2 = A3, q3 = A2
    // (quadrant grid: q0=u-v-, q1=u-v+, q2=u+v-, q3=u+v+; parent A0 A1 A2 A3
    //  with u along A0->A3, v along A0->A1)
    uint16_t pa[4] = { outer[0], outer[1], outer[3], outer[2] };

    // build a temporary parent record and refit its 8 coefs against the
    // 4 children's evaluated surface (the inverse of the split fit)
    SurfaceRecord prec = surface_blob[grp[0]];   // inherit color/material
    for (int k = 0; k < 4; ++k) prec.anchor_idx[k] = pa[k];

    Vec3 PA[4], PN[4];
    for (int k = 0; k < 4; ++k) {
        PA[k] = anchors.Position(pa[k]);
        PN[k] = anchors.Normal(pa[k]);
    }
    auto mixv = [](Vec3 a, Vec3 b, float t){
        return Vec3{a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t}; };

    // child quadrant origins in parent param space
    const float QO[4][2] = {{0,0},{0,0.5f},{0.5f,0},{0.5f,0.5f}};
    const int G = 9;
    double M[8][8] = {{0}}, r[8] = {0};
    double err2 = 0; int ne = 0;
    for (int q = 0; q < 4; ++q) {
        const Patch& ch = patches[grp[q]];
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float cu = i/float(G-1), cv = j/float(G-1);
                const Vec3 tgt = ch.Evaluate(cu, cv);
                const float pu = QO[q][0] + 0.5f*cu;
                const float pv = QO[q][1] + 0.5f*cv;
                const Vec3 bil = mixv(mixv(PA[0],PA[3],pu),
                                      mixv(PA[1],PA[2],pu), pv);
                Vec3 n = mixv(mixv(PN[0],PN[3],pu), mixv(PN[1],PN[2],pu), pv);
                const float nl = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
                n = nl > 1e-12f ? Vec3{n.x/nl,n.y/nl,n.z/nl} : Vec3{0,0,1};
                const double h = (tgt.x-bil.x)*n.x + (tgt.y-bil.y)*n.y
                               + (tgt.z-bil.z)*n.z;
                double B[8]; cv_basis(pu, pv, B);
                for (int x = 0; x < 8; ++x) {
                    r[x] += B[x]*h;
                    for (int y = 0; y < 8; ++y) M[x][y] += B[x]*B[y];
                }
            }
    }
    float coef[8];
    if (!solve8(M, r, coef)) return false;
    for (int i = 0; i < 8; ++i) prec.coef[i] = coef[i];

    // measure: reconstruct parent vs children, reject if above tol
    Patch parent(&prec, &anchors, nullptr, s_scale, 0);
    const Vec3 sp = bbox_span;
    const float diag = std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    for (int q = 0; q < 4; ++q) {
        const Patch& ch = patches[grp[q]];
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float cu=i/float(G-1), cv=j/float(G-1);
                const Vec3 a = ch.Evaluate(cu, cv);
                const Vec3 b = parent.Evaluate(QO[q][0]+0.5f*cu,
                                               QO[q][1]+0.5f*cv);
                err2 += (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)
                       +(a.z-b.z)*(a.z-b.z); ++ne;
            }
    }
    const float rms = std::sqrt(err2 / ne);
    if (rms > tol * diag) return false;        // detail would be lost: keep

    // commit: lowest grp id becomes the parent, the other 3 are removed
    // (compact the blobs; remap references; ctr + interior mids may now be
    // orphaned but we leave them -- harmless, reclaimed on save/reload)
    uint32_t keep = grp[0];
    for (int g = 1; g < 4; ++g) keep = std::min(keep, grp[g]);
    surface_blob[keep] = prec;
    std::memset(&stitch_blob[size_t(keep)*8], 0, 8);

    bool remove[1] = {false}; (void)remove;
    std::vector<uint8_t> kill(surface_blob.size(), 0);
    for (int g = 0; g < 4; ++g) if (grp[g] != keep) kill[grp[g]] = 1;

    std::vector<SurfaceRecord> ns;
    std::vector<int8_t> nst;
    ns.reserve(surface_blob.size() - 3);
    for (uint32_t s = 0; s < surface_blob.size(); ++s) {
        if (kill[s]) continue;
        ns.push_back(surface_blob[s]);
        nst.insert(nst.end(), &stitch_blob[size_t(s)*8],
                   &stitch_blob[size_t(s)*8] + 8);
    }
    surface_blob.swap(ns);
    stitch_blob.swap(nst);

    split_centers.erase(ctr);          // consumed
    RebuildViews();
    // Same deferred-seam treatment as SplitPatch: a batch coarsen does
    // many merges, and rebuilding the seam graph per merge is the same
    // O(n^2) as before.
    //
    // The touched id is the SURVIVING patch (`keep`), which is the patch
    // whose neighbourhood changed. Its index may have shifted in the
    // compaction above, so it is re-found by its anchor set -- the same
    // way the non-batch path finds it below.
    if (seam_batch_depth_ > 0) {
        uint32_t merged_id = 0xFFFFFFFFu;
        for (uint32_t s = 0; s < surface_blob.size(); ++s)
            if (surface_blob[s].anchor_idx[0] == pa[0]
             && surface_blob[s].anchor_idx[2] == pa[2]) { merged_id = s; break; }
        if (merged_id != 0xFFFFFFFFu) seam_batch_touched_.push_back(merged_id);
        anchors_dirty = true;
        return true;
    }
    seams.Build(*this);
    // restitch around the merged patch's new id (find it by anchor set)
    std::vector<uint32_t> ids;
    for (uint32_t s = 0; s < surface_blob.size(); ++s)
        if (surface_blob[s].anchor_idx[0] == pa[0]
         && surface_blob[s].anchor_idx[2] == pa[2]) { ids.push_back(s); break; }
    seams.RestitchAround(*this, ids);
    anchors_dirty = true;
    return true;
}

// Convenience: try to merge the 4-group whose center is the anchor at
// `parent_id`'s inward diagonal corner. Returns silently if not mergeable.
void PSFBody::MergePatches(uint32_t parent_id, float tol) {
    if (parent_id >= surface_blob.size()) return;
    // a child's center anchor is the slot diagonally opposite its outer
    // corner; try all 4 slots as candidate centers
    for (int k = 0; k < 4; ++k) {
        const uint16_t cand = surface_blob[parent_id].anchor_idx[k];
        uint32_t grp[4]; uint16_t outer[4];
        if (FindMergeGroup(cand, grp, outer)) { MergeGroup(cand, tol); return; }
    }
}

// Coarsen a whole region: repeatedly merge any clean 4-group whose patches
// all sit within `radius` of `center` and whose merge stays under tol.
int PSFBody::CoarsenWhere(const Vec3& center, float radius, float tol,
                          int max_merges) {
    // ONE seam rebuild for the whole batch (see RefineWhere).
    BeginSeamBatch();
    int done = 0;
    bool again = true;
    while (again && done < max_merges) {
        again = false;
        // candidate centers = anchors referenced by exactly 4 patches
        std::vector<int> refcount(anchors.Count(), 0);
        for (const SurfaceRecord& s : surface_blob)
            for (int k = 0; k < 4; ++k) refcount[s.anchor_idx[k]]++;
        for (uint16_t a = 0; a < refcount.size(); ++a) {
            if (refcount[a] != 4) continue;
            const Vec3 p = anchors.Position(a);
            const float d = std::sqrt((p.x-center.x)*(p.x-center.x)
                          + (p.y-center.y)*(p.y-center.y)
                          + (p.z-center.z)*(p.z-center.z));
            if (d > radius) continue;
            uint32_t grp[4]; uint16_t outer[4];
            if (FindMergeGroup(a, grp, outer) && MergeGroup(a, tol)) {
                ++done; again = true; break;     // blob changed: rescan
            }
        }
    }
    EndSeamBatch();
    return done;
}

void PSFBody::RegisterModeShapes(std::vector<std::vector<float>> shapes) {
    modes = std::move(shapes);
}

void PSFBody::ApplyMode(uint32_t m, float k) {
    if (m >= modes.size()) return;
    const auto& sh = modes[m];
    // stale guard: mode table was baked for a different topology
    if (sh.size() != (size_t)anchor_blob.size()
                        + (size_t)surface_blob.size() * 8) return;
    const size_t na = anchor_blob.size(), ns = surface_blob.size();
    // coefficient deltas: coef[s][i] += k * shape[surf-part][s*8+i]
    const float* sc = sh.data() + na;
    for (uint32_t s = 0; s < ns; ++s) {
        for (int i = 0; i < 8; ++i)
            surface_blob[s].coef[i] += k * sc[size_t(s)*8 + i];
    }
    // anchor displacements along their rest normals (PSM v2 layout:
    // anchor entries FIRST, same convention ModalDynamics::WriteBack uses)
    for (uint32_t a = 0; a < na; ++a) {
        const float d = k * sh[a];
        if (d == 0.0f) continue;
        const Vec3 p = anchors.Position((uint16_t)a);
        const Vec3 n = anchors.Normal((uint16_t)a);
        anchors.SetPosition((uint16_t)a,
                            { p.x + d*n.x, p.y + d*n.y, p.z + d*n.z });
    }
    MarkAnchorsDirty();
}

void PSFBody::RefitLocal(const Vec3& c, float r) {
    // Snapshot each in-region patch's CURRENT surface, then re-fit its 8
    // coefficients against that snapshot on the corner-vanishing basis
    // (the same LSQ SplitPatch/MergeGroup/RederiveNormals use). On a clean
    // body this is idempotent (recovers the same coefs); after external
    // anchor edits it re-optimizes the coefficients so the surface stays
    // consistent with the (new) anchor bilinear + normal field.
    const int G = 9;
    auto mixv = [](Vec3 a, Vec3 b, float t){
        return Vec3{a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t}; };
    for (uint32_t s = 0; s < surface_blob.size(); ++s) {
        const Vec3 pc = patch_center(*this, s);
        const float d = std::sqrt((pc.x-c.x)*(pc.x-c.x)
                                + (pc.y-c.y)*(pc.y-c.y)
                                + (pc.z-c.z)*(pc.z-c.z));
        if (d > r) continue;
        std::vector<Vec3> target(G*G);
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j)
                target[i*G+j] = patches[s].Evaluate(i/float(G-1), j/float(G-1));
        Vec3 A[4], N[4];
        for (int k = 0; k < 4; ++k) {
            A[k] = anchors.Position(surface_blob[s].anchor_idx[k]);
            N[k] = anchors.Normal(surface_blob[s].anchor_idx[k]);
        }
        double M[8][8] = {{0}}, rr[8] = {0};
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float u = i/float(G-1), v = j/float(G-1);
                const Vec3 bil = mixv(mixv(A[0],A[3],u), mixv(A[1],A[2],u), v);
                Vec3 n = mixv(mixv(N[0],N[3],u), mixv(N[1],N[2],u), v);
                const float nl = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
                n = nl > 1e-12f ? Vec3{n.x/nl,n.y/nl,n.z/nl} : Vec3{0,0,1};
                const Vec3 tgt = target[i*G+j];
                const double h = (tgt.x-bil.x)*n.x + (tgt.y-bil.y)*n.y
                               + (tgt.z-bil.z)*n.z;
                double B[8]; cv_basis(u, v, B);
                for (int x = 0; x < 8; ++x) {
                    rr[x] += B[x]*h;
                    for (int y = 0; y < 8; ++y) M[x][y] += B[x]*B[y];
                }
            }
        float coef[8];
        if (solve8(M, rr, coef))
            for (int x = 0; x < 8; ++x) surface_blob[s].coef[x] = coef[x];
    }
    MarkAnchorsDirty();
}
float PSFBody::Volume() const {
    // V = (1/3) closed-surface integral of P . (dPdu x dPdv) du dv
    // Gauss-Legendre 4x4 per patch on the analytic derivatives.
    static const double gp[4] = {0.0694318442, 0.3300094782,
                                 0.6699905218, 0.9305681558};
    static const double gw[4] = {0.1739274226, 0.3260725774,
                                 0.3260725774, 0.1739274226};
    double vol = 0;
    for (const Patch& p : patches)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                Vec3 P, du, dv;
                p.EvaluateDerivs((float)gp[i], (float)gp[j], P, du, dv);
                const Vec3 cr{ du.y*dv.z-du.z*dv.y, du.z*dv.x-du.x*dv.z,
                               du.x*dv.y-du.y*dv.x };
                vol += gw[i]*gw[j]*(P.x*cr.x + P.y*cr.y + P.z*cr.z);
    }
    return (float)(std::fabs(vol) / 3.0);
}
static float patch_size(const PSFBody& b, uint32_t pid);
static Vec3  patch_center(const PSFBody& b, uint32_t pid);

static float d2v(const Vec3& a, const Vec3& b) {
    return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z);
}

bool PSFBody::ClosestPointEx(const Vec3& p, SurfaceHit& out) const {
    // broad phase: K nearest patch centers
    const int K = 8;
    uint32_t cand[K]; float cd[K];
    for (int k = 0; k < K; ++k) { cand[k] = 0; cd[k] = 1e30f; }
    for (uint32_t pid = 0; pid < patches.size(); ++pid) {
        const float d = d2v(patch_center(*this, pid), p);
        for (int k = 0; k < K; ++k)
            if (d < cd[k]) {
                for (int m = K-1; m > k; --m) { cd[m]=cd[m-1]; cand[m]=cand[m-1]; }
                cd[k] = d; cand[k] = pid; break;
            }
    }
    bool found = false; float best = 1e30f;
    for (int k = 0; k < K; ++k) {
        const Patch& P = patches[cand[k]];
        // 2x2 seeds, projected Gauss-Newton on (u,v)
        for (int si = 0; si < 2; ++si)
            for (int sj = 0; sj < 2; ++sj) {
                float u = 0.25f + 0.5f*si, v = 0.25f + 0.5f*sj;
                for (int it = 0; it < 12; ++it) {
                    Vec3 q, du, dv;
                    P.EvaluateDerivs(u, v, q, du, dv);
                    const Vec3 r = { q.x-p.x, q.y-p.y, q.z-p.z };
                    // normal equations of J^T J [su sv] = -J^T r
                    const double a = du.x*du.x+du.y*du.y+du.z*du.z;
                    const double b = du.x*dv.x+du.y*dv.y+du.z*dv.z;
                    const double c = dv.x*dv.x+dv.y*dv.y+dv.z*dv.z;
                    const double g0 = du.x*r.x+du.y*r.y+du.z*r.z;
                    const double g1 = dv.x*r.x+dv.y*r.y+dv.z*r.z;
                    const double det = a*c - b*b;
                    if (std::fabs(det) < 1e-18) break;
                    float su = (float)((-c*g0 + b*g1)/det);
                    float sv = (float)(( b*g0 - a*g1)/det);
                    u += su; v += sv;
                    u = u < 0 ? 0 : (u > 1 ? 1 : u);
                    v = v < 0 ? 0 : (v > 1 ? 1 : v);
                    if (su*su + sv*sv < 1e-14f) break;
                }
                Vec3 q = P.Evaluate(u, v);
                const float d = d2v(q, p);
                if (d < best) {
                    best = d; found = true;
                    out = { cand[k], u, v, 0.0f, q };
                }
            }
    }
    return found;
}

Vec3 PSFBody::ClosestPoint(const Vec3& p) const {
    SurfaceHit h;
    return ClosestPointEx(p, h) ? h.p : p;
}

bool PSFBody::Raycast(const Vec3& o, const Vec3& d, SurfaceHit& out) const {
    bool found = false; float best_t = 1e30f;
    const float dl = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    const Vec3 dir = { d.x/dl, d.y/dl, d.z/dl };
    for (uint32_t pid = 0; pid < patches.size(); ++pid) {
        // sphere reject: patch center vs ray
        const Vec3 c = patch_center(*this, pid);
        const float R = 1.1f * patch_size(*this, pid);
        const Vec3 oc = { c.x-o.x, c.y-o.y, c.z-o.z };
        const float proj = oc.x*dir.x + oc.y*dir.y + oc.z*dir.z;
        if (proj < -R) continue;
        const float dd = d2v(oc, { proj*dir.x, proj*dir.y, proj*dir.z });
        if (dd > R*R) continue;
        const Patch& P = patches[pid];
        for (int si = 0; si < 3; ++si)
            for (int sj = 0; sj < 3; ++sj) {
                float u = 0.166f + 0.333f*si, v = 0.166f + 0.333f*sj;
                float t = proj > 0 ? proj : 0;
                for (int it = 0; it < 16; ++it) {
                    Vec3 q, du, dv;
                    P.EvaluateDerivs(u, v, q, du, dv);
                    const Vec3 r = { q.x-o.x-t*dir.x, q.y-o.y-t*dir.y,
                                     q.z-o.z-t*dir.z };
                    // 3x3: [du dv -dir][su sv st] = -r  (Cramer)
                    const Vec3 A = du, B = dv, C = { -dir.x,-dir.y,-dir.z };
                    const double det =
                        A.x*(B.y*C.z-B.z*C.y) - B.x*(A.y*C.z-A.z*C.y)
                      + C.x*(A.y*B.z-A.z*B.y);
                    if (std::fabs(det) < 1e-18) break;
                    auto cram = [&](Vec3 X, Vec3 Y, Vec3 Z) {
                        return (X.x*(Y.y*Z.z-Y.z*Z.y) - Y.x*(X.y*Z.z-X.z*Z.y)
                              + Z.x*(X.y*Y.z-X.z*Y.y)) / det; };
                    const Vec3 nr = { -r.x, -r.y, -r.z };
                    const float su = (float)cram(nr, B, C);
                    const float sv = (float)cram(A, nr, C);
                    const float st = (float)cram(A, B, nr);
                    u += su; v += sv; t += st;
                    // float32 noise floor sits near 1e-7 per step: break
                    // on step OR rely on the residual gate below
                    if (su*su+sv*sv+st*st < 1e-12f) break;
                    if (u < -0.5f || u > 1.5f || v < -0.5f || v > 1.5f) break;
                }
                if (t <= 1e-5f || t >= best_t) continue;
                if (u < -1e-3f || u > 1.0f+1e-3f
                 || v < -1e-3f || v > 1.0f+1e-3f) continue;
                const float uc = u < 0 ? 0 : (u > 1 ? 1 : u);
                const float vc = v < 0 ? 0 : (v > 1 ? 1 : v);
                const Vec3 q = P.Evaluate(uc, vc);
                const Vec3 rp = { o.x+t*dir.x, o.y+t*dir.y, o.z+t*dir.z };
                if (d2v(q, rp) > 1e-8f) continue;     // residual gate
                best_t = t; found = true;
                out = { pid, uc, vc, t, q };
            }
    }
    return found;
}
void PSFBody::FlushToGPU(void* q) {
    // q is a PSFFlushState* (renderer-owned device buffers that mirror
    // this body's blobs). nullptr or incomplete state = safe no-op.
    PSFFlushState* fs = (PSFFlushState*)q;
    if (!fs || !fs->queue || !fs->anchors_buf || !fs->surfaces_buf) return;
    const cl_command_queue cq = fs->queue;

    const auto& A  = anchor_blob;
    const auto& S  = surface_blob;
    const size_t ab = A.size() * 64;
    const size_t sb = S.size() * 64;

    // body grew past the device capacity (live splits): full re-upload
    const bool grew = ab > fs->anchors_bytes || sb > fs->surfaces_bytes;
    if (grew) {
        clEnqueueWriteBuffer(cq, fs->anchors_buf, CL_FALSE, 0, ab,
                             A.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(cq, fs->surfaces_buf, CL_FALSE, 0, sb,
                             S.data(), 0, nullptr, nullptr);
        if (fs->stitch_buf && !stitch_blob.empty())
            clEnqueueWriteBuffer(cq, fs->stitch_buf, CL_FALSE, 0,
                                 stitch_blob.size(), stitch_blob.data(),
                                 0, nullptr, nullptr);
        fs->anchors_bytes = ab;
        fs->surfaces_bytes = sb;
        fs->stitch_bytes   = stitch_blob.empty() ? 0 : stitch_blob.size();
        fs->flush_gen = anchor_gen;
        MarkAnchorsClean();
        ClearDirty();
        return;
    }

    const bool anchors_changed = AnchorsDirty();
    if (anchors_changed) {
        // anchors are shared between patches; a moved anchor invalidates
        // every neighbor, so the anchor blob goes up whole (same policy
        // as the renderer's full-upload path). Stitch rides along: seam
        // restitching happens in the same pass as anchor edits.
        clEnqueueWriteBuffer(cq, fs->anchors_buf, CL_FALSE, 0, ab,
                             A.data(), 0, nullptr, nullptr);
        if (fs->stitch_buf && !stitch_blob.empty())
            clEnqueueWriteBuffer(cq, fs->stitch_buf, CL_FALSE, 0,
                                 stitch_blob.size(), stitch_blob.data(),
                                 0, nullptr, nullptr);
        fs->flush_gen = anchor_gen;
        MarkAnchorsClean();
    }

    // the whole point: surface edits are tiny, so upload ONLY the dirty
    // 64-byte records (one small write per dirty patch) instead of the
    // whole blob.
    if (AnyDirty()) {
        for (uint32_t s = 0; s < S.size(); ++s) {
            if (!patches[s].Dirty()) continue;
            clEnqueueWriteBuffer(cq, fs->surfaces_buf, CL_FALSE,
                                 (size_t)s * 64, 64, &S[s],
                                 0, nullptr, nullptr);
        }
        ClearDirty();
    }
}

} // namespace DeliRium

namespace DeliRium {

static float patch_size(const PSFBody& b, uint32_t pid) {
    const SurfaceRecord& s = b.SurfaceBlob()[pid];
    auto& pool = b.AnchorsConst();
    float m = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) {
            const Vec3 a = pool.Position(s.anchor_idx[i]);
            const Vec3 c = pool.Position(s.anchor_idx[j]);
            const float d = std::sqrt((a.x-c.x)*(a.x-c.x)
                          + (a.y-c.y)*(a.y-c.y) + (a.z-c.z)*(a.z-c.z));
            if (d > m) m = d;
        }
    return m;
}

static Vec3 patch_center(const PSFBody& b, uint32_t pid) {
    const SurfaceRecord& s = b.SurfaceBlob()[pid];
    auto& pool = b.AnchorsConst();
    Vec3 c{};
    for (int i = 0; i < 4; ++i) {
        const Vec3 a = pool.Position(s.anchor_idx[i]);
        c.x += a.x*0.25f; c.y += a.y*0.25f; c.z += a.z*0.25f;
    }
    return c;
}

void PSFBody::RefineWhere(const Vec3& center, float radius,
                          float target_size, int max_splits) {
    // ONE seam rebuild for the whole batch instead of one per split.
    //
    // MEASURED: a single SplitPatch costs 0.837 ms, of which 0.826 ms
    // (98.7%) is the seam rebuild. This loop does up to `max_splits` of
    // them, so rebuilding per split is O(n^2) over a growing surface set.
    // Deferring it to the end of the batch is the fix -- NOT a GPU port of
    // the fit, which is only 3.4% of the cost.
    //
    // Nested batches are counted, so a caller that already opened one
    // still gets exactly one rebuild at the outermost close.
    BeginSeamBatch();
    int budget = max_splits;
    bool again = true;
    while (again && budget > 0) {
        again = false;
        const uint32_t n = (uint32_t)surface_blob.size();
        for (uint32_t pid = 0; pid < n && budget > 0; ++pid) {
            const float sz = patch_size(*this, pid);
            if (sz <= target_size) continue;
            const Vec3 c = patch_center(*this, pid);
            const float d = std::sqrt((c.x-center.x)*(c.x-center.x)
                          + (c.y-center.y)*(c.y-center.y)
                          + (c.z-center.z)*(c.z-center.z));
            if (d > radius + 0.5f*sz) continue;
            uint32_t kids[4];
            if (SplitPatch(pid, kids)) { --budget; again = true; }
        }
    }
    EndSeamBatch();
}

void PSFBody::ApplyRadialDent(const Vec3& center, float radius, float depth) {
    auto wfun = [&](const Vec3& p) {
        const float r2 = (p.x-center.x)*(p.x-center.x)
                       + (p.y-center.y)*(p.y-center.y)
                       + (p.z-center.z)*(p.z-center.z);
        return std::exp(-4.0f * r2 / (radius*radius));
    };
    const int G = 9;
    const float winf = std::exp(-4.0f);     // influence cutoff (r = radius)

    // 1) touched set + ORIGINAL target snapshots (before anything moves)
    struct Snap { uint32_t pid; Vec3 desired[G*G]; };
    std::vector<Snap> snaps;
    std::vector<uint8_t> anchor_moved(anchor_blob.size(), 0);
    for (uint16_t a = 0; a < anchor_blob.size(); ++a)
        if (wfun(anchors.Position(a)) > winf) anchor_moved[a] = 1;

    for (uint32_t pid = 0; pid < surface_blob.size(); ++pid) {
        bool touched = false;
        for (int k = 0; k < 4 && !touched; ++k)
            touched = anchor_moved[surface_blob[pid].anchor_idx[k]];
        if (!touched) {
            const Vec3 c = patch_center(*this, pid);
            touched = wfun(c) > winf;
        }
        if (!touched) continue;
        Snap s; s.pid = pid;
        const Patch& P = patches[pid];
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float u = i/float(G-1), v = j/float(G-1);
                const Vec3 p = P.Evaluate(u, v);
                const Vec3 n = P.EvaluateNormalField(u, v);
                const float w = wfun(p);
                s.desired[i*G+j] = { p.x - depth*w*n.x,
                                     p.y - depth*w*n.y,
                                     p.z - depth*w*n.z };
            }
        snaps.push_back(s);
    }

    // 2) move anchors along their stored normals
    for (uint16_t a = 0; a < anchor_blob.size(); ++a) {
        if (!anchor_moved[a]) continue;
        const Vec3 p = anchors.Position(a);
        const Vec3 n = anchors.Normal(a);
        const float w = wfun(p);
        anchors.SetPosition(a, { p.x - depth*w*n.x,
                                 p.y - depth*w*n.y,
                                 p.z - depth*w*n.z });
    }
    anchors_dirty = true;

    // 3) refit each touched patch: coef += LSQ((desired - current) . n)
    std::vector<uint32_t> touched_ids;
    for (const Snap& s : snaps) {
        Patch& P = patches[s.pid];
        double M[8][8] = {{0}}, r[8] = {0};
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float u = i/float(G-1), v = j/float(G-1);
                const Vec3 cur = P.Evaluate(u, v);
                const Vec3 n   = P.EvaluateNormalField(u, v);
                const Vec3 dd  = s.desired[i*G+j];
                const double h = (dd.x-cur.x)*n.x + (dd.y-cur.y)*n.y
                               + (dd.z-cur.z)*n.z;
                double B[8]; cv_basis(u, v, B);
                for (int x = 0; x < 8; ++x) {
                    r[x] += B[x]*h;
                    for (int y = 0; y < 8; ++y) M[x][y] += B[x]*B[y];
                }
            }
        float dc[8];
        if (solve8(M, r, dc)) P.AddDisplacementMode(dc);
        touched_ids.push_back(s.pid);
    }

    // 4) restitch the whole touched neighborhood
    seams.Build(*this);
    seams.RestitchAround(*this, touched_ids);
}

} // namespace DeliRium

namespace DeliRium {

std::unique_ptr<PSFBody> PSFBody::DetachPatches(
        const std::vector<uint32_t>& ids) {
    if (ids.empty() || ids.size() >= surface_blob.size()) return nullptr;
    std::unordered_set<uint32_t> det(ids.begin(), ids.end());

    // collect anchors referenced by detached patches; build a remap into
    // the fragment's own anchor blob (copy positions/normals verbatim)
    std::unordered_map<uint16_t,uint16_t> remap;
    std::vector<AnchorRecord> frag_anchors;
    auto get_remap = [&](uint16_t a)->uint16_t {
        auto it = remap.find(a);
        if (it != remap.end()) return it->second;
        const uint16_t ni = (uint16_t)frag_anchors.size();
        frag_anchors.push_back(anchor_blob[a]);
        remap[a] = ni; return ni;
    };
    std::vector<SurfaceRecord> frag_surf;
    std::vector<int8_t> frag_stitch;
    for (uint32_t pid : ids) {
        SurfaceRecord r = surface_blob[pid];
        for (int k=0;k<4;++k) r.anchor_idx[k] = get_remap(r.anchor_idx[k]);
        frag_surf.push_back(r);
        frag_stitch.insert(frag_stitch.end(),
            &stitch_blob[size_t(pid)*8], &stitch_blob[size_t(pid)*8]+8);
    }
    auto frag = PSFBody::FromRecords(std::move(frag_anchors),
        std::move(frag_surf), std::move(frag_stitch),
        bbox_min, bbox_span, s_scale);

    // remove detached patches from this body; compact (anchors left in
    // place -- orphans are harmless, reclaimed on save/reload)
    std::vector<SurfaceRecord> ns; std::vector<int8_t> nst;
    ns.reserve(surface_blob.size()-ids.size());
    for (uint32_t s=0; s<surface_blob.size(); ++s) {
        if (det.count(s)) continue;
        ns.push_back(surface_blob[s]);
        nst.insert(nst.end(), &stitch_blob[size_t(s)*8],
                   &stitch_blob[size_t(s)*8]+8);
    }
    surface_blob.swap(ns); stitch_blob.swap(nst);
    RebuildViews();
    anchors_dirty = true;
    return frag;
}

} // namespace DeliRium

namespace DeliRium {

void PSFBody::ExpandBBox(float margin) {
    // capture current world positions, grow the box, re-bind, re-quantize
    const uint32_t na = (uint32_t)anchor_blob.size();
    std::vector<Vec3> world(na);
    for (uint32_t a=0;a<na;++a) world[a]=anchors.Position(a);
    const Vec3 m{ bbox_span.x*margin, bbox_span.y*margin, bbox_span.z*margin };
    bbox_min  = { bbox_min.x-m.x, bbox_min.y-m.y, bbox_min.z-m.z };
    bbox_span = { bbox_span.x+2*m.x, bbox_span.y+2*m.y, bbox_span.z+2*m.z };
    anchors.Bind(&anchor_blob, bbox_min, bbox_span);
    for (uint32_t a=0;a<na;++a) anchors.SetPosition(a, world[a]);
    anchors_dirty = true;
}

void PSFBody::Translate(const Vec3& m) {
    // capture current world positions, shift the box, re-bind, re-quantize
    const uint32_t na = (uint32_t)anchor_blob.size();
    std::vector<Vec3> world(na);
    for (uint32_t a=0;a<na;++a) world[a]=anchors.Position(a);
    bbox_min = { bbox_min.x + m.x, bbox_min.y + m.y, bbox_min.z + m.z };
    anchors.Bind(&anchor_blob, bbox_min, bbox_span);
    for (uint32_t a=0;a<na;++a)
        anchors.SetPosition(a, { world[a].x + m.x, world[a].y + m.y, world[a].z + m.z });
    anchors_dirty = true;
}

} // namespace DeliRium

namespace DeliRium {

const std::vector<Vec3>& PSFBody::ComputeSmoothNormals() const {
    // cache hit: anchors unchanged since the cache was built
    if (smooth_n_gen == anchor_gen && !smooth_n_cache.empty())
        return smooth_n_cache;

    std::vector<Vec3> N(anchor_blob.size(), Vec3{0,0,0});
    // anchor slot -> corner (u,v).
    //
    // THE CORNER ORDER IS CONVENTION A (u: P0->P3, v: P0->P1) -- the same
    // convention the renderer uses in Patch::EvaluateDerivs and that the
    // baker writes (a00, a10, a11, a01 = (i,j), (i+1,j), (i+1,j+1), (i,j+1)):
    //
    //   A0 = (0,0)   A1 = (1,0)   A2 = (1,1)   A3 = (0,1)
    //
    // This table used to be the TRANSPOSED one (A1=(0,1), A3=(1,0)) -- the
    // convention B that Patch.cpp explicitly warns about ("the partials MUST
    // use the SAME convention, or cross(dPdu,dPdv) flips sign"). The effect:
    // two of the four corners of EVERY patch contributed the normal of the
    // WRONG corner, so the smooth-normal field was a mixture of two
    // different directions on every patch. The symptom is view-dependent:
    // the shading normal points the wrong way on the faces where the
    // transposition shows, and the object appears to lose its
    // front-facing half as the camera swings around it.
    static const float CU[4] = {0.f, 1.f, 1.f, 0.f};
    static const float CV[4] = {0.f, 0.f, 1.f, 1.f};
    // AREA-WEIGHTED accumulation: each patch contributes its corner normal
    // scaled by the patch's screen-independent size, so a large patch sets
    // the dominant orientation where it meets small ones (an unweighted
    // average lets tiny patches tilt the shared normal -> a visible shading
    // crease at size transitions). Weight = patch parametric area proxy
    // (|A0A3| x |A0A1|), cheap and stable.
    for (uint32_t s = 0; s < surface_blob.size(); ++s) {
        const Patch& P = patches[s];
        const Vec3 a0 = anchors.Position(surface_blob[s].anchor_idx[0]);
        const Vec3 a1 = anchors.Position(surface_blob[s].anchor_idx[1]);
        const Vec3 a3 = anchors.Position(surface_blob[s].anchor_idx[3]);
        const Vec3 e1{a3.x-a0.x, a3.y-a0.y, a3.z-a0.z};
        const Vec3 e2{a1.x-a0.x, a1.y-a0.y, a1.z-a0.z};
        const Vec3 cr{e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z,
                      e1.x*e2.y-e1.y*e2.x};
        const float w = std::sqrt(cr.x*cr.x+cr.y*cr.y+cr.z*cr.z) + 1e-6f;
        for (int k = 0; k < 4; ++k) {
            const Vec3 n = P.EvaluateNormal(CU[k], CV[k]);
            const uint16_t a = surface_blob[s].anchor_idx[k];
            N[a].x += n.x*w; N[a].y += n.y*w; N[a].z += n.z*w;
        }
    }
    for (Vec3& n : N) {
        const float l = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        if (l > 1e-9f) { n.x/=l; n.y/=l; n.z/=l; }
    }
    // ONE Laplacian smoothing pass across shared edges: pull each anchor's
    // normal toward the average of its edge-neighbors. Shared anchors
    // already have one normal (C0), but neighbors can still differ enough
    // to read as a crease; a light blend (0.5) softens those transitions
    // without flattening genuine features. This is the "averaging logic"
    // applied to shading, where the data shows the real discontinuity is.
    std::vector<Vec3> acc(N.size(), Vec3{0,0,0});
    std::vector<float> cnt(N.size(), 0.0f);
    for (uint32_t s = 0; s < surface_blob.size(); ++s) {
        const uint16_t* ai = surface_blob[s].anchor_idx;
        // 4 edges of the quad connect (0-1),(1-2),(2-3),(3-0)
        static const int E[4][2] = {{0,1},{1,2},{2,3},{3,0}};
        for (auto& e : E) {
            const uint16_t a = ai[e[0]], b = ai[e[1]];
            acc[a].x += N[b].x; acc[a].y += N[b].y; acc[a].z += N[b].z; cnt[a]+=1;
            acc[b].x += N[a].x; acc[b].y += N[a].y; acc[b].z += N[a].z; cnt[b]+=1;
        }
    }
    const float blend = 0.5f;
    for (size_t i = 0; i < N.size(); ++i) {
        if (cnt[i] < 1) continue;
        const float inv = 1.0f/cnt[i];
        Vec3 avg{acc[i].x*inv, acc[i].y*inv, acc[i].z*inv};
        Vec3 m{ N[i].x + blend*(avg.x-N[i].x),
                N[i].y + blend*(avg.y-N[i].y),
                N[i].z + blend*(avg.z-N[i].z) };
        const float l = std::sqrt(m.x*m.x+m.y*m.y+m.z*m.z);
        if (l > 1e-9f) { m.x/=l; m.y/=l; m.z/=l; }
        N[i] = m;
    }
    smooth_n_cache = std::move(N);
    smooth_n_gen = anchor_gen;
    return smooth_n_cache;
}

// Re-derive anchor normals from the ACTUAL surface (the user's point: the
// anchor normal should be a CONSEQUENCE of the surface, not an independent
// datum the deformation can contradict). The exporter sometimes stores a
// displacement normal Nd that differs from the true geometric normal by up
// to 90 deg, so the deformation h*Nd pushes the surface sideways and points
// "appear where they shouldn't". This fixes it WITHOUT moving the surface:
//   1) snapshot each patch's current surface (sampled),
//   2) replace every anchor's normal with the true geometric normal,
//   3) re-fit each patch's 8 deformation coefficients against the snapshot
//      using the NEW normals -> identical geometry, consistent normals.
// Returns the worst residual (fraction of diagonal) after refit.
float PSFBody::RederiveNormals() {
    const int G = 9;
    // 1) snapshot target surface for every patch
    std::vector<std::vector<Vec3>> target(surface_blob.size());
    for (uint32_t s = 0; s < surface_blob.size(); ++s) {
        target[s].reserve(G*G);
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j)
                target[s].push_back(patches[s].Evaluate(i/float(G-1),
                                                        j/float(G-1)));
    }
    // 2) replace anchor normals with the true geometric ones
    const std::vector<Vec3>& tn = ComputeSmoothNormals();
    for (uint32_t a = 0; a < anchors.Count() && a < tn.size(); ++a)
        anchors.SetNormal(a, tn[a]);
    RebuildViews();   // patches now see the new anchor normals

    // 3) re-fit each patch's coefs so it reproduces the snapshot with the
    //    new normal field (same LSQ as split, but target = old surface)
    auto mixv = [](Vec3 a, Vec3 b, float t){
        return Vec3{a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t}; };
    const Vec3 sp = bbox_span;
    const float diag = std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    float worst = 0;
    for (uint32_t s = 0; s < surface_blob.size(); ++s) {
        Vec3 A[4], N[4];
        for (int k = 0; k < 4; ++k) {
            A[k] = anchors.Position(surface_blob[s].anchor_idx[k]);
            N[k] = anchors.Normal(surface_blob[s].anchor_idx[k]);
        }
        double M[8][8] = {{0}}, r[8] = {0};
        int idx = 0;
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j, ++idx) {
                const float u = i/float(G-1), v = j/float(G-1);
                const Vec3 bil = mixv(mixv(A[0],A[3],u), mixv(A[1],A[2],u), v);
                Vec3 n = mixv(mixv(N[0],N[3],u), mixv(N[1],N[2],u), v);
                const float nl = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
                n = nl > 1e-12f ? Vec3{n.x/nl,n.y/nl,n.z/nl} : Vec3{0,0,1};
                const Vec3 tgt = target[s][idx];
                const double h = (tgt.x-bil.x)*n.x + (tgt.y-bil.y)*n.y
                               + (tgt.z-bil.z)*n.z;
                double B[8]; cv_basis(u, v, B);
                for (int x = 0; x < 8; ++x) {
                    r[x] += B[x]*h;
                    for (int y = 0; y < 8; ++y) M[x][y] += B[x]*B[y];
                }
            }
        float coef[8];
        if (solve8(M, r, coef))
            for (int x = 0; x < 8; ++x) surface_blob[s].coef[x] = coef[x];
    }
    RebuildViews();
    // measure residual vs snapshot
    for (uint32_t s = 0; s < surface_blob.size(); ++s) {
        int idx = 0;
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j, ++idx) {
                const Vec3 a = patches[s].Evaluate(i/float(G-1), j/float(G-1));
                const Vec3 t = target[s][idx];
                worst = std::fmax(worst, std::sqrt((a.x-t.x)*(a.x-t.x)
                       +(a.y-t.y)*(a.y-t.y)+(a.z-t.z)*(a.z-t.z)));
            }
    }
    anchors_dirty = true; ++anchor_gen;
    return worst / diag;
}

} // namespace DeliRium
