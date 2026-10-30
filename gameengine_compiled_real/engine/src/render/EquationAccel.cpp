#include "delirium/render/EquationAccel.h"
#include "delirium/psf/PSFBody.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace DeliRium {
namespace {

inline Vec3 dequant(const AnchorRecord& a, const Vec3& bbmin, const Vec3& bbspan) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&a);
    uint16_t px, py, pz;
    std::memcpy(&px, b + 32, 2);
    std::memcpy(&py, b + 34, 2);
    std::memcpy(&pz, b + 36, 2);
    return Vec3{
        bbmin.x + (px / 65535.f) * bbspan.x,
        bbmin.y + (py / 65535.f) * bbspan.y,
        bbmin.z + (pz / 65535.f) * bbspan.z};
}

inline float max3(float a, float b, float c) {
    return std::max(a, std::max(b, c));
}

inline void expand(Vec3& bmin, Vec3& bmax, const Vec3& p) {
    bmin.x = std::min(bmin.x, p.x); bmin.y = std::min(bmin.y, p.y); bmin.z = std::min(bmin.z, p.z);
    bmax.x = std::max(bmax.x, p.x); bmax.y = std::max(bmax.y, p.y); bmax.z = std::max(bmax.z, p.z);
}

inline float plane_dist(const float pl[4], float x, float y, float z) {
    return pl[0] * x + pl[1] * y + pl[2] * z + pl[3];
}

// Positive vertex of AABB along plane normal (for outside test).
inline float aabb_max_dist(const float pl[4], const Vec3& bmin, const Vec3& bmax) {
    float x = pl[0] >= 0.f ? bmax.x : bmin.x;
    float y = pl[1] >= 0.f ? bmax.y : bmin.y;
    float z = pl[2] >= 0.f ? bmax.z : bmin.z;
    return plane_dist(pl, x, y, z);
}

inline float aabb_min_dist(const float pl[4], const Vec3& bmin, const Vec3& bmax) {
    float x = pl[0] >= 0.f ? bmin.x : bmax.x;
    float y = pl[1] >= 0.f ? bmin.y : bmax.y;
    float z = pl[2] >= 0.f ? bmin.z : bmax.z;
    return plane_dist(pl, x, y, z);
}

int build_bvh_rec(std::vector<BVHNode>& nodes, std::vector<uint32_t>& leaf_idx,
                  std::vector<PatchAABB>& patches, int begin, int end, int depth) {
    BVHNode node{};
    node.bmin = {1e30f, 1e30f, 1e30f};
    node.bmax = {-1e30f, -1e30f, -1e30f};
    for (int i = begin; i < end; ++i) {
        expand(node.bmin, node.bmax, patches[leaf_idx[i]].bmin);
        expand(node.bmin, node.bmax, patches[leaf_idx[i]].bmax);
    }
    const int count = end - begin;
    if (count <= 4 || depth > 20) {
        node.left = node.right = -1;
        node.start = begin;
        node.count = count;
        nodes.push_back(node);
        return (int)nodes.size() - 1;
    }
    // Split longest axis at midpoint of centroids
    Vec3 ext{node.bmax.x - node.bmin.x, node.bmax.y - node.bmin.y, node.bmax.z - node.bmin.z};
    int axis = 0;
    if (ext.y > ext.x && ext.y >= ext.z) axis = 1;
    else if (ext.z > ext.x && ext.z >= ext.y) axis = 2;
    auto centroid = [&](uint32_t li) {
        const auto& p = patches[li];
        if (axis == 0) return 0.5f * (p.bmin.x + p.bmax.x);
        if (axis == 1) return 0.5f * (p.bmin.y + p.bmax.y);
        return 0.5f * (p.bmin.z + p.bmax.z);
    };
    int mid = begin + count / 2;
    std::nth_element(leaf_idx.begin() + begin, leaf_idx.begin() + mid, leaf_idx.begin() + end,
                     [&](uint32_t a, uint32_t b) { return centroid(a) < centroid(b); });
    int self = (int)nodes.size();
    nodes.push_back(node); // placeholder
    int L = build_bvh_rec(nodes, leaf_idx, patches, begin, mid, depth + 1);
    int R = build_bvh_rec(nodes, leaf_idx, patches, mid, end, depth + 1);
    nodes[self].left = L;
    nodes[self].right = R;
    nodes[self].start = 0;
    nodes[self].count = 0;
    nodes[self].bmin = nodes[L].bmin;
    nodes[self].bmax = nodes[L].bmax;
    expand(nodes[self].bmin, nodes[self].bmax, nodes[R].bmin);
    expand(nodes[self].bmin, nodes[self].bmax, nodes[R].bmax);
    return self;
}

float projected_extent_px(const Vec3& bmin, const Vec3& bmax, const float mvp[16],
                          int width, int height) {
    Vec3 corners[8] = {
        {bmin.x,bmin.y,bmin.z},{bmax.x,bmin.y,bmin.z},{bmin.x,bmax.y,bmin.z},{bmax.x,bmax.y,bmin.z},
        {bmin.x,bmin.y,bmax.z},{bmax.x,bmin.y,bmax.z},{bmin.x,bmax.y,bmax.z},{bmax.x,bmax.y,bmax.z}};
    float minx=1e30f,maxx=-1e30f,miny=1e30f,maxy=-1e30f;
    int any = 0;
    for (int i = 0; i < 8; ++i) {
        float x = corners[i].x, y = corners[i].y, z = corners[i].z;
        float cx = mvp[0]*x+mvp[1]*y+mvp[2]*z+mvp[3];
        float cy = mvp[4]*x+mvp[5]*y+mvp[6]*z+mvp[7];
        float cw = mvp[12]*x+mvp[13]*y+mvp[14]*z+mvp[15];
        if (cw < 1e-5f) continue;
        float sx = (cx/cw*0.5f+0.5f)*float(width);
        float sy = (0.5f-cy/cw*0.5f)*float(height);
        minx=std::min(minx,sx); maxx=std::max(maxx,sx);
        miny=std::min(miny,sy); maxy=std::max(maxy,sy);
        any = 1;
    }
    if (!any) return 0.f;
    return std::max(maxx-minx, maxy-miny);
}

} // namespace

bool AABBOutsideFrustum(const Vec3& bmin, const Vec3& bmax, const Frustum& fr) {
    for (int i = 0; i < 6; ++i) {
        // if the most positive corner is still outside, AABB is outside
        if (aabb_max_dist(fr.plane[i], bmin, bmax) < 0.f) return true;
    }
    return false;
}

bool AABBIntersectsFrustum(const Vec3& bmin, const Vec3& bmax, const Frustum& fr) {
    return !AABBOutsideFrustum(bmin, bmax, fr);
}

Frustum EquationAccel::FrustumFromMVP(const float mvp[16]) {
    Frustum fr{};
    const float r0[4] = {mvp[0], mvp[1], mvp[2], mvp[3]};
    const float r1[4] = {mvp[4], mvp[5], mvp[6], mvp[7]};
    const float r2[4] = {mvp[8], mvp[9], mvp[10], mvp[11]};
    const float r3[4] = {mvp[12], mvp[13], mvp[14], mvp[15]};
    auto set = [&](int i, float a, float b, float c, float d) {
        float L = std::sqrt(a * a + b * b + c * c);
        if (L < 1e-12f) L = 1.f;
        fr.plane[i][0] = a / L;
        fr.plane[i][1] = b / L;
        fr.plane[i][2] = c / L;
        fr.plane[i][3] = d / L;
    };
    set(0, r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]); // left
    set(1, r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]); // right
    set(2, r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]); // bottom
    set(3, r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]); // top
    set(4, r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]); // near
    set(5, r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]); // far
    return fr;
}

void EquationAccel::Build(const PSFBody& body) {
    patches.clear();
    nodes.clear();
    leaf_indices.clear();
    const auto& A = body.AnchorBlob();
    const auto& S = body.SurfaceBlob();
    Vec3 bbmin = body.BBoxMin(), bbspan = body.BBoxSpan();
    if (bbspan.x < 1e-8f && bbspan.y < 1e-8f && bbspan.z < 1e-8f) {
        bbmin = {-1.f, -1.f, -1.f};
        bbspan = {2.f, 2.f, 2.f};
    }
    patches.reserve(S.size());
    for (uint32_t si = 0; si < (uint32_t)S.size(); ++si) {
        const auto& s = S[si];
        PatchAABB pa{};
        pa.surface_index = si;
        pa.bmin = {1e30f, 1e30f, 1e30f};
        pa.bmax = {-1e30f, -1e30f, -1e30f};
        bool ok = true;
        for (int k = 0; k < 4; ++k) {
            if (s.anchor_idx[k] >= A.size()) { ok = false; break; }
            expand(pa.bmin, pa.bmax, dequant(A[s.anchor_idx[k]], bbmin, bbspan));
        }
        if (!ok) continue;
        float energy = 0.f;
        for (int k = 0; k < 8; ++k) energy = std::max(energy, std::fabs(s.coef[k]));
        // deform margin: coef scale * diagonal of patch
        float dx = pa.bmax.x-pa.bmin.x, dy = pa.bmax.y-pa.bmin.y, dz = pa.bmax.z-pa.bmin.z;
        float diag = std::sqrt(dx*dx+dy*dy+dz*dz);
        pa.deform_margin = energy * diag * 0.35f + 1e-3f;
        pa.bmin.x -= pa.deform_margin; pa.bmin.y -= pa.deform_margin; pa.bmin.z -= pa.deform_margin;
        pa.bmax.x += pa.deform_margin; pa.bmax.y += pa.deform_margin; pa.bmax.z += pa.deform_margin;
        patches.push_back(pa);
    }
    leaf_indices.resize(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) leaf_indices[i] = (uint32_t)i; // index into patches[]
    if (!leaf_indices.empty())
        build_bvh_rec(nodes, leaf_indices, patches, 0, (int)leaf_indices.size(), 0);
    std::printf("EquationAccel: patches=%zu bvh_nodes=%zu\n", patches.size(), nodes.size());
}

void EquationAccel::QueryFrustum(const Frustum& fr, const float mvp[16],
                                 int width, int height, const Vec3& eye,
                                 int uvn_min, int uvn_max) {
    visible.clear();
    uvn_per_vis.clear();
    if (nodes.empty() || patches.empty()) {
        // fallback: all patches
        for (size_t i = 0; i < patches.size(); ++i) {
            visible.push_back(patches[i].surface_index);
            uvn_per_vis.push_back((uint8_t)uvn_min);
        }
        return;
    }
    std::vector<int> stack;
    stack.push_back(0);
    while (!stack.empty()) {
        int ni = stack.back();
        stack.pop_back();
        if (ni < 0 || ni >= (int)nodes.size()) continue;
        const BVHNode& n = nodes[ni];
        if (AABBOutsideFrustum(n.bmin, n.bmax, fr)) continue;
        if (n.left < 0 && n.right < 0) {
            for (int i = 0; i < n.count; ++i) {
                uint32_t pi = leaf_indices[n.start + i];
                const PatchAABB& pa = patches[pi];
                if (AABBOutsideFrustum(pa.bmin, pa.bmax, fr)) continue;
                float px = projected_extent_px(pa.bmin, pa.bmax, mvp, width, height);
                // Center of patch AABB → distance to camera (conical depth LOD)
                Vec3 ctr{
                    0.5f * (pa.bmin.x + pa.bmax.x),
                    0.5f * (pa.bmin.y + pa.bmax.y),
                    0.5f * (pa.bmin.z + pa.bmax.z)};
                float dx = ctr.x - eye.x, dy = ctr.y - eye.y, dz = ctr.z - eye.z;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                // Near patches (small dist) get more equation probes; far get uvn_min.
                // Screen size still matters so tiny near-slivers don't explode.
                float near_w = 1.f / (0.35f + dist);       // stronger toward camera
                float screen_w = std::min(px * 0.08f, float(uvn_max));
                int uvn = uvn_min + (int)(near_w * screen_w * 2.5f);
                if (uvn > uvn_max) uvn = uvn_max;
                if (px < 1.2f) uvn = uvn_min; // subpixel: keep cheap
                if (dist > 8.f) uvn = uvn_min;
                visible.push_back(pa.surface_index);
                uvn_per_vis.push_back((uint8_t)uvn);
            }
        } else {
            if (n.left >= 0) stack.push_back(n.left);
            if (n.right >= 0) stack.push_back(n.right);
        }
    }
}

} // namespace DeliRium
