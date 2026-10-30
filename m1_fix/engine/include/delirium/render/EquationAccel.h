// Acceleration for PSF equation rendering: patch AABB, frustum, BVH, adaptive uvn.
// Visible set is conical (view frustum / tile cones) — never a fixed tri budget.
#pragma once
#include <cstdint>
#include <vector>
#include "../core/Types.h"

namespace DeliRium {

class PSFBody;
struct Camera;

struct PatchAABB {
    Vec3 bmin{}, bmax{};
    uint32_t surface_index = 0;
    float deform_margin = 0.f;
    float projected_px = 0.f;  // filled during cull (approx screen extent)
};

struct Frustum {
    // ax+by+cz+d >= 0 is inside for each plane
    float plane[6][4]{};
};

struct BVHNode {
    Vec3 bmin{}, bmax{};
    int left = -1, right = -1; // children indices into nodes; -1 = leaf
    int start = 0, count = 0;  // leaf: range into leaf_indices
};

struct EquationAccel {
    std::vector<PatchAABB> patches;
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> leaf_indices; // surface indices
    std::vector<uint32_t> visible;      // output of conical search
    std::vector<uint8_t> uvn_per_vis;   // adaptive uvn aligned with visible[]

    // Rebuild from body blobs (call when anchors dirty / load).
    void Build(const PSFBody& body);

    // Extract 6 frustum planes from row-major MVP = P*V.
    static Frustum FrustumFromMVP(const float mvp[16]);

    // Conical search: frustum ∩ BVH → visible + adaptive uvn from projected size.
    void QueryFrustum(const Frustum& fr, const float mvp[16],
                      int width, int height,
                      const Vec3& eye,
                      int uvn_min = 2, int uvn_max = 8);
};

bool AABBOutsideFrustum(const Vec3& bmin, const Vec3& bmax, const Frustum& fr);
bool AABBIntersectsFrustum(const Vec3& bmin, const Vec3& bmax, const Frustum& fr);

} // namespace DeliRium
