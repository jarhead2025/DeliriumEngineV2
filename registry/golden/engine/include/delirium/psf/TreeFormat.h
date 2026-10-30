// =============================================================================
// TreeFormat.h - the .dlt (DeliRium Tree) format: a model as PATCHES, each a
// tree of summation-of-types layers.
//
// A model = many patches. Each patch is the tree:
//   shape(u,v) = sum_i K_i * SurfaceType_i(u,v)   (K_i is a 3D vector)
//   detail(u,v)= sum_j Kd_j * DetailType_j(u,v)   (on the residual)
//   color(u,v) = sum_k Kc_k * ColorType_k(u,v)    (cosine field)
// evaluated to a dense point cloud -> pixels. Shared edges are harmonized so
// patches amalgamate with no gaps.
//
// Format (little-endian):
//   magic "DLT\0"      4 B
//   patch_count u32    4 B
//   shape_types u8     (max surface types per patch, e.g. 8)
//   detail_types u8    (max detail types per patch)
//   color_order u8     (cosine color order)
//   pad u8
//   --- per patch ---
//   shape:  shape_types * (u8 type_idx + 3*f32 Kx,Ky,Kz)
//   detail: detail_types * (u8 type_idx + 3*f32)
//   color:  3*(color_order+1)^2 f32   (channel-major cosine coeffs)
//   edge_ring: 4 * u32 neighbor patch id (for amalgam; 0xFFFFFFFF = none)
// =============================================================================
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "../core/Types.h"

namespace DeliRium {

struct TreePatch {
    // shape layer: sparse sum of surface types (idx + 3D coefficient)
    std::vector<uint8_t> shape_idx;     // which SurfaceType
    std::vector<Vec3>    shape_k;       // its 3D coefficient
    // detail layer: sparse sum of detail types
    std::vector<uint8_t> detail_idx;
    std::vector<Vec3>    detail_k;
    // color layer: dense cosine field, channel-major [3][(L+1)^2]
    int                  color_order = 0;
    std::vector<float>   color_coeff;
    // amalgam: neighbor patch ids per edge (0xFFFFFFFF = boundary)
    uint32_t             neighbor[4] = {0xFFFFFFFFu,0xFFFFFFFFu,
                                        0xFFFFFFFFu,0xFFFFFFFFu};
};

class TreeModel {
public:
    bool Load(const std::string& path);
    bool Save(const std::string& path) const;

    size_t PatchCount() const { return patches_.size(); }
    const TreePatch& Patch(size_t i) const { return patches_[i]; }
    TreePatch&       Patch(size_t i)       { return patches_[i]; }
    void AddPatch(const TreePatch& p) { patches_.push_back(p); }

    // Evaluate one patch's tree at (u,v): position, normal, color.
    // lod_shape / lod_detail / lod_color choose how many terms to sum
    // (continuous LOD). Returns the leaf point.
    struct Point { Vec3 pos; Vec3 normal; Vec3 color; };
    Point EvalPatch(size_t patch, float u, float v,
                    float lod_shape, float lod_detail, float lod_color) const;

    size_t ByteSize() const;

private:
    std::vector<TreePatch> patches_;
};

} // namespace DeliRium
