// =============================================================================
// PSFFile.h - Reader/writer of the .psf v3 container (self-contained):
//   64B header | AnchorRecord[] | SurfaceRecord[] | stitch int8[n_surf*8]
// Header: 0 magic "PSF\0" | 4 u32 version=3 | 8 u32 n_anchors
//         12 u32 n_surfaces | 16 u32 n_bones | 20 u32 flags (bit0 stitch)
//         24 f32 bbox_min[3] | 36 f32 bbox_span[3] | 48 f32 s_scale
// Produced by tools/psf_encode.py; reference decoder tools/psf_validate.py.
// =============================================================================
#pragma once
#include <string>
#include <vector>
#include "../core/Types.h"

namespace DeliRium {

struct PSFFile {
    uint32_t version = 3;
    Vec3  bbox_min{}, bbox_span{};
    float s_scale = 0.0f;
    std::vector<AnchorRecord>  anchors;
    std::vector<SurfaceRecord> surfaces;
    std::vector<int8_t>        stitch;     // 8 per surface (may be empty)
    std::vector<BoneRecord>    bones;

    static bool Load(const std::string& path, PSFFile& out, std::string* err = nullptr);
    bool        Save(const std::string& path) const;
};

} // namespace DeliRium
