// FILEPATH: engine/src/io/PSFFile.cpp
// PSF v2/v3 container: 64B header | AnchorRecord[n] | SurfaceRecord[n] | optional stitch | bones
#include "delirium/io/PSFFile.h"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace DeliRium {

bool PSFFile::Load(const std::string& path, PSFFile& out, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "open failed: " + path;
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff sz = in.tellg();
    in.seekg(0, std::ios::beg);
    if (sz < 64) {
        if (err) *err = "file too small: " + path;
        return false;
    }
    std::vector<char> buf(static_cast<size_t>(sz));
    if (!in.read(buf.data(), sz)) {
        if (err) *err = "read failed: " + path;
        return false;
    }
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf.data());
    if (std::memcmp(p, "PSF\0", 4) != 0) {
        if (err) *err = "bad magic (want PSF\\0): " + path;
        return false;
    }
    uint32_t version = 0, n_anchors = 0, n_surfaces = 0, n_bones = 0, flags = 0;
    std::memcpy(&version, p + 4, 4);
    std::memcpy(&n_anchors, p + 8, 4);
    std::memcpy(&n_surfaces, p + 12, 4);
    std::memcpy(&n_bones, p + 16, 4);
    std::memcpy(&flags, p + 20, 4);
    if (version < 1 || version > 3) {
        if (err) *err = "unsupported PSF version";
        return false;
    }
    float bbox[7] = {};
    std::memcpy(bbox, p + 24, sizeof(bbox));
    out = PSFFile{};
    out.version = version;
    out.bbox_min = {bbox[0], bbox[1], bbox[2]};
    out.bbox_span = {bbox[3], bbox[4], bbox[5]};
    out.s_scale = bbox[6];

    const size_t need =
        64u + size_t(n_anchors) * 64u + size_t(n_surfaces) * 64u;
    if (buf.size() < need) {
        if (err) *err = "truncated body records";
        return false;
    }
    size_t off = 64;
    out.anchors.resize(n_anchors);
    if (n_anchors) {
        std::memcpy(out.anchors.data(), p + off, n_anchors * 64u);
        off += n_anchors * 64u;
    }
    out.surfaces.resize(n_surfaces);
    if (n_surfaces) {
        std::memcpy(out.surfaces.data(), p + off, n_surfaces * 64u);
        off += n_surfaces * 64u;
    }
    // optional stitch (8 bytes per surface) if flags bit0 or leftover matches
    const size_t stitch_bytes = size_t(n_surfaces) * 8u;
    if ((flags & 1u) && off + stitch_bytes <= buf.size()) {
        out.stitch.resize(stitch_bytes);
        std::memcpy(out.stitch.data(), p + off, stitch_bytes);
        off += stitch_bytes;
    }
    if (n_bones && off + size_t(n_bones) * 64u <= buf.size()) {
        out.bones.resize(n_bones);
        std::memcpy(out.bones.data(), p + off, n_bones * 64u);
    }
    if (err) err->clear();
    return true;
}

bool PSFFile::Save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    char hdr[64] = {};
    std::memcpy(hdr, "PSF\0", 4);
    uint32_t version = this->version ? this->version : 3u;
    uint32_t n_a = static_cast<uint32_t>(anchors.size());
    uint32_t n_s = static_cast<uint32_t>(surfaces.size());
    uint32_t n_b = static_cast<uint32_t>(bones.size());
    uint32_t flags = stitch.empty() ? 0u : 1u;
    std::memcpy(hdr + 4, &version, 4);
    std::memcpy(hdr + 8, &n_a, 4);
    std::memcpy(hdr + 12, &n_s, 4);
    std::memcpy(hdr + 16, &n_b, 4);
    std::memcpy(hdr + 20, &flags, 4);
    float bbox[7] = {bbox_min.x, bbox_min.y, bbox_min.z,
                     bbox_span.x, bbox_span.y, bbox_span.z, s_scale};
    std::memcpy(hdr + 24, bbox, sizeof(bbox));
    out.write(hdr, 64);
    if (!anchors.empty())
        out.write(reinterpret_cast<const char*>(anchors.data()),
                  anchors.size() * 64);
    if (!surfaces.empty())
        out.write(reinterpret_cast<const char*>(surfaces.data()),
                  surfaces.size() * 64);
    if (!stitch.empty())
        out.write(reinterpret_cast<const char*>(stitch.data()), stitch.size());
    if (!bones.empty())
        out.write(reinterpret_cast<const char*>(bones.data()), bones.size() * 64);
    return bool(out);
}

} // namespace DeliRium
