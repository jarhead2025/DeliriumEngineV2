// FILEPATH: engine/src/psf/PSFBody.cpp  (replace FromFile / add FromRecords)
// Minimal factories so HasBody() can succeed after loading a real .psf
#include "delirium/psf/PSFBody.h"
#include "delirium/io/PSFFile.h"
#include <memory>
#include <utility>

namespace DeliRium {

std::unique_ptr<PSFBody> PSFBody::FromRecords(
    std::vector<AnchorRecord> a,
    std::vector<SurfaceRecord> s,
    std::vector<int8_t> stitch,
    Vec3 bbmin,
    Vec3 bbspan,
    float s_scale_in) {
    auto body = std::unique_ptr<PSFBody>(new PSFBody());
    body->anchor_blob = std::move(a);
    body->surface_blob = std::move(s);
    body->stitch_blob = std::move(stitch);
    body->bbox_min = bbmin;
    body->bbox_span = bbspan;
    body->s_scale = s_scale_in;
    body->anchors_dirty = true;
    // Patches/views can be built later; non-null body is enough for M1 HasBody.
    return body;
}

std::unique_ptr<PSFBody> PSFBody::FromFile(const char* path) {
    if (!path || !path[0]) return nullptr;
    PSFFile file;
    std::string err;
    if (!PSFFile::Load(path, file, &err)) return nullptr;
    if (file.anchors.empty() && file.surfaces.empty()) return nullptr;
    return FromRecords(std::move(file.anchors), std::move(file.surfaces),
                       std::move(file.stitch), file.bbox_min, file.bbox_span,
                       file.s_scale);
}

} // namespace DeliRium
