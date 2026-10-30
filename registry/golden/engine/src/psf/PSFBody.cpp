// FILEPATH: engine/src/psf/PSFBody.cpp
#include "delirium/psf/PSFBody.h"
#include "delirium/io/PSFFile.h"
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
    return body;
}

std::unique_ptr<PSFBody> PSFBody::FromFile(const char* path) {
    if (!path || !path[0]) {
        std::fprintf(stderr, "PSFBody::FromFile: empty path\n");
        return nullptr;
    }
    PSFFile file;
    std::string err;
    if (!PSFFile::Load(path, file, &err)) {
        std::fprintf(stderr, "PSFBody::FromFile: Load failed for '%s': %s\n",
                     path, err.c_str());
        return nullptr;
    }
    if (file.anchors.empty() && file.surfaces.empty()) {
        std::fprintf(stderr, "PSFBody::FromFile: no anchors/surfaces in '%s'\n", path);
        return nullptr;
    }
    std::fprintf(stderr, "PSFBody::FromFile: OK '%s' anchors=%zu surfaces=%zu ver=%u\n",
                 path, file.anchors.size(), file.surfaces.size(), file.version);
    return FromRecords(std::move(file.anchors), std::move(file.surfaces),
                       std::move(file.stitch), file.bbox_min, file.bbox_span,
                       file.s_scale);
}

bool PSFBody::SplitPatch(uint32_t patch_id, uint32_t children_out[4]) {
    (void)patch_id; (void)children_out; return false;
}
void PSFBody::MergePatches(uint32_t parent_id, float tol) { (void)parent_id; (void)tol; }
bool PSFBody::MergeGroup(uint16_t ctr, float tol) { (void)ctr; (void)tol; return false; }
bool PSFBody::FindMergeGroup(uint16_t ctr, uint32_t grp[4], uint16_t outer[4]) const {
    (void)ctr; (void)grp; (void)outer; return false;
}
std::unique_ptr<PSFBody> PSFBody::DetachPatches(const std::vector<uint32_t>& ids) {
    (void)ids; return nullptr;
}
void PSFBody::ApplyRadialDent(const Vec3& center, float radius, float depth) {
    (void)center; (void)radius; (void)depth;
}
void PSFBody::RefitLocal(const Vec3& center, float radius) { (void)center; (void)radius; }
void PSFBody::ApplyMode(uint32_t mode_id, float k) { (void)mode_id; (void)k; }
float PSFBody::Volume() const { return 0; }
bool PSFBody::ClosestPointEx(const Vec3& p, SurfaceHit& out) const { (void)p; (void)out; return false; }
Vec3 PSFBody::ClosestPoint(const Vec3& p) const { (void)p; (void)0; }
bool PSFBody::Raycast(const Vec3& o, const Vec3& d, SurfaceHit& out) const {
    (void)o; (void)d; (void)out; return false;
}
void PSFBody::FlushToGPU(void* cl_queue) { (void)cl_queue; }
void PSFBody::RebuildViews() {}
void PSFBody::ExpandBBox(float margin) { (void)margin; }
const std::vector<Vec3>& PSFBody::ComputeSmoothNormals() const {
    static std::vector<Vec3> empty;
    return empty;
}
float PSFBody::RederiveNormals() { return 0; }

} // namespace DeliRium
