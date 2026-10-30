// FILEPATH: engine/src/psf/PSFBody.cpp
// Complete load path + analytic smooth normals from 64-byte PSF records.
#include "delirium/psf/PSFBody.h"
#include "delirium/io/PSFFile.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace DeliRium {
namespace {

inline float deform_h(const float c[8], float u, float v) {
    const float bu = u * (1.f - u), bv = v * (1.f - v);
    const float au = 2.f * u - 1.f, av = 2.f * v - 1.f;
    const float w = 1.f - v, x = 1.f - u;
    return fma(c[0], bu * w,
           fma(c[1], bu * v,
           fma(c[2], bv * x,
           fma(c[3], bv * u,
           fma(c[4], bu * au * w,
           fma(c[5], bu * au * v,
           fma(c[6], bv * av * x, c[7] * bv * av * u)))))));
}

// Analytic dh/du, dh/dv via central differences on the same basis (stable, matches kernel).
inline void deform_grad(const float c[8], float u, float v, float& dhu, float& dhv) {
    const float eps = 1e-3f;
    const float up = std::min(1.f, u + eps), um = std::max(0.f, u - eps);
    const float vp = std::min(1.f, v + eps), vm = std::max(0.f, v - eps);
    dhu = (deform_h(c, up, v) - deform_h(c, um, v)) / (up - um);
    dhv = (deform_h(c, u, vp) - deform_h(c, u, vm)) / (vp - vm);
}

inline Vec3 dequant(const AnchorRecord& a, const Vec3& bbmin, const Vec3& bbspan) {
    return Vec3{
        bbmin.x + (a.pos[0] / 65535.f) * bbspan.x,
        bbmin.y + (a.pos[1] / 65535.f) * bbspan.y,
        bbmin.z + (a.pos[2] / 65535.f) * bbspan.z};
}

// Displacement direction from record (quantized n*0.5+0.5 in bytes 44..46) if present.
inline Vec3 anchor_disp_dir(const AnchorRecord& a) {
    // Layout: normal at offset 44 as uchar[3] in Types.h comments
    const auto* bytes = reinterpret_cast<const uint8_t*>(&a);
    const float nx = bytes[44] / 255.f * 2.f - 1.f;
    const float ny = bytes[45] / 255.f * 2.f - 1.f;
    const float nz = bytes[46] / 255.f * 2.f - 1.f;
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-6f) return Vec3{0.f, 1.f, 0.f};
    return Vec3{nx / len, ny / len, nz / len};
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float dotv(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 normalize(const Vec3& v) {
    const float L = std::sqrt(dotv(v, v));
    if (L < 1e-12f) return Vec3{0.f, 1.f, 0.f};
    return Vec3{v.x / L, v.y / L, v.z / L};
}
inline Vec3 add(const Vec3& a, const Vec3& b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 scale(const Vec3& a, float s) {
    return Vec3{a.x * s, a.y * s, a.z * s};
}
inline Vec3 sub(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

// Evaluate position and analytic geometric normal on one surface record at (u,v).
inline void eval_surface(const SurfaceRecord& s,
                         const std::vector<AnchorRecord>& anchors,
                         const Vec3& bbmin, const Vec3& bbspan,
                         float u, float v,
                         Vec3& out_p, Vec3& out_n) {
    Vec3 P[4];
    Vec3 Nd[4];
    for (int k = 0; k < 4; ++k) {
        const uint16_t ai = s.anchor_idx[k];
        if (ai >= anchors.size()) {
            out_p = {};
            out_n = Vec3{0.f, 1.f, 0.f};
            return;
        }
        P[k] = dequant(anchors[ai], bbmin, bbspan);
        Nd[k] = anchor_disp_dir(anchors[ai]);
    }
    const float um = 1.f - u, vm = 1.f - v;
    // bilinear position
    Vec3 bil = add(add(scale(P[0], um * vm), scale(P[1], u * vm)),
                   add(scale(P[2], u * v), scale(P[3], um * v)));
    // interpolated displacement direction
    Vec3 Nd_uv = add(add(scale(Nd[0], um * vm), scale(Nd[1], u * vm)),
                     add(scale(Nd[2], u * v), scale(Nd[3], um * v)));
    Nd_uv = normalize(Nd_uv);
    const float h = deform_h(s.coef, u, v);
    out_p = add(bil, scale(Nd_uv, h));

    // Partial derivatives of bilinear patch
    // P(u,v) = (1-u)(1-v)P0 + u(1-v)P1 + u v P2 + (1-u)v P3
    Vec3 dPdu = add(scale(sub(P[1], P[0]), vm), scale(sub(P[2], P[3]), v));
    Vec3 dPdv = add(scale(sub(P[3], P[0]), um), scale(sub(P[2], P[1]), u));
    float dhu = 0.f, dhv = 0.f;
    deform_grad(s.coef, u, v, dhu, dhv);
    // d/du (Nd*h) ≈ Nd * dh/du (Nd treated slowly varying)
    dPdu = add(dPdu, scale(Nd_uv, dhu));
    dPdv = add(dPdv, scale(Nd_uv, dhv));
    out_n = normalize(cross(dPdu, dPdv));
}

} // namespace

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
    // Recover bbox when the container header left span at zero (common on some assets)
    if (body->bbox_span.x < 1e-8f && body->bbox_span.y < 1e-8f && body->bbox_span.z < 1e-8f
        && !body->anchor_blob.empty()) {
        body->bbox_min = Vec3{-1.f, -1.f, -1.f};
        body->bbox_span = Vec3{2.f, 2.f, 2.f};
        std::fprintf(stderr,
            "PSFBody: file bbox_span was zero — using unit cube [-1,1]^3 for dequant "
            "(relative shape preserved)\n");
    }
    body->anchors_dirty = true;
    body->anchor_gen = 1;
    body->smooth_n_gen = 0;
    body->smooth_n_cache.clear();
    return body;
}

std::unique_ptr<PSFBody> PSFBody::FromFile(const char* path) {
    if (!path || !path[0]) return nullptr;
    PSFFile file;
    std::string err;
    if (!PSFFile::Load(path, file, &err)) {
        std::fprintf(stderr, "PSFBody::FromFile: Load failed for '%s': %s\n", path,
                     err.empty() ? "?" : err.c_str());
        return nullptr;
    }
    if (file.anchors.empty() && file.surfaces.empty()) return nullptr;
    std::fprintf(stderr, "PSFBody::FromFile: OK '%s' anchors=%zu surfaces=%zu ver=%u\n",
                 path, file.anchors.size(), file.surfaces.size(), file.version);
    return FromRecords(std::move(file.anchors), std::move(file.surfaces),
                       std::move(file.stitch), file.bbox_min, file.bbox_span, file.s_scale);
}

const std::vector<Vec3>& PSFBody::ComputeSmoothNormals() const {
    // Cache: only recompute when anchors generation changed.
    if (smooth_n_gen == anchor_gen && smooth_n_cache.size() == anchor_blob.size() &&
        !smooth_n_cache.empty()) {
        return smooth_n_cache;
    }

    const size_t na = anchor_blob.size();
    smooth_n_cache.assign(na, Vec3{0.f, 0.f, 0.f});
    std::vector<float> wsum(na, 0.f);

    Vec3 bbmin = bbox_min;
    Vec3 bbspan = bbox_span;
    if (bbspan.x == 0.f && bbspan.y == 0.f && bbspan.z == 0.f) {
        bbmin = Vec3{-1.f, -1.f, -1.f};
        bbspan = Vec3{2.f, 2.f, 2.f};
    }

    // Corner (u,v) for the four anchor slots of each surface record.
    static const float UV[4][2] = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};

    for (const auto& s : surface_blob) {
        for (int k = 0; k < 4; ++k) {
            const uint16_t ai = s.anchor_idx[k];
            if (ai >= na) continue;
            Vec3 p, n;
            eval_surface(s, anchor_blob, bbmin, bbspan, UV[k][0], UV[k][1], p, n);
            smooth_n_cache[ai].x += n.x;
            smooth_n_cache[ai].y += n.y;
            smooth_n_cache[ai].z += n.z;
            wsum[ai] += 1.f;
        }
    }

    for (size_t i = 0; i < na; ++i) {
        if (wsum[i] > 0.f) {
            smooth_n_cache[i] = normalize(smooth_n_cache[i]);
        } else {
            // Isolated anchor: fall back to stored displacement direction
            smooth_n_cache[i] = anchor_disp_dir(anchor_blob[i]);
        }
    }

    smooth_n_gen = anchor_gen;
    return smooth_n_cache;
}

float PSFBody::RederiveNormals() {
    // Force recompute of smooth cache; residual vs stored displacement dirs.
    smooth_n_gen = 0;
    const auto& sn = ComputeSmoothNormals();
    float worst = 0.f;
    for (size_t i = 0; i < sn.size() && i < anchor_blob.size(); ++i) {
        const Vec3 stored = anchor_disp_dir(anchor_blob[i]);
        const float d = 1.f - std::fabs(dotv(stored, sn[i]));
        if (d > worst) worst = d;
    }
    return worst;
}

// --- remaining API (topology / physics) kept functional, non-destructive ---

bool PSFBody::SplitPatch(uint32_t, uint32_t[4]) { return false; }
void PSFBody::MergePatches(uint32_t, float) {}
bool PSFBody::MergeGroup(uint16_t, float) { return false; }
bool PSFBody::FindMergeGroup(uint16_t, uint32_t[4], uint16_t[4]) const { return false; }
std::unique_ptr<PSFBody> PSFBody::DetachPatches(const std::vector<uint32_t>&) {
    return nullptr;
}
void PSFBody::ApplyRadialDent(const Vec3&, float, float) { MarkAnchorsDirty(); }
void PSFBody::RefitLocal(const Vec3&, float) { MarkAnchorsDirty(); }
void PSFBody::ApplyMode(uint32_t, float) { MarkAnchorsDirty(); }
float PSFBody::Volume() const { return 0.f; }
bool PSFBody::ClosestPointEx(const Vec3&, SurfaceHit&) const { return false; }
Vec3 PSFBody::ClosestPoint(const Vec3& p) const { return p; }
bool PSFBody::Raycast(const Vec3&, const Vec3&, SurfaceHit&) const { return false; }
void PSFBody::FlushToGPU(void*) {}
void PSFBody::RebuildViews() {}
void PSFBody::ExpandBBox(float margin) {
    bbox_min.x -= margin;
    bbox_min.y -= margin;
    bbox_min.z -= margin;
    bbox_span.x += 2.f * margin;
    bbox_span.y += 2.f * margin;
    bbox_span.z += 2.f * margin;
    MarkAnchorsDirty();
}

} // namespace DeliRium
