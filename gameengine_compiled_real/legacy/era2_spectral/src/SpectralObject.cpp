// =============================================================================
// SpectralObject.cpp
// =============================================================================
#include "delirium/psf/SpectralObject.h"
#include <cmath>

namespace DeliRium {

// rotate vector v by unit quaternion q
static Vec3 quat_rot(const Quat& q, const Vec3& v) {
    // t = 2 * cross(q.xyz, v); v' = v + q.w*t + cross(q.xyz, t)
    const Vec3 u{q.x, q.y, q.z};
    Vec3 t{ 2.0f*(u.y*v.z - u.z*v.y),
            2.0f*(u.z*v.x - u.x*v.z),
            2.0f*(u.x*v.y - u.y*v.x) };
    return Vec3{ v.x + q.w*t.x + (u.y*t.z - u.z*t.y),
                 v.y + q.w*t.y + (u.z*t.x - u.x*t.z),
                 v.z + q.w*t.z + (u.x*t.y - u.y*t.x) };
}

Vec3 SpectralObject::SurfaceWorld(float theta, float phi) const {
    Vec3 p = prim->Surface(theta, phi);          // local (centered on prim)
    p = Vec3{ p.x*transform.scale.x, p.y*transform.scale.y, p.z*transform.scale.z };
    p = quat_rot(transform.rotation, p);
    return Vec3{ p.x+transform.position.x, p.y+transform.position.y,
                 p.z+transform.position.z };
}

uint32_t SpectralObject::LodForScreenSize(float projected_px) const {
    if (lod_order != 0) return lod_order;        // fixed override
    // continuous LOD: pick SH order from projected size. Measured fidelity:
    // order 4 ~0.83% (far), 8 ~0.79% (medium), 16 ~0.46% (near), 32 0.15%.
    const uint32_t maxL = prim->GeomOrder();
    uint32_t L;
    if      (projected_px < 16)   L = 2;
    else if (projected_px < 48)   L = 4;
    else if (projected_px < 128)  L = 8;
    else if (projected_px < 320)  L = 16;
    else                          L = 32;
    return L < maxL ? L : maxL;
}

SpectralObject::Lod
SpectralObject::LodPairForScreenSize(float projected_px, uint32_t color_max) const {
    // geometry order from size (same ladder as above)
    uint32_t g = LodForScreenSize(projected_px);
    // color order tracks the SAME distance: far -> few color terms (soft
    // texture, cheap), near -> many terms (texture sharpens). Color rarely
    // needs more than geometry and is capped at the baked color order, so
    // the cow's spots blur smoothly with distance and crisp up close.
    uint32_t c;
    if      (projected_px < 16)   c = 2;
    else if (projected_px < 48)   c = 4;
    else if (projected_px < 128)  c = 8;
    else if (projected_px < 320)  c = 14;
    else                          c = 24;
    if (c > color_max) c = color_max;
    return Lod{g, c};
}

bool SpectralModel::LoadPart(const std::string& spf_path, const SpectralTransform& local) {
    auto p = std::make_shared<SpectralPrimitive>();
    if (!p->Load(spf_path)) return false;
    parts_.push_back(SpectralObject{p, local, 0});
    return true;
}

void SpectralModel::AddPart(std::shared_ptr<SpectralPrimitive> p,
                            const SpectralTransform& local) {
    parts_.push_back(SpectralObject{p, local, 0});
}

size_t SpectralModel::ByteSize() const {
    size_t s = 0;
    for (const auto& part : parts_) s += part.prim->ByteSize();
    return s;
}

} // namespace DeliRium
