#include "delirium/psf/AnchorPool.h"
#include <cmath>

namespace DeliRium {

void AnchorPool::Bind(std::vector<AnchorRecord>* b, Vec3 mn, Vec3 sp) {
    blob = b; bbmin = mn; bbspan = sp;
    dedupe.clear();
    for (uint16_t i = 0; i < blob->size(); ++i)
        dedupe.emplace(PosKey((*blob)[i]), i);
}

Vec3 AnchorPool::Position(uint16_t i) const {
    const AnchorRecord& a = (*blob)[i];
    const float k = 1.0f / 65535.0f;
    return { bbmin.x + bbspan.x * a.pos[0] * k,
             bbmin.y + bbspan.y * a.pos[1] * k,
             bbmin.z + bbspan.z * a.pos[2] * k };
}

Vec3 AnchorPool::Normal(uint16_t i) const {
    const AnchorRecord& a = (*blob)[i];
    const float k = 2.0f / 255.0f;
    float x = a.normal[0] * k - 1.0f;
    float y = a.normal[1] * k - 1.0f;
    float z = a.normal[2] * k - 1.0f;
    float l = std::sqrt(x*x + y*y + z*z);
    if (l < 1e-12f) return {0, 0, 1};
    return { x/l, y/l, z/l };
}

void AnchorPool::SetPosition(uint16_t i, const Vec3& p) {
    AnchorRecord& a = (*blob)[i];
    auto q = [](float t) {
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        return (uint16_t)(t * 65535.0f + 0.5f);
    };
    a.pos[0] = q((p.x - bbmin.x) / bbspan.x);
    a.pos[1] = q((p.y - bbmin.y) / bbspan.y);
    a.pos[2] = q((p.z - bbmin.z) / bbspan.z);
}

void AnchorPool::SetNormal(uint16_t i, const Vec3& n) {
    AnchorRecord& a = (*blob)[i];
    auto q = [](float t) {
        float v = (t * 0.5f + 0.5f) * 255.0f + 0.5f;
        v = v < 0 ? 0 : (v > 255 ? 255 : v);
        return (uint8_t)v;
    };
    a.normal[0] = q(n.x); a.normal[1] = q(n.y); a.normal[2] = q(n.z);
}

} // namespace DeliRium
