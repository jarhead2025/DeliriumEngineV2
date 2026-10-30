#include "delirium/psf/AnchorPool.h"
#include "delirium/psf/Patch.h"

namespace DeliRium {

uint64_t AnchorPool::PosKey(const AnchorRecord& a) const {
    return (uint64_t(a.pos[0]) << 32) | (uint64_t(a.pos[1]) << 16) | a.pos[2];
}

uint16_t AnchorPool::CreateSnapped(const Patch& coarse, float u, float v) {
    if (!blob || blob->size() >= 0xFFFF) return 0xFFFF;
    AnchorRecord rec{};
    rec.connectivity[0] = rec.connectivity[1] = 0xFFFF;
    rec.connectivity[2] = rec.connectivity[3] = 0xFFFF;
    // position and displacement direction sampled ON the coarse surface
    const Vec3 p = coarse.Evaluate(u, v);
    const Vec3 n = coarse.EvaluateNormalField(u, v);
    blob->push_back(rec);
    const uint16_t idx = (uint16_t)(blob->size() - 1);
    SetPosition(idx, p);
    SetNormal(idx, n);
    // dedupe by exact quantized position (reuses hanging anchors created
    // by an earlier neighbor split when quantization coincides)
    const uint64_t key = PosKey((*blob)[idx]);
    auto it = dedupe.find(key);
    if (it != dedupe.end()) { blob->pop_back(); return it->second; }
    dedupe.emplace(key, idx);
    return idx;
}

} // namespace DeliRium
