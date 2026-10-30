// =============================================================================
// AnchorPool.h - Shared-corner storage with dequantization helpers.
// Anchors are the ONLY ground-truth positions in the format; everything
// else is equations relative to them. Sharing them between patches is
// what makes cracks impossible at corners by construction.
// =============================================================================
#pragma once
#include <vector>
#include <unordered_map>
#include "../core/Types.h"

namespace DeliRium {

class AnchorPool {
public:
    void  Bind(std::vector<AnchorRecord>* blob, Vec3 bbmin, Vec3 bbspan);
    Vec3  Position(uint16_t idx) const;            // TODO dequant uint16->world
    Vec3  Normal(uint16_t idx) const;              // TODO dequant uint8, normalize
    void  SetPosition(uint16_t idx, const Vec3& p);// TODO quantize + dirty
    void  SetNormal(uint16_t idx, const Vec3& n);  // TODO
    // Create (or reuse, deduped by quantized position) an anchor sitting
    // exactly ON a coarse patch surface. Returns 0xFFFF on overflow.
    uint16_t CreateSnapped(const class Patch& coarse, float u, float v);
    size_t   Count() const { return blob ? blob->size() : 0; }
    // Raw record access for the CPU Patch layer (Patch.cpp evaluates the
    // surface from the packed 64-byte records, mirroring the GPU kernel).
    const std::vector<AnchorRecord>* Blob() const { return blob; }
    Vec3 BBoxMin()  const { return bbmin; }
    Vec3 BBoxSpan() const { return bbspan; }
private:
    uint64_t PosKey(const AnchorRecord& a) const;
    std::vector<AnchorRecord>* blob = nullptr;
    Vec3 bbmin{}, bbspan{};
    std::unordered_map<uint64_t, uint16_t> dedupe;   // packed pos -> idx
};

} // namespace DeliRium
