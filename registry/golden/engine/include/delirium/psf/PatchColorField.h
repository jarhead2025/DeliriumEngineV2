// =============================================================================
// PatchColorField.h - 2D cosine color field baked onto a patch's (u,v) domain.
//
// THE COLOR LAYER (rev-15 vision restored): the FIGURE is the amalgamated
// patches (geometry, untouched). This is ONLY the color/texture on top of a
// patch surface -- it never affects shape. Each patch stores a small set of
// 2D cosine coefficients per channel:
//     color(u,v) = sum_{i,j<=L} K_ij * cos(pi i u) * cos(pi j v)   per channel
// This replaces the old 4-pattern color (Sum K_i P_i, limited to a 16-entry
// dictionary) with a field that represents ARBITRARY texture, baked from the
// model's texture via UVs.
//
// FREQUENCY LOD (the user's idea, applied to color where it belongs): summing
// only low (i,j) orders gives coarse color (distant patch); adding higher
// orders reveals fine texture (near patch), continuously, no popping. The
// cosine basis is the natural choice for a flat (u,v) patch domain (spherical
// harmonics are for spheres, not patches).
// =============================================================================
#pragma once
#include <cstdint>
#include <vector>
#include "../core/Types.h"

namespace DeliRium {

class PatchColorField {
public:
    PatchColorField() = default;

    // Build from per-channel coefficients (order L: (L+1)^2 coefs/channel,
    // row-major i*(L+1)+j). coeffs layout: channel-major [3][(L+1)^2].
    void Set(int order, const float* r, const float* g, const float* b);

    // Evaluate color at (u,v) in [0,1]^2. lod_order is a CONTINUOUS order:
    // bands with max(i,j) <= floor(lod) full, the next weighted by frac --
    // smooth frequency LOD, no popping. Clamped to the stored order.
    Vec3 Eval(float u, float v, float lod_order) const;
    // Full-detail evaluation (lod = stored order).
    Vec3 Eval(float u, float v) const { return Eval(u, v, (float)order_); }

    int  Order() const { return order_; }
    bool Valid() const { return order_ >= 0; }
    size_t ByteSize() const { return coeff_.size() * sizeof(float); }
    const std::vector<float>& Coeffs() const { return coeff_; }

private:
    int order_ = -1;
    std::vector<float> coeff_;   // [3][(order+1)^2], channel-major
};

} // namespace DeliRium
