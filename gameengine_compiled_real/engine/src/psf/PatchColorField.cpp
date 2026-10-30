// =============================================================================
// PatchColorField.cpp
// =============================================================================
#include "delirium/psf/PatchColorField.h"
#include <cmath>
#include <algorithm>

namespace DeliRium {

void PatchColorField::Set(int order, const float* r, const float* g,
                          const float* b) {
    order_ = order;
    const int n = (order + 1) * (order + 1);
    coeff_.resize(3 * n);
    std::copy(r, r + n, coeff_.begin());
    std::copy(g, g + n, coeff_.begin() + n);
    std::copy(b, b + n, coeff_.begin() + 2 * n);
}

Vec3 PatchColorField::Eval(float u, float v, float lod_order) const {
    if (order_ < 0) return Vec3{0.6f, 0.6f, 0.6f};
    const int n = (order_ + 1) * (order_ + 1);
    // continuous LOD: full bands up to floor(lod), the next weighted by frac.
    float lod = std::min(lod_order, (float)order_);
    int Lint = (int)std::floor(lod);
    float frac = lod - (float)Lint;
    if (Lint < 0) { Lint = 0; frac = 0.0f; }

    // precompute cos(pi*i*u), cos(pi*j*v) up to the stored order
    const float pu = (float)M_PI * u, pv = (float)M_PI * v;
    Vec3 c{0, 0, 0};
    for (int i = 0; i <= order_; ++i) {
        float ci = std::cos(pu * i);
        for (int j = 0; j <= order_; ++j) {
            int band = std::max(i, j);
            float w;
            if (band <= Lint)        w = 1.0f;
            else if (band == Lint+1) w = frac;   // fading-in band
            else                     w = 0.0f;
            if (w == 0.0f) continue;
            float cj = std::cos(pv * j);
            float basis = ci * cj * w;
            int idx = i * (order_ + 1) + j;
            c.x += coeff_[idx]         * basis;
            c.y += coeff_[idx + n]     * basis;
            c.z += coeff_[idx + 2 * n] * basis;
        }
    }
    c.x = std::min(std::max(c.x, 0.0f), 1.0f);
    c.y = std::min(std::max(c.y, 0.0f), 1.0f);
    c.z = std::min(std::max(c.z, 0.0f), 1.0f);
    return c;
}

} // namespace DeliRium
