// =============================================================================
// ProceduralDetail.cpp - Analytic fractal noise with exact derivatives.
// Uses gradient (Perlin-style) noise with a quintic fade, so both the
// value AND its gradient are smooth and analytic -- essential because the
// detail must perturb the surface normal correctly for lighting. The
// noise is C2, the fade is the standard 6t^5-15t^4+10t^3, and the chain
// rule is carried through every octave. Deterministic from an integer
// seed; no tables loaded at runtime (matches the "own kernels, no deps"
// philosophy -- this compiles unchanged into OpenCL).
// =============================================================================
#include "delirium/psf/ProceduralDetail.h"
#include <cmath>

namespace DeliRium {

// ---- integer hash -> gradient (deterministic, table-free) ------------------
static inline uint32_t hash2(int xi, int yi, uint32_t seed) {
    uint32_t h = seed + 0x9E3779B9u;
    h ^= (uint32_t)xi * 0x85EBCA6Bu; h = (h << 13) | (h >> 19); h *= 0xC2B2AE35u;
    h ^= (uint32_t)yi * 0x27D4EB2Fu; h = (h << 15) | (h >> 17); h *= 0x165667B1u;
    h ^= h >> 16;
    return h;
}
static inline void grad2(int xi, int yi, uint32_t seed, float& gx, float& gy) {
    // 8 evenly spaced unit gradients (cheap, isotropic enough for detail)
    const uint32_t h = hash2(xi, yi, seed) & 7u;
    const float a = (float)h * 0.78539816339f;   // h * 45deg
    gx = std::cos(a); gy = std::sin(a);
}

// quintic fade and its derivative
static inline float fade(float t)  { return t*t*t*(t*(t*6.f-15.f)+10.f); }
static inline float dfade(float t) { return 30.f*t*t*(t*(t-2.f)+1.f); }

// single-octave gradient noise at (x,y); also returns d/dx, d/dy
static float pnoise(float x, float y, uint32_t seed, float& dnx, float& dny) {
    const int xi = (int)std::floor(x), yi = (int)std::floor(y);
    const float fx = x - xi, fy = y - yi;
    float g00x,g00y,g10x,g10y,g01x,g01y,g11x,g11y;
    grad2(xi,   yi,   seed, g00x, g00y);
    grad2(xi+1, yi,   seed, g10x, g10y);
    grad2(xi,   yi+1, seed, g01x, g01y);
    grad2(xi+1, yi+1, seed, g11x, g11y);
    // dot of gradient with distance vector at each corner
    const float n00 = g00x*fx       + g00y*fy;
    const float n10 = g10x*(fx-1.f) + g10y*fy;
    const float n01 = g01x*fx       + g01y*(fy-1.f);
    const float n11 = g11x*(fx-1.f) + g11y*(fy-1.f);
    const float u = fade(fx), v = fade(fy);
    const float du = dfade(fx), dv = dfade(fy);
    // bilinear blend of corner dots with faded weights
    const float nx0 = n00 + u*(n10-n00);
    const float nx1 = n01 + u*(n11-n01);
    const float val = nx0 + v*(nx1-nx0);
    // analytic partials (product/chain rule through the fade)
    // d(n..)/dx = gradient x-component; d(n..)/dy = gradient y-component
    const float dnx0dx = g00x + u*(g10x-g00x) + du*(n10-n00);
    const float dnx1dx = g01x + u*(g11x-g01x) + du*(n11-n01);
    dnx = dnx0dx + v*(dnx1dx - dnx0dx);
    const float dnx0dy = g00y + u*(g10y-g00y);
    const float dnx1dy = g01y + u*(g11y-g01y);
    dny = dnx0dy + v*(dnx1dy - dnx0dy) + dv*(nx1-nx0);
    return val;
}

DetailParams ProceduralDetail::Decode(uint32_t sd, float amp_scale) {
    DetailParams p;
    p.amplitude   = ((sd & 0xFF)) / 255.0f * amp_scale;
    p.base_period = (float)(1u << (((sd >> 8) & 0xF) + 1));
    p.octaves     = (int)((sd >> 12) & 0x7);
    p.type        = (int)((sd >> 15) & 0x7);
    p.lacunarity  = 1.7f + 0.1f * ((sd >> 18) & 0x7);
    p.gain        = 0.35f + 0.05f * ((sd >> 21) & 0x7);
    p.seed        = (sd >> 24) & 0xFF;
    return p;
}

uint32_t ProceduralDetail::Encode(const DetailParams& p, float amp_scale) {
    uint32_t sd = 0;
    int amp = (int)std::lround(p.amplitude / (amp_scale <= 0 ? 1 : amp_scale) * 255.0f);
    amp = amp < 0 ? 0 : (amp > 255 ? 255 : amp);
    sd |= (uint32_t)amp & 0xFF;
    int per = (int)std::lround(std::log2(std::fmax(p.base_period,2.0f))) - 1;
    per = per < 0 ? 0 : (per > 15 ? 15 : per);
    sd |= (uint32_t)per << 8;
    sd |= (uint32_t)(p.octaves & 0x7) << 12;
    sd |= (uint32_t)(p.type & 0x7) << 15;
    int lac = (int)std::lround((p.lacunarity - 1.7f) / 0.1f);
    lac = lac < 0 ? 0 : (lac > 7 ? 7 : lac);
    sd |= (uint32_t)lac << 18;
    int gn = (int)std::lround((p.gain - 0.35f) / 0.05f);
    gn = gn < 0 ? 0 : (gn > 7 ? 7 : gn);
    sd |= (uint32_t)gn << 21;
    sd |= (p.seed & 0xFF) << 24;
    return sd;
}

// shape one octave's value+grad according to type (carrying the chain rule)
static inline void shape(int type, float n, float dnx, float dny,
                         float& s, float& dsx, float& dsy) {
    if (type == 1) {            // ridged: 1 - |n|, sharpened
        const float a = std::fabs(n);
        const float r = 1.0f - a;
        s = r*r;                // squared ridge
        const float sign = n < 0 ? -1.0f : 1.0f;
        const float dr = -2.0f*r*sign;
        dsx = dr*dnx; dsy = dr*dny;
    } else if (type == 2) {     // billow: |n|
        const float a = std::fabs(n);
        s = 2.0f*a - 1.0f;
        const float sign = n < 0 ? -1.0f : 1.0f;
        dsx = 2.0f*sign*dnx; dsy = 2.0f*sign*dny;
    } else {                    // fbm: n
        s = n; dsx = dnx; dsy = dny;
    }
}

float ProceduralDetail::Evaluate(const DetailParams& p, float u, float v) {
    if (!p.enabled()) return 0.0f;
    float freq = p.base_period, amp = 1.0f, sum = 0.0f, norm = 0.0f;
    for (int o = 0; o < p.octaves; ++o) {
        float dnx, dny;
        const float n = pnoise(u*freq, v*freq, p.seed + o*101u, dnx, dny);
        float s, dsx, dsy; shape(p.type, n, dnx, dny, s, dsx, dsy);
        sum += amp * s; norm += amp;
        amp *= p.gain; freq *= p.lacunarity;
    }
    return p.amplitude * (norm > 0 ? sum / norm : 0.0f);
}

void ProceduralDetail::Gradient(const DetailParams& p, float u, float v,
                                float& dd_du, float& dd_dv) {
    dd_du = dd_dv = 0.0f;
    if (!p.enabled()) return;
    float freq = p.base_period, amp = 1.0f, norm = 0.0f;
    float gu = 0.0f, gv = 0.0f;
    for (int o = 0; o < p.octaves; ++o) {
        float dnx, dny;
        const float n = pnoise(u*freq, v*freq, p.seed + o*101u, dnx, dny);
        float s, dsx, dsy; shape(p.type, n, dnx, dny, s, dsx, dsy);
        // chain rule: d/du of noise(u*freq) = freq * dnx
        gu += amp * dsx * freq;
        gv += amp * dsy * freq;
        norm += amp;
        amp *= p.gain; freq *= p.lacunarity;
    }
    const float k = p.amplitude * (norm > 0 ? 1.0f/norm : 0.0f);
    dd_du = k * gu; dd_dv = k * gv;
}

} // namespace DeliRium
