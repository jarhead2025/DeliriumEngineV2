// =============================================================================
// ProceduralDetail.h - High-frequency surface detail as EQUATIONS, not data.
// The format compresses smooth surfaces superbly but degrades to ~6x on
// rough "hyperreal" content (pores, grain, roughness). Instead of storing
// that detail, we GENERATE it: surface_def packs a compact descriptor of a
// fractal (fBm) displacement field that the splat evaluates analytically,
// at any resolution, sub-pixel, for zero extra bytes. This is the math
// equivalent of a normal map -- but it is real displacement with an exact
// analytic gradient, so silhouettes and lighting are correct, and it costs
// nothing in the file (the 6x floor becomes unbounded for self-similar
// detail).
//
// surface_def (uint32) layout:
//   bits  0..7  : amplitude   (q8, * amp_scale below -> height units)
//   bits  8..11 : base period in (u,v) cells = 2^(this+1)  (2..65536)
//   bits 12..14 : octaves (0..7); 0 disables detail
//   bits 15..17 : type (0 fbm, 1 ridged, 2 billow)
//   bits 18..20 : lacunarity index -> 1.7 + 0.1*idx (1.7..2.4)
//   bits 21..23 : gain index       -> 0.35 + 0.05*idx (0.35..0.70)
//   bits 24..31 : seed (per-patch decorrelation)
// amp_scale is a fraction of the patch's parametric domain; the detail is
// added along the patch normal field so it follows curvature.
// =============================================================================
#pragma once
#include <cstdint>

namespace DeliRium {

struct DetailParams {
    float    amplitude;     // height units (already scaled)
    float    base_period;   // in (u,v) cells
    int      octaves;
    int      type;          // 0 fbm, 1 ridged, 2 billow
    float    lacunarity;
    float    gain;
    uint32_t seed;
    bool enabled() const { return octaves > 0 && amplitude != 0.0f; }
};

class ProceduralDetail {
public:
    // amp_scale converts the 8-bit amplitude code to height units (set by
    // the body from its diagonal so detail is resolution-independent).
    static DetailParams Decode(uint32_t surface_def, float amp_scale);
    static uint32_t     Encode(const DetailParams& p, float amp_scale);

    // detail displacement at (u,v) and its analytic gradient (d/du, d/dv)
    static float Evaluate(const DetailParams& p, float u, float v);
    static void  Gradient(const DetailParams& p, float u, float v,
                          float& dd_du, float& dd_dv);
};

} // namespace DeliRium
