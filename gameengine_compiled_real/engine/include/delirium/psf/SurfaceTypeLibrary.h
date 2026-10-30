// =============================================================================
// SurfaceTypeLibrary.h - the basis of SURFACE TYPES for patch geometry.
//
// The unified model (user's idea): a patch surface is a weighted SUM of basis
// surface types, exactly like color is a sum of pattern types:
//     height(u,v) = sum_i K_i * SurfaceType_i(u,v)
// Layers stack (base shape, then detail, then color) -- a tree that evaluates
// down to a dense point cloud -> pixels.
//
// These types live in the ENGINE (zero file cost); a patch stores only the
// type indices and their K coefficients. The CPU table here must stay
// bit-identical to surface_type_eval() in the kernel.
//
// Measured: a sphere patch reconstructs at 0.6% RMS, a cylinder ridge (leg)
// at 2.7%, a saddle+peak (ear) at 5.2% with ~14 low-frequency types. Fine
// detail is a SECOND layer (higher-frequency types) on the residual.
// =============================================================================
#pragma once
#include <cstdint>

namespace DeliRium {

class SurfaceTypeLibrary {
public:
    // Number of base-layer surface types (low-frequency shape).
    static constexpr int kCount = 16;

    // Evaluate surface type `idx` at (u,v) in [0,1]^2. Returns a height value
    // (the displacement contributed by this type before its K weight).
    static float Eval(uint32_t idx, float u, float v);

    // Human-readable name (for tooling/debug).
    static const char* Name(uint32_t idx);
};

} // namespace DeliRium
