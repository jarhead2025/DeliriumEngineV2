// =============================================================================
// PatternLibrary.h - The presaved pattern dictionary (engine-side, zero
// file cost). Color AND displacement detail are sums  sum_i K_i * P_i(u,v).
// Indices+K live in the records; the functions live here and inside the
// kernels. CPU table must stay bit-compatible with pattern_eval() in
// kernels/psf_splat.cl.
// =============================================================================
#pragma once
#include <cstdint>

namespace DeliRium {

class PatternLibrary {
public:
    static constexpr int kCount = 16;
    static float Eval(uint32_t idx, float u, float v);          // TODO match .cl
    // Greedy OMP fit of n terms against a sampled target field.
    static void  Fit(const float* target, int w, int h,
                     int terms, uint8_t* idx_out, int8_t* k_out); // TODO
};

} // namespace DeliRium
