// =============================================================================
// SurfaceTypeLibrary.cpp - the base-layer surface type basis.
// Each type is a simple named height field over the patch (u,v) domain. A
// patch's base shape is sum_i K_i * Eval(i, u, v). KEEP BIT-IDENTICAL to
// surface_type_eval() in kernels/psf_splat.cl.
// =============================================================================
#include "delirium/psf/SurfaceTypeLibrary.h"
#include <cmath>

namespace DeliRium {

float SurfaceTypeLibrary::Eval(uint32_t idx, float u, float v) {
    const float uc = u - 0.5f, vc = v - 0.5f;
    const float PI = 3.14159265358979323846f;
    switch (idx) {
        case 0:  return 1.0f;                              // plano (offset)
        case 1:  return uc;                                // tilt u
        case 2:  return vc;                                // tilt v
        case 3:  return 1.0f - (uc*uc + vc*vc) * 4.0f;     // cupula (dome)
        case 4:  return uc * vc * 4.0f;                    // silla (saddle)
        case 5:  return uc * uc * 4.0f;                    // valle u
        case 6:  return vc * vc * 4.0f;                    // valle v
        case 7:  return std::cos(2.0f*PI*u);               // onda u
        case 8:  return std::cos(2.0f*PI*v);               // onda v
        case 9:  return std::cos(2.0f*PI*u)*std::cos(2.0f*PI*v); // eggbox
        case 10: return std::cos(4.0f*PI*u);               // onda2 u
        case 11: return std::cos(4.0f*PI*v);               // onda2 v
        case 12: return std::exp(-(uc*uc + vc*vc) / 0.05f);// pico (peak)
        case 13: return std::sin(2.0f*PI*u) * vc;          // twist
        case 14: return uc * (vc*vc) * 4.0f;               // bend
        case 15: return (uc*uc - vc*vc) * 2.0f;            // monkey-ish
        default: return 0.0f;
    }
}

const char* SurfaceTypeLibrary::Name(uint32_t idx) {
    static const char* n[16] = {
        "plano","tilt_u","tilt_v","cupula","silla","valle_u","valle_v",
        "onda_u","onda_v","eggbox","onda2_u","onda2_v","pico","twist",
        "bend","monkey" };
    return idx < 16 ? n[idx] : "?";
}

} // namespace DeliRium
