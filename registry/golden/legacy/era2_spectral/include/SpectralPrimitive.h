// =============================================================================
// SpectralPrimitive.h - "modelos complejos con ecuaciones simples".
//
// The user's vision, made concrete: a complex model is a SIMPLE primitive (a
// sphere: center + base radius) plus a baked DETAIL FIELD expressed as a
// spherical-harmonic expansion -- the 3D analogue of baking detail into a 2D
// texture, but the detail is integral geometry, stored as a few hundred
// coefficients instead of thousands of anchors/triangles.
//
//   surface(theta,phi) = center + dir(theta,phi) * (base_r + Rgeom(theta,phi))
//   color(theta,phi)   = (Rr, Rg, Rb)(theta,phi)
//
// where R*(theta,phi) = sum_{l,m} c_lm * Y_lm(theta,phi) are real spherical
// harmonic fields. No anchors, no per-patch coefficients, no seams (a single
// continuous field over the whole primitive). Detail lives in the EQUATION
// (the coefficients), not in the COUNT of elements -- so a 6 KB file carries
// what a 460 KB patch model (or a multi-MB mesh) carried, at equal fidelity.
//
// Complex / non-star-shaped objects are COMPOSITIONS of these primitives
// (the user's "objetos individuales que conforman algo mas grande"): a
// character is ~15 capsule-primitives, each with its own baked field; physics
// moves the primitives and the detail rides along.
// =============================================================================
#pragma once
#include "delirium/core/Types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace DeliRium {

class SpectralPrimitive {
public:
    bool Load(const std::string& path);
    bool LoadFromMemory(const uint8_t* data, size_t n);

    // Evaluate the surface point for a direction given by spherical angles
    // (theta in [0,pi] polar, phi in [-pi,pi] azimuth).
    Vec3 Surface(float theta, float phi) const;
    // Evaluate the baked color field at the same direction.
    Vec3 Color(float theta, float phi) const;
    // Analytic-ish normal via finite difference of the field (cheap, exact
    // enough; the field is smooth).
    Vec3 Normal(float theta, float phi) const;

    Vec3  Center()    const { return center_; }
    float BaseRadius()const { return base_r_; }
    uint32_t GeomOrder() const { return Lg_; }
    uint32_t ColorOrder()const { return Lc_; }
    size_t  ByteSize() const { return bytes_; }
    // Raw coefficient pointers for GPU upload (geom (Lg+1)^2, color
    // 3*(Lc+1)^2 channel-major). Stable for the lifetime of the object.
    const float* GeomCoeffs()  const { return gcoef_.data(); }
    const float* ColorCoeffs() const { return ccoef_.data(); }

private:
    Vec3 center_{};
    float base_r_ = 1.0f;
    uint32_t Lg_ = 0, Lc_ = 0;
    std::vector<float> gcoef_;             // (Lg+1)^2
    std::vector<float> ccoef_;             // 3*(Lc+1)^2 (RGB interleaved by ch)
    size_t bytes_ = 0;

    // real spherical harmonics evaluated up to order L into `out`
    static void EvalSH(int L, float theta, float phi, std::vector<float>& out);
    float Field(const std::vector<float>& coef, int L,
                float theta, float phi) const;
};

} // namespace DeliRium
