// =============================================================================
// ConstitutiveModel.h - Real stress-strain-driven material extension.
//
// ADDITIVE to Material.h: WeldCategory / WeldProfileFor / Material are left
// completely untouched (tests/validate_weldcat.cpp keeps testing exactly
// what it tested before). This file adds an OPT-IN, physically-grounded
// alternative: instead of picking one of 5 hand-tuned categories, derive
// the weld/relaxation behavior from a real digitized stress-strain curve
// (solids, via LocalFEM's tangent modulus) or a real fluid's viscosity
// (via the diffusion equation that IS what the Jacobi relaxation solves).
//
// =============================================================================
// PART 1 -- SOLIDS: real stress-strain curve driving FEM
// =============================================================================
// A stress-strain curve is a piecewise-linear table (strain, stress) pairs.
// This is the correct way to plug in a REAL diagram: digitize points off an
// actual datasheet/paper and hand them in, instead of trusting a single
// fixed Young's modulus for the whole loading range. LocalFEM's local
// stiffness should use the TANGENT modulus (local slope of the curve) at
// the region's current strain, not a constant E -- that is what makes the
// material genuinely nonlinear (elastic region stiff, yield plateau soft,
// hardening region stiff again), matching how real elastic-plastic FEM
// codes handle nonlinearity (tangent-stiffness Newton iteration).
//
// HONEST SCOPE: this table interpolates UNIAXIAL stress-strain response.
// It does not model multiaxial yield surfaces (von Mises, Tresca), rate
// dependence, or hysteresis/damage -- those are real, separate extensions
// on top of this if needed later.
// =============================================================================
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "WeldProfile.h"   // SOLO el tipo WeldProfile, sin ciclo

namespace DeliRium {

struct StressStrainPoint { float strain; float stress; };  // Pa, dimensionless

class StressStrainCurve {
public:
    // Points must be given in increasing strain order (not validated here
    // beyond a debug-time assert in real use -- keep authoring responsibility
    // explicit rather than silently re-sorting real digitized data).
    explicit StressStrainCurve(std::vector<StressStrainPoint> pts)
        : points_(std::move(pts)) {}

    // stress(strain), piecewise-linear interpolation; clamps outside range
    // to the boundary segment's slope (linear extrapolation) so the model
    // stays defined rather than discontinuous at the ends of the digitized
    // data.
    float Stress(float strain) const {
        if (points_.size() < 2) return points_.empty() ? 0.0f : points_[0].stress;
        if (strain <= points_.front().strain) {
            const auto& a = points_[0]; const auto& b = points_[1];
            const float slope = (b.stress-a.stress)/(b.strain-a.strain);
            return a.stress + slope*(strain - a.strain);
        }
        for (size_t i = 1; i < points_.size(); ++i) {
            if (strain <= points_[i].strain) {
                const auto& a = points_[i-1]; const auto& b = points_[i];
                const float t = (strain - a.strain)/(b.strain - a.strain);
                return a.stress + t*(b.stress - a.stress);
            }
        }
        const auto& a = points_[points_.size()-2]; const auto& b = points_.back();
        const float slope = (b.stress-a.stress)/(b.strain-a.strain);
        return b.stress + slope*(strain - b.strain);
    }

    // d(stress)/d(strain) at a given strain -- the TANGENT modulus that
    // should drive FEM local stiffness instead of a fixed E. Uses the
    // slope of whichever segment contains `strain` (or the boundary
    // segment's slope outside the digitized range).
    float TangentModulus(float strain) const {
        if (points_.size() < 2) return 0.0f;
        if (strain <= points_.front().strain) {
            const auto& a = points_[0]; const auto& b = points_[1];
            return (b.stress-a.stress)/(b.strain-a.strain);
        }
        for (size_t i = 1; i < points_.size(); ++i)
            if (strain <= points_[i].strain) {
                const auto& a = points_[i-1]; const auto& b = points_[i];
                return (b.stress-a.stress)/(b.strain-a.strain);
            }
        const auto& a = points_[points_.size()-2]; const auto& b = points_.back();
        return (b.stress-a.stress)/(b.strain-a.strain);
    }

    // Initial (zero-strain) modulus -- the classical "E" for small
    // deformations, used to seed a WeldProfile continuously (see below).
    float InitialModulus() const { return TangentModulus(0.0f); }

    // Ductility proxy: strain span of the digitized curve. A brittle
    // material's curve is short (fractures near the elastic limit); a
    // ductile one stretches far past yield. This is what should separate
    // "Rigid/Brittle-like" from "Soft/Fluid-like" weld behavior for solids,
    // replacing the hand-picked WeldCategory enum with a continuous,
    // curve-derived measurement.
    float StrainRange() const {
        return points_.empty() ? 0.0f : (points_.back().strain - points_.front().strain);
    }

    // Modal damping ratio (zeta), derived from the SAME curve: brittle
    // materials (tiny StrainRange, e.g. glass) ring for a long time when
    // struck -- low damping. Ductile materials (large StrainRange, e.g.
    // rubber) absorb vibration energy through plastic/hysteretic
    // deformation cycles and barely oscillate -- high damping. Verified
    // ordering (test_damping.cpp): Glass 0.0093 < MildSteel 0.1033 <
    // Rubber 0.30, all within the physically typical structural damping
    // range (~0.005-0.3). Reuses the identical log-scale ductility_frac
    // as WeldProfileFromCurve for consistency between the two material
    // effects (a material that welds "soft" also damps "heavily").
    float DampingRatio(float zeta_min = 0.005f, float zeta_max = 0.30f,
                       float ductility_ref = 0.20f) const {
        const float ductility = StrainRange();
        const float frac_raw = std::log1p(ductility) / std::log1p(ductility_ref);
        const float frac = std::fmin(frac_raw, 3.0f) / 3.0f;
        return zeta_min + (zeta_max - zeta_min) * frac;
    }

    const std::vector<StressStrainPoint>& Points() const { return points_; }

private:
    std::vector<StressStrainPoint> points_;
};

// --- Illustrative presets ---------------------------------------------------
// LABELED HONESTLY: these are textbook-typical digitizations (order-of-
// magnitude correct, standard engineering reference shapes), NOT measured
// lab data for a specific material sample. For a real simulation, digitize
// your own curve from a real datasheet/paper and build a StressStrainCurve
// directly -- that is the intended normal use of this class.
namespace MaterialPresets {

    // Structural mild steel (illustrative ASTM-A36-like shape): linear
    // elastic to yield ~250 MPa at strain ~0.00125 (E=200 GPa), a near-flat
    // plastic plateau (Luders strain) to ~2%, then modest strain hardening
    // to ~400 MPa at ~20% before the digitization stops (real steel
    // necks/fractures beyond this -- not modeled here).
    inline StressStrainCurve MildSteel() {
        return StressStrainCurve({
            {0.0f,        0.0f},
            {0.00125f,    250e6f},
            {0.02f,       255e6f},
            {0.20f,       400e6f},
        });
    }

    // Vulcanized rubber (illustrative): very low initial modulus, mild
    // S-shaped stiffening at large strain (real rubber is hyperelastic;
    // this piecewise-linear table approximates the same qualitative shape
    // -- soft start, stiffening finish -- without a full hyperelastic
    // strain-energy formulation).
    inline StressStrainCurve Rubber() {
        return StressStrainCurve({
            {0.0f,  0.0f},
            {0.5f,  1.0e6f},
            {2.0f,  3.0e6f},
            {4.0f,  15.0e6f},
        });
    }

    // Glass / ceramic (illustrative): linear elastic, no plastic plateau
    // at all -- fractures near the elastic limit. StrainRange() is tiny,
    // which is exactly what should map to Brittle-like weld behavior.
    inline StressStrainCurve Glass() {
        return StressStrainCurve({
            {0.0f,    0.0f},
            {0.008f,  560e6f},   // fractures here, no plastic region
        });
    }

} // namespace MaterialPresets

// =============================================================================
// PART 2 -- FLUIDS: the honest CFD bridge
// =============================================================================
// VERIFIED FINDING (see fluid_grounding_check.py, run before writing this):
// plugging a real fluid's MOLECULAR kinematic viscosity directly into the
// diffusion equation that the seam relaxation solver implements requires
// ~150,000 iterations for water to visually "melt" a seam at a 1-frame
// timescale and ~1 meter object scale. That is not a bug -- it is
// physically correct: real water barely mixes by molecular viscosity alone
// at these length/time scales; real-world mixing comes from turbulence
// (advection), not molecular diffusion. The existing WeldCategory::Fluid
// profile (relax=0.95, 64 iterations) is quantitatively consistent with a
// MOLECULAR viscosity around 2.3e-3 m^2/s (glycerin/thin-honey range), not
// water.
//
// So the physically honest way to get "water melts fast" gameplay behavior
// out of a real number is the same one real CFD uses for the same problem:
// EDDY / TURBULENT VISCOSITY (a real, standard concept -- e.g. Reynolds-
// averaged closures), which is molecular viscosity plus an "apparent"
// viscosity from unresolved turbulent mixing, and can legitimately be
// orders of magnitude larger than the molecular value.
// =============================================================================
struct FluidModel {
    float density            = 1000.0f;   // kg/m^3
    float dynamic_viscosity  = 1.0e-3f;    // Pa*s (mu), MOLECULAR value
    // Turbulence intensity in [0,1]: 0 = laminar (molecular viscosity only,
    // will be extremely slow to visually melt at gameplay scale -- honest
    // and correct), 1 = as turbulent as `eddy_ratio_max` allows.
    float turbulence_intensity = 0.0f;

    float KinematicViscosity() const { return dynamic_viscosity / density; }

    // Eddy (turbulent) viscosity, modeled as a MULTIPLE of the fluid's own
    // molecular viscosity: nu_t = turbulence_intensity * eddy_ratio_max *
    // nu_molecular. The eddy-to-molecular viscosity ratio is a real,
    // documented turbulence quantity (commonly called nu_t/nu); reported
    // values for real turbulent flows span roughly 10^2-10^5 depending on
    // flow intensity/scale -- eddy_ratio_max=1e4 here is a representative,
    // tunable ceiling, not a measured constant for any specific flow.
    //
    // FIX vs. an earlier version of this model: making nu_t proportional
    // to nu_molecular (instead of a pure geometry/dt term independent of
    // the fluid) keeps different real fluids DISTINGUISHABLE even under
    // strong turbulence -- water and honey no longer collapse to the same
    // effective viscosity just because both are "turbulent". This is the
    // physically correct expectation: in real turbulent flows, the eddy
    // viscosity of a more viscous fluid is still typically larger than
    // that of a less viscous one at comparable Reynolds regimes.
    float EddyViscosity(float eddy_ratio_max = 1.0e4f) const {
        return turbulence_intensity * eddy_ratio_max * KinematicViscosity();
    }

    float EffectiveViscosity(float eddy_ratio_max = 1.0e4f) const {
        return KinematicViscosity() * (1.0f + turbulence_intensity * eddy_ratio_max);
    }
};

// Named real fluids (molecular kinematic viscosity in m^2/s, from standard
// engineering tables -- these ARE real measured values, unlike the solid
// presets' illustrative curves).
namespace FluidPresets {
    inline FluidModel Water()     { return {998.0f,  1.002e-3f, 0.0f}; }   // 20C
    inline FluidModel OliveOil()  { return {920.0f,  0.081f,    0.0f}; }
    inline FluidModel Glycerin()  { return {1260.0f, 1.5f,      0.0f}; }
    inline FluidModel Honey()     { return {1420.0f, 8.5f,      0.0f}; }   // typical
    inline FluidModel Tar()       { return {1100.0f, 1.0e5f,    0.0f}; }   // illustrative, highly variable
} // namespace FluidPresets

// =============================================================================
// PART 3 -- Bridges to WeldProfile (the only thing LocalFEM/SeamGraph
// actually consume today). Both bridges are ADDITIVE: WeldProfileFor(...)
// from Material.h is untouched, existing categorized behavior (and
// validate_weldcat.cpp) is unaffected.
// =============================================================================

// Derive a WeldProfile from a real fluid's EFFECTIVE viscosity via the
// diffusion equation the Jacobi relaxation solver actually implements:
//   diffusion length after N steps of dt:  L = sqrt(2 * nu_eff * N * dt)
//   => iterations to settle a seam of length L_seam:
//        N = L_seam^2 / (2 * nu_eff * dt)
//   per-step diffusion number (this IS the "relax" step fraction), with
//   dx = L_seam / 20 as a nominal seam subdivision and the standard
//   explicit-scheme stability cap r <= 0.5:
//        r = clamp(nu_eff * dt / dx^2, 0, 0.5)
inline WeldProfile WeldProfileFromFluid(const FluidModel& fluid,
                                        float seam_length, float dt,
                                        int min_iters = 4, int max_iters = 256,
                                        float eddy_ratio_max = 1.0e4f) {
    const float nu = std::fmax(fluid.EffectiveViscosity(eddy_ratio_max), 1e-9f);
    const float dx = std::fmax(seam_length / 20.0f, 1e-9f);
    const float r  = std::fmin(nu * dt / (dx * dx), 0.5f);
    const float N_real = (seam_length*seam_length) / (2.0f * nu * dt);
    const int iters = (int)std::round(std::fmin(std::fmax(N_real, (float)min_iters),
                                                (float)max_iters));
    // stitch/cap/smooth still scale with how "fluid-like" this is (r near
    // its 0.5 stability ceiling = strongly diffusive = fully stitched,
    // matching the existing Fluid category's stitch=1.0, cap=1.0)
    const float fluidness = r / 0.5f;               // in [0,1]
    return WeldProfile{
        /*relax*/     r,
        /*smooth*/    0.05f + 0.20f*fluidness,
        /*iterations*/iters,
        /*cap_frac*/  0.20f + 0.80f*fluidness,
        /*stitch*/    0.15f + 0.85f*fluidness,
    };
}

// Derive a WeldProfile from a real stress-strain curve for solids: stiffer
// initial modulus -> lower relax/cap (holds crisp edges, like Rigid);
// larger ductility (StrainRange) -> higher smooth/stitch (melts together
// more, like Soft); a curve with almost no strain range (glass-like) ->
// near-Brittle behavior. `modulus_ref`/`ductility_ref` calibrate what
// counts as "very stiff" / "very ductile" for your asset scale (defaults
// are steel-vs-rubber order of magnitude).
inline WeldProfile WeldProfileFromCurve(const StressStrainCurve& curve,
                                        float modulus_ref = 200e9f,
                                        float ductility_ref = 0.20f) {
    const float E0 = std::fmax(curve.InitialModulus(), 1.0f);
    const float ductility = curve.StrainRange();
    // stiffness_frac: 1 = as stiff as modulus_ref (steel-like) -> rigid;
    // 0 = negligible initial stiffness -> soft start
    const float stiffness_frac = std::fmin(E0 / modulus_ref, 1.0f);
    // LOG-SCALE ductility measure: strain ranges span orders of magnitude
    // (glass ~0.008, steel ~0.20, rubber ~4.0+) -- a linear frac/ref clamp
    // saturates steel and rubber to the SAME value at ref=0.20, losing all
    // differentiation between them. log1p keeps them distinct across the
    // whole realistic range while still normalizing "around 1" at ref.
    const float ductility_frac_raw = std::log1p(ductility) / std::log1p(ductility_ref);
    const float ductility_frac = std::fmin(ductility_frac_raw, 3.0f);  // bound, not saturate-at-1
    // rigid materials relax little; ductile ones relax/smooth more
    const float relax  = std::fmin(0.20f + 0.30f*ductility_frac*(1.0f - 0.5f*stiffness_frac), 0.95f);
    const float smooth = std::fmin(0.02f + 0.12f*ductility_frac, 0.30f);
    const int   iters  = (int)std::fmin(10 + 25.0f*ductility_frac, 80.0f);
    const float cap    = std::fmin(0.15f + 0.35f*ductility_frac, 1.0f);
    const float stitch = ductility_frac < 0.05f ? 0.10f     // glass-like: barely stitches
                                                 : std::fmin(0.4f + 0.3f*ductility_frac, 1.0f);
    return WeldProfile{relax, smooth, iters, cap, stitch};
}

} // namespace DeliRium
// (verificacion suelta, no se agrega al archivo final todavia)
