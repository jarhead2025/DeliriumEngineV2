// =============================================================================
// ModalDynamics.h - Reduced-order deformation: the far/cheap scale.
// Modes are precomputed OFFLINE by tools/psf_modes.py (assembled on the
// SAME patch basis, with seam coupling so modes are crack-consistent);
// the engine only integrates scalar amplitudes:
//     q_m'' + 2 zeta w_m q_m' + w_m^2 q_m = f_m
// and writes coefficients back as  coef = rest + sum_m q_m * shape_m.
// Runtime cost: n_modes scalar ODEs + one FMA sweep over coefficients.
// =============================================================================
#pragma once
#include <vector>
#include <string>
#include "../psf/PSFBody.h"
#include "Material.h"

namespace DeliRium {

class ModalDynamics {
public:
    // Load .modes (PSM v1) and snapshot the body's rest coefficients.
    // zeta starts at whatever tools/psf_modes.py baked into the file
    // (fixed ZETA=0.03 constant at export time) -- call SetDamping() or
    // SetDampingFromMaterial() afterward to override it live, without
    // needing to regenerate the .modes file per material.
    bool LoadModes(const std::string& path, const PSFBody& body);

    // Semi-implicit Euler on each modal oscillator.
    void Step(float dt);
    void Reset();   // zero all modal amplitudes (return to rest shape)

    // Generalized force on mode m from a scalar impulse along the surface
    // normal at (surface, u, v): f_m += Phi_m(s,u,v) * j, where
    // Phi_m = basis(u,v) . shape_m[s].   (world-point -> (s,u,v) mapping
    // arrives with PSFBody::ClosestPoint in a later milestone)
    void ApplyImpulseAt(uint32_t surface, float u, float v, float impulse);

    // Write rest + sum q_m shape_m into the body's surface records.
    void WriteBack(PSFBody& body) const;

    // direct amplitude access (tests / cinematic control)
    void  SetAmplitude(int m, float q, float q_dot = 0.0f);
    float Amplitude(int m) const { return q[m]; }
    float Frequency(int m) const { return freq[m]; }   // rad/s
    int   ModeCount() const { return (int)freq.size(); }
    float TotalEnergy() const;       // sum 1/2 (q_dot^2 + w^2 q^2)

    // --- NUEVO: conexion con el sistema de materiales real ----------------
    // Overrides the damping ratio baked into the loaded .modes file.
    // Same zeta applies uniformly to every mode (matches the ORIGINAL
    // single-scalar semantics already in Step()'s ODE -- this does not
    // change to per-mode damping, just makes the one scalar live-settable).
    void SetDamping(float z) { zeta = z; }
    float Damping() const { return zeta; }

    // Convenience: pulls zeta from the material's real stress-strain
    // curve (StressStrainCurve::DampingRatio -- brittle rings, ductile
    // deadens, verified physically ordered: Glass 0.0093 < Steel 0.1033
    // < Rubber 0.30). No-op (keeps whatever zeta already has, i.e. the
    // .modes file's baked value) if the material has no curve set --
    // same opt-in/backward-compatible pattern as
    // Material::EffectiveWeldProfile.
    void SetDampingFromMaterial(const Material& m) {
        if (m.stress_strain_curve)
            zeta = m.stress_strain_curve->DampingRatio();
    }

    // assembled-in-engine path: still offline-only by design
    void Precompute(const PSFBody&, const Material&, int) {} // see tools/psf_modes.py

private:
    std::vector<float> freq, q, q_dot, f_ext;     // per mode
    // v2 mode shapes: [n_modes][n_anchor + n_surf*8] -- anchors FIRST.
    // Anchor entries displace each anchor along its rest normal; this is
    // what creates global object modes (bending) vs per-patch flutter.
    std::vector<float> shapes;
    std::vector<float>    rest_coef;              // [n_surf*8]
    std::vector<Vec3>     rest_apos, rest_anrm;   // [n_anchor]
    std::vector<uint16_t> aidx;                   // [n_surf*4]
    float zeta = 0.0f;
    float amp_max = 0.0f;   // amplitude saturation (set from body size)
    uint32_t n_surf = 0, n_anchor = 0;
};

} // namespace DeliRium
