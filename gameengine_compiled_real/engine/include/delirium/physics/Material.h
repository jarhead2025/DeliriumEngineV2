#pragma once
#include "WeldProfile.h"         // <- NUEVO (rompe el ciclo)
#include "ConstitutiveModel.h"   // <- NUEVO
#include <memory>                // <- NUEVO

namespace DeliRium {

enum class WeldCategory { Rigid, Elastic, Soft, Fluid, Brittle };

// WeldProfile struct ahora vive en WeldProfile.h (incluido arriba) --
// SIN CAMBIOS de contenido, solo de ubicacion.

inline WeldProfile WeldProfileFor(WeldCategory c) {
    switch (c) {
        case WeldCategory::Rigid:   return { 0.35f, 0.05f, 24, 0.40f, 0.85f };
        case WeldCategory::Elastic: return { 0.60f, 0.10f, 32, 0.55f, 1.00f };
        case WeldCategory::Soft:    return { 0.80f, 0.18f, 44, 0.80f, 1.00f };
        case WeldCategory::Fluid:   return { 0.95f, 0.25f, 64, 1.00f, 1.00f };
        case WeldCategory::Brittle: return { 0.15f, 0.00f, 10, 0.18f, 0.15f };
    }
    return { 0.60f, 0.10f, 32, 0.55f, 1.00f };
}

struct Material {
    float density        = 1000.0f;
    float youngs_modulus = 1e6f;
    float poisson        = 0.45f;
    float damping        = 0.02f;
    float restitution    = 0.3f;
    float friction       = 0.6f;
    WeldCategory weld    = WeldCategory::Elastic;

    std::shared_ptr<StressStrainCurve> stress_strain_curve;
    std::shared_ptr<FluidModel>        fluid_model;
    float reference_seam_length = 0.05f;

    WeldProfile EffectiveWeldProfile(float diag, float dt) const {
        if (fluid_model)
            return WeldProfileFromFluid(*fluid_model,
                                        reference_seam_length*diag, dt);
        if (stress_strain_curve)
            return WeldProfileFromCurve(*stress_strain_curve);
        return WeldProfileFor(weld);
    }
};

} // namespace DeliRium
