// =============================================================================
// WeldProfile.h - Extraido de Material.h para romper el ciclo de includes:
// Material.h necesita ConstitutiveModel.h (Material::EffectiveWeldProfile
// llama a WeldProfileFromCurve/WeldProfileFromFluid), y ConstitutiveModel.h
// necesita el TIPO WeldProfile (no WeldCategory ni Material) para el
// tipo de retorno de esas mismas funciones. Sin este header separado,
// Material.h <-> ConstitutiveModel.h se incluyen mutuamente y, segun el
// orden de inclusion, WeldProfile queda sin declarar (confirmado con un
// compilador real: 'WeldProfile' does not name a type).
// =============================================================================
#pragma once
namespace DeliRium {
struct WeldProfile {
    float relax;        // step fraction toward the partner midpoint
    float smooth;        // along-seam smoothing blend [0,1]
    int   iterations;   // relaxation passes
    float cap_frac;      // max step as fraction of nearest-partner distance
    float stitch;        // how much of the interior stitch to apply [0,1]
};
}
