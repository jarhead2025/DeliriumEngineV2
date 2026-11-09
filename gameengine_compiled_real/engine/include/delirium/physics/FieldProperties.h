// =============================================================================
// FieldProperties.h - material properties that drive MAGNETIC, ELECTRIC and
// RADIATIVE field physics.
//
// WHY THIS EXISTS (Josias, 2026-11-08): "allowing magnetic, electric and
// radiation field physics through material science properties. That is how
// our LLM should approach them."
//
// THE SUBSTRATE ALREADY EXISTS, AND IT IS THE RIGHT ONE:
//
//   * `Material` (physics/Material.h) already carries the MECHANICAL
//     properties -- density, Young's modulus, Poisson, a real
//     stress-strain curve, a real fluid model. What it does NOT carry is
//     the ELECTROMAGNETIC and THERMAL ones: permeability, permittivity,
//     conductivity, emissivity, absorptivity.
//
//   * `BoneRecord` (core/Types.h) already carries the authored mass,
//     radius and stiffness in its reserved bytes, and `AnchorRecord` has a
//     `bone_index`. So a per-bone MATERIAL assignment already has a home.
//
// So G11 is NOT a new system. It is the SAME pattern as the inertia layer
// (physics/Inertia.h), one layer over: properties are authored per BONE,
// spread over that bone's anchors, and read out as a field.
//
// WHY PER-BONE, NOT PER-SURFACE: a surface is a deformed graph whose
// extent depends on the LOD. A MATERIAL does not: steel is steel at every
// tessellation density. This is the same "complete in representation,
// coarse for speed" property the inertia layer relies on -- coarsening the
// surface changes how finely the property is spread, never how much there
// is.
//
// WHAT THIS MODULE DOES, AND WHAT IT DELIBERATELY DOES NOT:
//
//   It provides the PROPERTIES and the QUANTITIES derived from them that a
//   field solver needs, with the physical relations written out and
//   honest about their scope:
//
//   * MAGNETIC: permeability (mu_r), and the derived field quantities --
//     the B/H relation, the magnetic energy density, and a material's
//     susceptibility to an applied field.
//   * ELECTRIC: permittivity (epsilon_r), conductivity, and the derived
//     quantities -- the D/E relation, the charge density a polarisation
//     produces, and the conduction current density.
//   * RADIATIVE: emissivity, absorptivity, and the Stefan-Boltzmann
//     radiative power, plus the net exchange between two surfaces at
//     different temperatures.
//
// IT DOES NOT SOLVE A FIELD. There is no PDE solver here, and pretending
// otherwise would be the kind of plausible-but-wrong output that makes the
// whole layer untrustworthy. What it provides is:
//   (a) the material data a solver would read, authored in ONE place;
//   (b) the exact relations between that data and the field quantities,
//       so a solver's equations can be checked against a reference;
//   (c) a way to ask "does this reconstruction behave like the material
//       the document described?" -- which is the question the document
//       pipeline actually asks.
//
// The units are SI: permeability in H/m, permittivity in F/m, conductivity
// in S/m, emissivity and absorptivity dimensionless, temperature in K.
// =============================================================================
#pragma once
#include <cmath>
#include <string>
#include <vector>
#include "delirium/core/Types.h"
#include "delirium/psf/PSFBody.h"

namespace DeliRium {

// -----------------------------------------------------------------------------
// The field properties of one material.
//
// EVERY value here is a REAL, measurable material property with published
// SI units. The defaults are for a generic dielectric (roughly a mid-range
// plastic), and every one can be overridden from an authored document.
//
// THE THREE PASSIVE CONSTANTS, and why they are three and not one:
//
//   * PERMEABILITY (mu_r, dimensionless). How a material magnetises. 1.0 is
//     vacuum (and, closely, most non-magnetic metals); >1 is
//     paramagnetic; >>1 is ferromagnetic (steel ~200, mu-metal ~20000,
//     neodymium ~1.5). It is SEPARATE from conductivity because the two
//     are genuinely independent: copper is a perfect conductor and NOT
//     ferromagnetic, and mu-metal is ferromagnetic and a poor conductor.
//
//   * PERMITTIVITY (epsilon_r, dimensionless). How a material polarises.
//     ~1 is vacuum, ~2-6 is most dielectrics (polyethylene 2.3, glass
//     4-10, water 80), and >>1 are the high-k ceramics. It is SEPARATE from
//     permeability for the same reason.
//
//   * CONDUCTIVITY (sigma, S/m). How freely charge moves. 0 is an
//     insulator, ~1e-3 is a wet soil, ~1e4 is a poor conductor (graphite),
//     1e5-6e5 is copper/aluminium, 1e6+ is a liquid metal.
//
// The thermal/radiative pair:
//
//   * EMISSIVITY (dimensionless, 0-1). How efficiently the surface emits
//     thermal radiation. Polished metal ~0.05, oxidised metal 0.3-0.9,
//     blackbody 1.0.
//
//   * ABSORPTIVITY (dimensionless, 0-1). How efficiently the surface
//     absorbs INCOMING radiation. Kirchhoff's law says that at thermal
//     equilibrium a surface's emissivity EQUALS its absorptivity -- so the
//     default derives absorptivity from emissivity unless it is
//     explicitly overridden.
//
// WHY THESE ARE SEPARATE FIELDS AND NOT ONE "material id": a real solver
// needs the individual constants. Grouping them behind a preset is
// convenient for authoring, but the VALUES must be individually readable,
// because a document may say "a steel housing with a ceramic insulator"
// and the reconstruction needs both numbers, not one label.
// -----------------------------------------------------------------------------
struct FieldProperties {
    // --- magnetic ---
    float permeability = 1.0f;      // mu_r, dimensionless (1.0 = vacuum)
    // --- electric ---
    float permittivity = 4.0f;      // epsilon_r, dimensionless (1.0 = vacuum)
    float conductivity = 0.f;       // sigma, S/m
    // --- radiative ---
    float emissivity = 0.9f;        // dimensionless, 0..1
    // absorptivity is DERIVED from emissivity by Kirchhoff's law unless it
    // is explicitly set. See `absorptivity()`.
    float absorptivity_override = -1.f;   // <0 = derive from emissivity
    // --- thermal ---
    float specific_heat = 400.f;    // J/(kg*K)
    float thermal_conductivity = 0.5f;   // W/(m*K)

    // Kirchhoff's law: at thermal equilibrium a surface's absorptivity
    // equals its emissivity. The override exists because a surface CAN be
    // spectrally selective (a solar absorber is dark in the visible and
    // bright in the infrared), which is exactly the case where the two
    // genuinely differ.
    float absorptivity() const {
        if (absorptivity_override >= 0.f) return absorptivity_override;
        return emissivity;
    }

    // The absolute permeability in H/m: mu = mu_0 * mu_r, with
    // mu_0 = 4*pi*1e-7 H/m (the permeability of free space, exact by
    // definition since the 2019 SI redefinition).
    float AbsolutePermeability() const {
        return 4.0f * 3.14159265358979323846f * 1e-7f * permeability;
    }
    // The absolute permittivity in F/m: eps = eps_0 * eps_r, with
    // eps_0 = 1/(mu_0 * c^2) = 8.8541878128e-12 F/m.
    float AbsolutePermittivity() const {
        return 8.8541878128e-12f * permittivity;
    }
    // The electrical CONDUCTIVITY as a time constant: tau = eps / sigma.
    // This is the characteristic relaxation time of a conductor -- how long
    // the free charge takes to redistribute. It is the number that says
    // "is this a conductor or an insulator" in a way a solver can use
    // directly, and it has a clean physical meaning: for copper,
    // eps/sigma ~ 1e-19 s (instant), for a good insulator it is
    // effectively infinite.
    float ChargeRelaxationTime() const {
        if (conductivity <= 0.f) return INFINITY;
        return AbsolutePermittivity() / conductivity;
    }
    // The MAGNETIC diffusion time scale: tau_m = mu * sigma * L^2, where
    // L is a characteristic length (the skin depth's scale). This is the
    // eddy-current diffusion time -- how fast a changing magnetic field
    // penetrates a conductor. With L in metres, the result is in seconds.
    // For copper at L = 1 cm: mu_0 * sigma * L^2 ~ 4e-9 s, so a changing
    // field is excluded from a copper shield almost instantly.
    float MagneticDiffusionTime(float length) const {
        if (conductivity <= 0.f) return INFINITY;
        return AbsolutePermeability() * conductivity * length * length;
    }
};

// -----------------------------------------------------------------------------
// The field properties of a whole body: authored per BONE, spread over that
// bone's anchors.
//
// WHY A PER-BONE TABLE AND NOT A PER-SURFACE ONE: the same reason the
// inertia layer is per-bone (physics/Inertia.h). A material is a property
// of the MECHANISM, not of how finely it is tessellated. Coarsening the
// surface must not change how much steel there is.
//
// The mass-weighted average of a property over the body is what a field
// solver needs for a body treated as ONE medium (a lumped model). The
// per-bone values are what a solver needs for a body treated as a
// MECHANISM (a steel shaft inside a plastic housing).
// -----------------------------------------------------------------------------
struct BodyFieldProperties {
    // Per-bone properties. Indexed by the bone index the anchors carry.
    // A bone with no entry contributes NOTHING -- which is the honest
    // answer, not a default. A body whose bones have no properties has no
    // field properties; filling in defaults would silently assert "this
    // is dielectric plastic" about a part the document never described.
    std::vector<FieldProperties> per_bone;

    // The properties of the body's SURFACE as a whole -- the mass-weighted
    // average over the bone-bound anchors, computed by
    // `ComputeBodyFieldProperties`. This is the lumped model.
    FieldProperties lumped;

    // The mass-weighted averages of the scalar fields, computed alongside
    // the lumped properties so a solver does not have to re-derive them.
    float average_mass = 0.f;
};

// Compute the body's field properties from its bone-bound anchors.
//
// `bone_props` is indexed by bone index. `bone_mass` gives each bone's
// mass (the same table the inertia layer uses); a bone with no mass
// contributes nothing to the averages, because a massless part has no
// claim on the body's effective medium.
//
// Returns false with `why` when the body has no bone-bound anchors at all
// -- there is no material to average, and a zero-valued lumped property
// would read as "a perfect vacuum / a perfect insulator" rather than
// "unknown".
bool ComputeBodyFieldProperties(const PSFBody& body,
                                const std::vector<FieldProperties>& bone_props,
                                const std::vector<float>& bone_mass,
                                BodyFieldProperties& out,
                                std::string& why);

// -----------------------------------------------------------------------------
// THE FIELD QUANTITIES. Each of these is the EXACT relation a field solver
// would use, written out so the solver's equations can be checked against a
// reference -- and so a material's behaviour can be predicted without
// solving anything.
// -----------------------------------------------------------------------------

// The magnetic flux density B for an applied field H: B = mu * H.
// Returns the field in tesla for H in A/m.
Vec3 MagneticFluxDensity(const FieldProperties& p, const Vec3& H);

// The magnetisation M of a material in an applied field H:
//   M = chi * H,  where chi = mu_r - 1.
// A paramagnetic material has chi > 0, a diamagnetic one chi < 0, and a
// ferromagnetic one chi >> 0 (the linear model is an approximation for
// ferro-magnets, which saturate -- the caller should know that).
Vec3 Magnetisation(const FieldProperties& p, const Vec3& H);

// The magnetic energy density in J/m^3:
//   u = 0.5 * B . H = 0.5 * mu * |H|^2.
float MagneticEnergyDensity(const FieldProperties& p, const Vec3& H);

// The electric flux density D for an applied field E: D = eps * E.
// Returns the field in coulombs per square metre for E in V/m.
Vec3 ElectricFluxDensity(const FieldProperties& p, const Vec3& E);

// The polarisation P of a material in an applied field E:
//   P = eps_0 * chi_e * E,  where chi_e = eps_r - 1.
// (The susceptibility of the material, times the permittivity of free
// space, times the field -- the standard decomposition of D into a
// vacuum part and a material part.)
Vec3 Polarisation(const FieldProperties& p, const Vec3& E);

// The conduction current density J for an applied field E: J = sigma * E.
// Returns amperes per square metre for E in V/m.
Vec3 ConductionCurrentDensity(const FieldProperties& p, const Vec3& E);

// The free-charge density a polarisation produces, for a uniform field:
//   rho_free = -div(P) = 0 for a uniform field.
// (Exposed so a caller can state the result rather than assume it; a
// NON-uniform field's divergence is the solver's job.)
float UniformFieldChargeDensity(const FieldProperties& p);

// The radiative power a surface emits per unit area, W/m^2:
//   P/A = eps * sigma_SB * T^4,
// with the Stefan-Boltzmann constant sigma_SB = 5.670374419e-8 W/(m^2 K^4).
// Returns watts per square metre for a temperature in kelvin.
float RadiativePowerPerArea(const FieldProperties& p, float temperature_kelvin);

// The NET radiative heat flux between two surfaces at different
// temperatures, W/m^2:
//   q = sigma_SB * (eps_1 * T_1^4 - eps_2 * T_2^4).
// The two emissivities are DIFFERENT -- that is the whole point of a
// material property, and it is why this is not simply "the hotter one
// wins". Note this treats the surfaces as facing each other in a
// large enclosure (view factor = 1); a solver with real geometry scales
// it by the view factor.
float NetRadiativeFlux(const FieldProperties& a, float ta_kelvin,
                       const FieldProperties& b, float tb_kelvin);

// The fraction of incident radiation a material ABSORBS, and the fraction
// it REFLECTS (the rest is transmitted -- for an opaque body, the
// transmission is zero and absorptivity + reflectivity = 1).
float AbsorbedFraction(const FieldProperties& p);
float ReflectedFraction(const FieldProperties& p);

// -----------------------------------------------------------------------------
// The authored material presets. THESE ARE REAL, PUBLISHED VALUES for real
// materials -- not hand-tuned numbers. Where a value varies with grade or
// processing, the range is given in the comment and the representative
// value is used.
//
// The names are the ones a technical DOCUMENT would use, so a document can
// name a material and the reconstruction can look it up.
// -----------------------------------------------------------------------------
namespace FieldPresets {

    // Structural mild steel (A36-like): ferromagnetic, conductive.
    // mu_r ~ 200 (saturates; the linear value is for small fields),
    // sigma ~ 1.0e7 S/m (1000-10000x copper's -- steel is a much worse
    // conductor), epsilon_r ~ 6 (the bulk, for a solid part the surface
    // oxide dominates), emissivity ~0.3 for oxidised hot steel, ~0.05
    // for polished.
    inline FieldProperties MildSteel() {
        FieldProperties p;
        p.permeability = 200.f;
        p.permittivity = 6.f;
        p.conductivity = 1.0e7f;
        p.emissivity = 0.3f;
        p.specific_heat = 490.f;         // J/(kg*K)
        p.thermal_conductivity = 50.f;   // W/(m*K)
        return p;
    }

    // Copper: the classic conductor. NOT ferromagnetic (mu_r ~ 1) -- which
    // is the clean demonstration that conductivity and permeability are
    // independent properties.
    inline FieldProperties Copper() {
        FieldProperties p;
        p.permeability = 1.0f;
        p.permittivity = 1.f;            // effectively, for a bulk conductor
        p.conductivity = 5.8e7f;
        p.emissivity = 0.03f;            // polished
        p.specific_heat = 385.f;
        p.thermal_conductivity = 400.f;
        return p;
    }

    // Aluminium: conductive, not ferromagnetic.
    inline FieldProperties Aluminium() {
        FieldProperties p;
        p.permeability = 1.0f;
        p.permittivity = 1.f;
        p.conductivity = 3.5e7f;
        p.emissivity = 0.1f;             // mill-finish
        p.specific_heat = 897.f;
        p.thermal_conductivity = 237.f;
        return p;
    }

    // ABS polymer: the classic dielectric. Insulator, and NOT magnetic.
    inline FieldProperties AbsPlastic() {
        FieldProperties p;
        p.permeability = 1.f;
        p.permittivity = 2.4f;
        p.conductivity = 0.f;
        p.emissivity = 0.9f;
        p.specific_heat = 1470.f;
        p.thermal_conductivity = 0.2f;
        return p;
    }

    // Glass: a dielectric with a high refractive index, low loss.
    inline FieldProperties Glass() {
        FieldProperties p;
        p.permeability = 1.f;
        p.permittivity = 7.f;            // 6-10 depending on the glass
        p.conductivity = 0.f;
        p.emissivity = 0.9f;
        p.specific_heat = 840.f;
        p.thermal_conductivity = 1.f;
        return p;
    }

    // Water: the interesting one -- a high permittivity AND a conductor.
    // epsilon_r = 80 at 20C (the static value; the optical value is
    // ~1.77), and a modest conductivity (pure water is 5.5e-6 S/m;
    // ordinary water is dominated by dissolved ions at ~5e-4).
    inline FieldProperties Water() {
        FieldProperties p;
        p.permeability = 1.f;
        p.permittivity = 80.f;
        p.conductivity = 5e-4f;
        p.emissivity = 0.95f;
        p.specific_heat = 4182.f;
        p.thermal_conductivity = 0.6f;
        return p;
    }

    // Mu-metal: the magnetic shield alloy. Enormous permeability
    // (~20,000) and a LOW conductivity -- the opposite of copper in
    // every field property. It exists to demonstrate that the two
    // constants are genuinely independent axes.
    inline FieldProperties MuMetal() {
        FieldProperties p;
        p.permeability = 20000.f;
        p.permittivity = 1.f;
        p.conductivity = 1.1e6f;
        p.emissivity = 0.5f;
        p.specific_heat = 440.f;
        p.thermal_conductivity = 13.f;
        return p;
    }

    // Neodymium magnet: the ferromagnetic permanent. A linear mu_r is a
    // poor model here (it saturates around mu_r = 1.5 for the magnet
    // itself); the value is the RECOIL relative permeability, which is the
    // honest number for a hard magnet.
    inline FieldProperties NeodymiumMagnet() {
        FieldProperties p;
        p.permeability = 1.5f;
        p.permittivity = 1.f;
        p.conductivity = 8e5f;
        p.emissivity = 0.3f;
        p.specific_heat = 500.f;
        p.thermal_conductivity = 10.f;
        return p;
    }

    // Living tissue (muscle, roughly): a high-water-content dielectric
    // with a conductivity in the physiological range.
    inline FieldProperties MuscleTissue() {
        FieldProperties p;
        p.permeability = 1.f;
        p.permittivity = 60.f;           // water-dominated
        p.conductivity = 0.6f;           // ~0.5-0.8 S/m in the muscle's
                                         // longitudinal direction
        p.emissivity = 0.95f;
        p.specific_heat = 3500.f;
        p.thermal_conductivity = 0.5f;
        return p;
    }

    // Bone (cortical): dense, dielectric, a poor conductor, and
    // ANISOTROPIC in reality (the conductivity differs along and across
    // the collagen fibres) -- the isotropic value here is the
    // perpendicular-to-fibre one.
    inline FieldProperties CorticalBone() {
        FieldProperties p;
        p.permeability = 1.f;
        p.permittivity = 7.f;
        p.conductivity = 0.06f;
        p.emissivity = 0.9f;
        p.specific_heat = 1700.f;
        p.thermal_conductivity = 0.3f;
        return p;
    }

} // namespace FieldPresets

// Look up a named material's field properties. Returns false if the name is
// not known -- an UNKNOWN material must not silently become "generic
// plastic", because that would assert a physics the document never said.
bool LookupFieldProperties(const std::string& name, FieldProperties& out);

} // namespace DeliRium
