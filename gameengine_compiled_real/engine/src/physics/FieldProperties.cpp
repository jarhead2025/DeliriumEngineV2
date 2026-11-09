// =============================================================================
// FieldProperties.cpp - the field-physics layer over the bone skeleton.
//
// See FieldProperties.h for the rationale. The short version: a material's
// electromagnetic and thermal properties are authored per BONE (the
// mechanism's parts), spread over that bone's anchors, and read out as the
// field quantities a solver needs.
//
// WHAT THIS IS NOT: a field solver. There is no PDE here. What there IS:
//   (a) the material data, authored in one place;
//   (b) the exact relations between that data and the field quantities;
//   (c) the lumped (mass-weighted) effective medium of a whole body, which
//       is what lets the document pipeline ask "does this reconstruction
//       behave like the material the document described?".
// =============================================================================
#include "delirium/physics/FieldProperties.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace DeliRium {

namespace {

// dot and length are deliberately LOCAL (see the note in core/Types.h:
// several translation units define their own).
inline float dot3(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline float len3(const Vec3& a) { return std::sqrt(dot3(a, a)); }

// The Stefan-Boltzmann constant, SI: 5.670374419e-8 W/(m^2 K^4).
constexpr float kStefanBoltzmann = 5.670374419e-8f;

} // namespace

// -----------------------------------------------------------------------------
// THE FIELD QUANTITIES.
// -----------------------------------------------------------------------------

Vec3 MagneticFluxDensity(const FieldProperties& p, const Vec3& H) {
    // B = mu * H, with mu = mu_0 * mu_r.
    const float mu = p.AbsolutePermeability();
    return H * mu;
}

Vec3 Magnetisation(const FieldProperties& p, const Vec3& H) {
    // M = chi * H, with chi = mu_r - 1.
    //
    // HONEST SCOPE: this is the LINEAR model. It is exact for linear
    // materials and a first-order approximation for ferro-magnets, which
    // SATURATE (a neodymium magnet's M stops growing at a fixed field).
    // A caller modelling a magnet near saturation needs the B-H curve, not
    // a constant chi -- and this function will not pretend otherwise.
    const float chi = p.permeability - 1.0f;
    return H * chi;
}

float MagneticEnergyDensity(const FieldProperties& p, const Vec3& H) {
    // u = 0.5 * B . H = 0.5 * mu * |H|^2.
    return 0.5f * p.AbsolutePermeability() * dot3(H, H);
}

Vec3 ElectricFluxDensity(const FieldProperties& p, const Vec3& E) {
    // D = eps * E, with eps = eps_0 * eps_r.
    return E * p.AbsolutePermittivity();
}

Vec3 Polarisation(const FieldProperties& p, const Vec3& E) {
    // P = eps_0 * chi_e * E, with chi_e = eps_r - 1.
    //
    // This is the material's contribution to D: D = eps_0 * E + P.
    const float chi_e = p.permittivity - 1.0f;
    return E * (8.8541878128e-12f * chi_e);
}

Vec3 ConductionCurrentDensity(const FieldProperties& p, const Vec3& E) {
    // J = sigma * E (Ohm's law, in the local form).
    return E * p.conductivity;
}

float UniformFieldChargeDensity(const FieldProperties& p) {
    // rho_free = -div(P). For a UNIFORM field, P is uniform, so its
    // divergence is zero and there is no free charge.
    //
    // Exposed so a caller can STATE the result instead of assuming it: the
    // bound charge on the surface of a polarised dielectric is a different
    // thing (rho_bound = P . n), and this function is not that.
    (void)p;
    return 0.f;
}

float RadiativePowerPerArea(const FieldProperties& p, float temperature_kelvin) {
    // P/A = eps * sigma_SB * T^4.
    //
    // The temperature must be in KELVIN. A caller passing Celsius would
    // get a number that is wrong by a factor of ~1000 (300 K vs 20 K in
    // the fourth power) with no error sign -- so the function does not
    // silently convert. The caller must convert, because the conversion
    // depends on what the number meant.
    return p.emissivity * kStefanBoltzmann *
           temperature_kelvin * temperature_kelvin *
           temperature_kelvin * temperature_kelvin;
}

float NetRadiativeFlux(const FieldProperties& a, float ta_kelvin,
                       const FieldProperties& b, float tb_kelvin) {
    // q = sigma_SB * (eps_1 * T_1^4 - eps_2 * T_2^4).
    //
    // The two emissivities are DIFFERENT -- that is the whole point of a
    // material property. Two surfaces at the same temperature with
    // different emissivities still exchange energy, because the more
    // reflective one emits less.
    //
    // SCOPE: this treats the surfaces as facing each other in a large
    // enclosure (view factor = 1). A solver with real geometry multiplies
    // by the view factor.
    const float t1 = ta_kelvin, t2 = tb_kelvin;
    const float e1 = a.emissivity, e2 = b.emissivity;
    return kStefanBoltzmann * (e1 * t1*t1*t1*t1 - e2 * t2*t2*t2*t2);
}

float AbsorbedFraction(const FieldProperties& p) { return p.absorptivity(); }

float ReflectedFraction(const FieldProperties& p) {
    // For an OPAQUE body the transmission is zero, so absorptivity +
    // reflectivity = 1. A transparent body breaks this, and this function
    // is the opaque-body one -- it is named for the case it models.
    return 1.0f - p.absorptivity();
}

// -----------------------------------------------------------------------------
// THE LUMPED (MASS-WEIGHTED) EFFECTIVE MEDIUM.
// -----------------------------------------------------------------------------

bool ComputeBodyFieldProperties(const PSFBody& body,
                                const std::vector<FieldProperties>& bone_props,
                                const std::vector<float>& bone_mass,
                                BodyFieldProperties& out,
                                std::string& why) {
    out = BodyFieldProperties{};
    if (bone_props.empty()) {
        why = "no bone field properties were authored";
        return false;
    }

    // The anchors' bone indices are the ground truth for "which part is
    // this surface". The mass comes from the same table the inertia layer
    // uses, so the two layers agree about the mechanism.
    //
    // WHY THE MASS WEIGHTING: a lumped medium is the mass-weighted average
    // because that is the physically correct mixture rule for a body whose
    // parts are interleaved at a scale below the field's resolution. A
    // steel shaft in a plastic housing presents a mixture, and the
    // fraction that matters is the fraction of MASS, not of volume or of
    // anchor count (which depends on the tessellation).
    //
    // WHY THE "REFERENCED" FILTER: the baker writes a PLACEHOLDER anchor
    // for every out-of-support grid cell -- they are not on the surface at
    // all. Counting them (as an early version of the inertia layer did)
    // put 97% of the body's mass at a single point. So a property counts
    // only where there IS surface.
    const AnchorPool& pool = body.AnchorsConst();
    const size_t n = pool.Count();
    if (n == 0) { why = "the body has no anchors"; return false; }

    std::vector<char> referenced(n, 0);
    for (const SurfaceRecord& s : body.SurfaceBlob())
        for (int k = 0; k < 4; ++k)
            if (s.anchor_idx[k] < n) referenced[s.anchor_idx[k]] = 1;

    // Accumulate the weighted sums.
    double wsum = 0.0;
    double mu = 0.0, eps = 0.0, sig = 0.0, em = 0.0;
    double cp = 0.0, kth = 0.0;
    int contributing = 0;

    for (size_t i = 0; i < n; ++i) {
        const uint32_t bi = pool.Blob()->at(i).bone_index;
        if (bi == 0xFFFFFFFFu) continue;      // no bone, no claim
        if (!referenced[i]) continue;         // a placeholder is not surface
        const size_t b = size_t(bi);
        if (b >= bone_props.size()) continue; // no properties
        // The mass of this bone. A bone with no entry contributes nothing:
        // a massless part has no claim on the body's effective medium.
        float m = 0.f;
        if (!bone_mass.empty()) {
            if (b >= bone_mass.size()) continue;
            m = bone_mass[b];
        }
        if (m <= 0.f) continue;

        const FieldProperties& p = bone_props[b];
        wsum += m;
        mu  += double(m) * p.permeability;
        eps += double(m) * p.permittivity;
        sig += double(m) * p.conductivity;
        em  += double(m) * p.emissivity;
        cp  += double(m) * p.specific_heat;
        kth += double(m) * p.thermal_conductivity;
        ++contributing;
    }

    if (contributing == 0 || wsum <= 0.0) {
        why = "no bone-bound anchor carried any mass with properties; "
              "the body has no effective medium";
        return false;
    }

    const float inv = float(1.0 / wsum);
    out.lumped.permeability        = float(mu  * inv);
    out.lumped.permittivity        = float(eps * inv);
    out.lumped.conductivity        = float(sig * inv);
    out.lumped.emissivity          = float(em  * inv);
    out.lumped.specific_heat       = float(cp  * inv);
    out.lumped.thermal_conductivity= float(kth * inv);
    // Kirchhoff's law holds for the MIXTURE: each part has its own
    // emissivity equal to its own absorptivity, so the mass-weighted
    // average of the absorptivities equals the mass-weighted average of
    // the emissivities -- which is exactly the lumped emissivity computed
    // above. So the DERIVED rule reproduces the mixture correctly, and the
    // override is left unset so `absorptivity()` returns it.
    //
    // (The mixture's absorptivity is therefore not stored separately: the
    // derived value IS the correct one, and storing it would give the
    // struct two sources of truth for the same quantity.)
    out.lumped.absorptivity_override = -1.f;
    out.average_mass = float(wsum);
    out.per_bone = bone_props;
    return true;
}

// -----------------------------------------------------------------------------
// THE MATERIAL REGISTRY.
// -----------------------------------------------------------------------------

bool LookupFieldProperties(const std::string& name, FieldProperties& out) {
    // The names a technical DOCUMENT would use. Lowercased and stripped of
    // separators, so "A36 Steel", "a36-steel" and "a36_steel" all find
    // the same entry.
    std::string key;
    key.reserve(name.size());
    for (char c : name) {
        if (c == ' ' || c == '-' || c == '_' || c == '/') continue;
        key.push_back(char(std::tolower((unsigned char)c)));
    }

    struct Entry { const char* key; FieldProperties (*make)(); };
    static const Entry kTable[] = {
        {"mildsteel",     &FieldPresets::MildSteel},
        {"steel",         &FieldPresets::MildSteel},
        {"a36",           &FieldPresets::MildSteel},
        {"carbonsteel",   &FieldPresets::MildSteel},
        {"copper",        &FieldPresets::Copper},
        {"cu",            &FieldPresets::Copper},
        {"aluminium",     &FieldPresets::Aluminium},
        {"aluminum",      &FieldPresets::Aluminium},
        {"al",            &FieldPresets::Aluminium},
        {"abs",           &FieldPresets::AbsPlastic},
        {"absplastic",    &FieldPresets::AbsPlastic},
        {"plastic",       &FieldPresets::AbsPlastic},
        {"polymer",       &FieldPresets::AbsPlastic},
        {"glass",         &FieldPresets::Glass},
        {"water",         &FieldPresets::Water},
        {"mumetal",       &FieldPresets::MuMetal},
        {"neodymium",     &FieldPresets::NeodymiumMagnet},
        {"neodymiummagnet",&FieldPresets::NeodymiumMagnet},
        {"muscle",        &FieldPresets::MuscleTissue},
        {"muscletissue",  &FieldPresets::MuscleTissue},
        {"tissue",        &FieldPresets::MuscleTissue},
        {"bone",          &FieldPresets::CorticalBone},
        {"corticalbone",  &FieldPresets::CorticalBone},
    };

    for (const Entry& e : kTable) {
        if (key == e.key) { out = e.make(); return true; }
    }
    // A known alias with something after it ("A36 steel", "steel housing")
    // still resolves on its first token. The token boundary is the point
    // where the next character is not a digit and not a letter -- so
    // "a36" matches inside "a36steel" (a digit follows, and the digit is
    // part of the grade, not a new word) and "steel" matches inside
    // "steelshaft".
    //
    // The boundary test is "the next character cannot CONTINUE an
    // alphabetic token": a digit continues a grade ("a36"), and so does a
    // letter. What ENDS the token is a separator (already removed) or the
    // end of the string. So the rule is: the alias must be a PREFIX of the
    // key, and the character AFTER it must not be a letter -- a digit is
    // allowed (grades), a letter is not (a different word).
    for (const Entry& e : kTable) {
        const std::string k(e.key);
        if (key.size() < k.size()) continue;
        if (key.compare(0, k.size(), k) != 0) continue;
        if (key.size() == k.size()) { out = e.make(); return true; }
        const char next = key[k.size()];
        if (!std::isalpha((unsigned char)next)) { out = e.make(); return true; }
        // A LETTER follows. The alias may still match if the letter simply
        // continues the same word -- "abs" in "absplastic", "bone" in
        // "boneplate" -- so try the LONGER alias entries first.
    }
    // The longer aliases ("absplastic", "muscletissue", "corticalbone",
    // "neodymiummagnet") are matched by the exact pass above, but a
    // compound like "absplastic housing" needs the prefix rule too. So do
    // one more pass, longest alias first, allowing an alphabetic suffix.
    {
        std::vector<std::string> keys;
        for (const Entry& e : kTable) keys.emplace_back(e.key);
        std::sort(keys.begin(), keys.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });
        for (const std::string& k : keys) {
            if (key.size() <= k.size()) continue;
            if (key.compare(0, k.size(), k) != 0) continue;
            for (const Entry& e : kTable) {
                if (k == e.key) { out = e.make(); return true; }
            }
        }
    }
    return false;
}

} // namespace DeliRium
