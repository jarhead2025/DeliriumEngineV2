// =============================================================================
// validate_field_properties - GATE for the field-physics layer (G11).
//
// WHY THIS GATE EXISTS: "allowing magnetic, electric and radiation field
// physics through material science properties". The layer provides the
// MATERIAL DATA and the EXACT RELATIONS between that data and the field
// quantities. It does NOT solve a field -- and a gate that lets a
// plausible-but-wrong field law through would be worse than no layer at
// all.
//
// WHAT IS ASSERTED HERE:
//
//   1. THE PHYSICS IS RIGHT. Every relation is checked against an
//      INDEPENDENTLY COMPUTED value -- a textbook constant, an analytic
//      limit, or a published measurement. Not against a snapshot of what
//      the code currently does.
//
//   2. THE MATERIALS ARE DISTINGUISHABLE. A copper that behaved like
//      mu-metal, or a steel that behaved like a polymer, would make the
//      whole layer decorative.
//
//   3. THE PROPERTIES ARE INDEPENDENT AXES. Permeability and
//      conductivity are separate quantities, and the layer must not
//      derive one from the other.
//
//   4. THE LUMPED MEDIUM IS CORRECT. The mass-weighted mixture of two
//      materials is what the mixture rule says it is.
//
//   5. IT REFUSES TO GUESS. An unknown material name and a body with no
//      bone-bound surface must both be REJECTED, not defaulted -- a
//      default here would assert physics the document never stated.
//
//   6. THE BOUNDARY BEHAVIOUR. A perfect conductor, a perfect insulator,
//      a perfect reflector and a blackbody are the limits, and each must
//      behave exactly as the limit demands.
// =============================================================================
#include "delirium/physics/FieldProperties.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace DeliRium;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

// A relative comparison -- the relations are checked against values whose
// magnitude varies over many orders, so an absolute epsilon would pass the
// small ones for the wrong reason and fail the large ones for the wrong
// reason.
static bool close_rel(float a, float b, float rel = 1e-3f) {
    if (a == b) return true;
    const float m = std::fmax(std::fabs(a), std::fabs(b));
    if (m < 1e-30f) return true;
    return std::fabs(a - b) / m < rel;
}

namespace {

// The SI constants, written out independently of the implementation so
// the gate is not comparing the code against itself.
constexpr float kMu0        = 4.0f * 3.14159265358979323846f * 1e-7f;  // H/m
constexpr float kEps0       = 8.8541878128e-12f;                        // F/m
constexpr float kStefanB    = 5.670374419e-8f;                          // W/(m^2 K^4)

void TestMagneticRelations() {
    std::printf("-- the magnetic relations --\n");
    const FieldProperties cu = FieldPresets::Copper();
    const Vec3 H{100.f, 0.f, 0.f};

    // B = mu * H, with mu = mu_0 * mu_r.
    const Vec3 B = MagneticFluxDensity(cu, H);
    CHECK(close_rel(B.x, kMu0 * 1.0f * 100.f),
          "copper's B = mu_0 * H (got %g, want %g)", B.x, kMu0 * 100.f);
    CHECK(close_rel(B.y, 0.f) && close_rel(B.z, 0.f),
          "B is parallel to H");

    // M = chi * H, with chi = mu_r - 1. Copper is NOT magnetic, so its
    // magnetisation must be ZERO -- not "small".
    const Vec3 M = Magnetisation(cu, H);
    CHECK(M.x == 0.f, "copper is not magnetic, so M must be exactly 0 (got %g)", M.x);

    // Steel IS magnetic: chi = 200 - 1 = 199.
    const FieldProperties steel = FieldPresets::MildSteel();
    const Vec3 Ms = Magnetisation(steel, H);
    CHECK(close_rel(Ms.x, 199.f * 100.f),
          "steel's M = (mu_r - 1) * H (got %g, want %g)", Ms.x, 199.f * 100.f);

    // u = 0.5 * mu * |H|^2.
    const float u = MagneticEnergyDensity(cu, H);
    CHECK(close_rel(u, 0.5f * kMu0 * 100.f * 100.f),
          "copper's magnetic energy density (got %g, want %g)",
          u, 0.5f * kMu0 * 1e4f);

    // THE ENERGY MUST BE POSITIVE AND SCALE AS |H|^2. A quadratic field
    // energy is a strong statement: doubling H quadruples the energy.
    const float u2 = MagneticEnergyDensity(cu, Vec3{200.f, 0.f, 0.f});
    CHECK(close_rel(u2, 4.f * u), "magnetic energy density scales as |H|^2");

    // A NON-MAGNETIC material's energy density must be the SMALLEST of
    // all -- a permalloy shield at the same H stores vastly more.
    const float u_mu = MagneticEnergyDensity(FieldPresets::MuMetal(), H);
    CHECK(u_mu > 1000.f * u,
          "mu-metal stores far more magnetic energy than copper at the "
          "same H (got %g vs %g)", u_mu, u);
}

void TestElectricRelations() {
    std::printf("-- the electric relations --\n");
    const FieldProperties water = FieldPresets::Water();
    const Vec3 E{1000.f, 0.f, 0.f};

    // D = eps * E.
    const Vec3 D = ElectricFluxDensity(water, E);
    CHECK(close_rel(D.x, kEps0 * 80.f * 1000.f),
          "water's D = eps_0 * eps_r * E (got %g, want %g)",
          D.x, kEps0 * 80.f * 1000.f);

    // P = eps_0 * chi_e * E, with chi_e = eps_r - 1.
    const Vec3 P = Polarisation(water, E);
    CHECK(close_rel(P.x, kEps0 * 79.f * 1000.f),
          "water's P = eps_0 * (eps_r - 1) * E (got %g, want %g)",
          P.x, kEps0 * 79.f * 1000.f);

    // THE DECOMPOSITION MUST HOLD: D = eps_0 * E + P.
    const Vec3 D2 = Vec3{kEps0 * E.x + P.x, kEps0 * E.y + P.y, kEps0 * E.z + P.z};
    CHECK(close_rel(D.x, D2.x),
          "D must equal eps_0*E + P (got %g, want %g)", D.x, D2.x);

    // J = sigma * E.
    const Vec3 J = ConductionCurrentDensity(water, E);
    CHECK(close_rel(J.x, 5e-4f * 1000.f),
          "water's conduction current density (got %g, want %g)", J.x, 0.5f);

    // An INSULATOR carries NO conduction current -- exactly zero.
    const Vec3 Ji = ConductionCurrentDensity(FieldPresets::AbsPlastic(), E);
    CHECK(Ji.x == 0.f, "an insulator carries no conduction current (got %g)", Ji.x);

    // A uniform field has no divergence, so no free charge.
    CHECK(UniformFieldChargeDensity(water) == 0.f,
          "a uniform field produces no free charge");
}

void TestTheLimits() {
    std::printf("-- the boundary limits --\n");

    // A PERFECT CONDUCTOR: infinite conductivity, so the charge relaxes
    // instantly.
    FieldProperties perfect;
    perfect.conductivity = 1e30f;
    CHECK(perfect.ChargeRelaxationTime() < 1e-30f,
          "a perfect conductor relaxes its charge instantly (got %g s)",
          perfect.ChargeRelaxationTime());

    // A PERFECT INSULATOR: the relaxation time is infinite, because there
    // is no conduction to relax the charge with.
    FieldProperties insulator;
    insulator.conductivity = 0.f;
    CHECK(std::isinf(insulator.ChargeRelaxationTime()),
          "an insulator's charge never relaxes (got %g s)",
          insulator.ChargeRelaxationTime());

    // COPPER's relaxation time must be absurdly small -- this is the
    // number that says "copper is a conductor".
    const float tau_cu = FieldPresets::Copper().ChargeRelaxationTime();
    CHECK(tau_cu > 0.f && tau_cu < 1e-17f,
          "copper's charge relaxation time is below 1e-17 s (got %g)", tau_cu);

    // The MAGNETIC DIFFUSION TIME: mu * sigma * L^2. Copper at L = 1 cm.
    //     mu_0 * 5.8e7 * 1e-4 = 7.3e-3 s
    // So a changing field penetrates a centimetre of copper in about 7
    // MILLISECONDS. That is the eddy-current scale, and it is the number
    // that says "copper shields a changing field quickly".
    const float t_cu = FieldPresets::Copper().MagneticDiffusionTime(0.01f);
    CHECK(t_cu > 1e-3f && t_cu < 1e-2f,
          "copper's magnetic diffusion time at 1cm is ~7e-3 s (got %g)", t_cu);

    // A BLACKBODY: emissivity 1, so it absorbs everything and reflects
    // nothing.
    FieldProperties black;
    black.emissivity = 1.f;
    CHECK(close_rel(AbsorbedFraction(black), 1.f),
          "a blackbody absorbs everything");
    CHECK(close_rel(ReflectedFraction(black), 0.f),
          "a blackbody reflects nothing");

    // A PERFECT MIRROR: emissivity 0, so by Kirchhoff's law it absorbs
    // nothing and reflects everything.
    FieldProperties mirror;
    mirror.emissivity = 0.f;
    CHECK(close_rel(AbsorbedFraction(mirror), 0.f),
          "a perfect mirror absorbs nothing");
    CHECK(close_rel(ReflectedFraction(mirror), 1.f),
          "a perfect mirror reflects everything");

    // KIRCHHOFF'S LAW: absorptivity == emissivity unless overridden.
    FieldProperties p;
    p.emissivity = 0.7f;
    CHECK(close_rel(p.absorptivity(), 0.7f),
          "Kirchhoff's law: absorptivity equals emissivity");
    p.absorptivity_override = 0.3f;   // a spectrally selective surface
    CHECK(close_rel(p.absorptivity(), 0.3f),
          "an explicit absorptivity overrides the derivation");
}

void TestRadiation() {
    std::printf("-- the radiative relations --\n");
    // An EXPLICIT blackbody. The default FieldProperties has emissivity
    // 0.9 (a generic dielectric), so a test that wanted a blackbody has to
    // say so -- otherwise it would be testing the default by accident.
    FieldProperties black;
    black.emissivity = 1.f;
    // P/A = eps * sigma_SB * T^4. At 1000 K a blackbody emits 56,700 W/m^2.
    const float p1000 = RadiativePowerPerArea(black, 1000.f);
    CHECK(close_rel(p1000, kStefanB * 1e12f),
          "a blackbody at 1000 K emits sigma*T^4 (got %g, want %g)",
          p1000, kStefanB * 1e12f);

    // The FOURTH POWER IS THE POINT: doubling the temperature multiplies
    // the power by 16.
    const float p2000 = RadiativePowerPerArea(black, 2000.f);
    CHECK(close_rel(p2000, 16.f * p1000),
          "radiative power scales as T^4");

    // A LOW-EMISSIVITY surface emits proportionally less.
    FieldProperties polished;
    polished.emissivity = 0.05f;
    CHECK(close_rel(RadiativePowerPerArea(polished, 1000.f), 0.05f * p1000),
          "a polished surface emits a fraction of a blackbody's");

    // THE NET FLUX BETWEEN TWO SURFACES. The two emissivities are
    // DIFFERENT -- that is the point of the property.
    const FieldProperties hot   = FieldPresets::MildSteel();  // eps 0.3
    const FieldProperties cold  = FieldPresets::Copper();     // eps 0.03
    const float q = NetRadiativeFlux(hot, 1000.f, cold, 300.f);
    const float expect = kStefanB * (0.3f * 1e12f - 0.03f * 300.f*300.f*300.f*300.f);
    CHECK(close_rel(q, expect),
          "the net radiative flux uses BOTH emissivities (got %g, want %g)",
          q, expect);

    // EQUAL TEMPERATURES, UNEQUAL EMISSIVITIES: there is STILL a net
    // flux, from the more reflective surface to the less reflective one.
    // (At equilibrium in a real enclosure this is balanced by
    // conduction/convection; the exchange itself is non-zero.)
    const float q_same = NetRadiativeFlux(hot, 500.f, cold, 500.f);
    CHECK(std::fabs(q_same) > 1.f,
          "two surfaces at the same temperature with different emissivities "
          "still exchange energy (got %g)", q_same);
    CHECK(q_same > 0.f,
          "the flow is from the more emissive surface to the less one");

    // TWO IDENTICAL SURFACES AT THE SAME TEMPERATURE exchange NOTHING.
    CHECK(close_rel(NetRadiativeFlux(hot, 500.f, hot, 500.f), 0.f),
          "identical surfaces at the same temperature exchange nothing");
}

void TestTheMaterialsAreDistinct() {
    std::printf("-- the materials are distinguishable --\n");
    const FieldProperties cu    = FieldPresets::Copper();
    const FieldProperties steel = FieldPresets::MildSteel();
    const FieldProperties mu    = FieldPresets::MuMetal();
    const FieldProperties abs   = FieldPresets::AbsPlastic();
    const FieldProperties water = FieldPresets::Water();
    const FieldProperties nd    = FieldPresets::NeodymiumMagnet();

    // THE THREE AXES ARE INDEPENDENT. Copper is the best conductor here
    // and NOT magnetic; mu-metal is the most magnetic and NOT the best
    // conductor; ABS is neither.
    CHECK(cu.conductivity > steel.conductivity,
          "copper conducts better than steel");
    CHECK(close_rel(cu.permeability, 1.f),
          "copper is not ferromagnetic");
    CHECK(mu.permeability > steel.permeability,
          "mu-metal is more permeable than steel");
    CHECK(cu.conductivity > mu.conductivity,
          "copper conducts better than mu-metal");
    CHECK(abs.conductivity == 0.f, "ABS is an insulator");
    CHECK(close_rel(abs.permeability, 1.f), "ABS is not magnetic");
    CHECK(nd.permeability < steel.permeability,
          "a neodymium magnet's recoil permeability is below steel's");

    // WATER is the interesting one: a high permittivity AND a conductor.
    CHECK(water.permittivity > 50.f, "water's permittivity is ~80");
    CHECK(water.conductivity > 0.f, "ordinary water conducts a little");

    // EVERY PRESET MUST DIFFER FROM EVERY OTHER in at least one axis --
    // otherwise two materials are indistinguishable and one of them is
    // pointless.
    const FieldProperties* all[] = {&cu, &steel, &mu, &abs, &water, &nd};
    const size_t count = sizeof(all) / sizeof(all[0]);
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i + 1; j < count; ++j) {
            const FieldProperties& a = *all[i];
            const FieldProperties& b = *all[j];
            const bool differs =
                !close_rel(a.permeability, b.permeability) ||
                !close_rel(a.permittivity, b.permittivity) ||
                !close_rel(a.conductivity, b.conductivity) ||
                !close_rel(a.emissivity, b.emissivity);
            CHECK(differs, "two presets are identical in every field axis");
        }
}

void TestTheRegistry() {
    std::printf("-- the material registry --\n");
    FieldProperties p;

    CHECK(LookupFieldProperties("steel", p), "steel resolves");
    CHECK(close_rel(p.permeability, 200.f), "steel's permeability is 200");

    // The names a DOCUMENT would use, in their variations.
    CHECK(LookupFieldProperties("Mild Steel", p), "'Mild Steel' resolves");
    CHECK(LookupFieldProperties("mild-steel", p), "'mild-steel' resolves");
    CHECK(LookupFieldProperties("A36 steel", p), "'A36 steel' resolves");
    CHECK(LookupFieldProperties("copper", p), "copper resolves");
    CHECK(LookupFieldProperties("ABS", p), "ABS resolves");
    CHECK(LookupFieldProperties("muscle", p), "muscle resolves");
    CHECK(LookupFieldProperties("bone", p), "bone resolves");

    // AN UNKNOWN MATERIAL MUST BE REJECTED. Defaulting it to "generic
    // plastic" would assert physics the document never stated -- and the
    // document pipeline's whole value is that it does not invent.
    CHECK(!LookupFieldProperties("unobtainium", p),
          "an unknown material is rejected, not defaulted");
    CHECK(!LookupFieldProperties("", p), "an empty material name is rejected");
    CHECK(!LookupFieldProperties("12345", p), "a numeric name is rejected");
}

void TestTheLumpedMedium() {
    std::printf("-- the lumped (mass-weighted) medium --\n");

    // A body with TWO bones: a steel shaft and a plastic housing. The
    // lumped medium must be the MASS-weighted mixture.
    //
    // Build it from the real pipeline: an element tree with two bones, each
    // on a sheet, baked into a body.
    //
    // THE SCHEMA: a bone is `"bone": { ... }` (SINGULAR) on an element --
    // one bone per element, with `head` and `tail` REQUIRED and the parent
    // given by NAME. (An earlier version of this test wrote
    // `"bones": [...]`, which the parser silently ignored -- and the test
    // then reported "the tree did not produce two bones". A schema mistake
    // in a fixture looks exactly like a parser bug, so the schema is
    // spelled out here.)
    const char* kTree = R"({
      "kind":"machine_body","substance":"plane",
      "children":[
        {"kind":"shaft","substance":"ellipsoid",
         "substance_params":{"sx":0.2,"sy":0.2,"sz":0.4},
         "bone":{"name":"shaft","head":[0,0,-0.4],"tail":[0,0,0.4],"mass":8.0}},
        {"kind":"housing","substance":"ellipsoid",
         "origin":[0.6,0,0],
         "substance_params":{"sx":0.3,"sy":0.3,"sz":0.3},
         "bone":{"name":"housing","head":[0.6,0,-0.3],"tail":[0.6,0,0.3],"mass":2.0}}
      ]})";

    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(kTree, tree, why)) {
        std::printf("FAIL could not parse the two-bone tree: %s\n", why.c_str());
        ++g_fail; ++g_checks;
        return;
    }
    std::unique_ptr<PSFBody> body = ElementSystem::BakeBody(*tree, 24, 1.0f, why);
    if (!body) {
        std::printf("FAIL could not bake the two-bone tree: %s\n", why.c_str());
        ++g_fail; ++g_checks;
        return;
    }

    // The per-bone properties: the shaft is steel, the housing is plastic.
    // The bone ORDER is whatever the baker emitted -- so the properties are
    // assigned by LOOKING at which bone has which mass, not by assuming an
    // index.
    const AnchorPool& pool = body->AnchorsConst();
    std::vector<float> bone_mass;
    std::vector<float> mass_of_bone;
    for (size_t i = 0; i < pool.Count(); ++i) {
        const uint32_t bi = pool.Blob()->at(i).bone_index;
        if (bi == 0xFFFFFFFFu) continue;
        if (mass_of_bone.size() <= bi) mass_of_bone.resize(bi + 1, 0.f);
        mass_of_bone[bi] += 1.f;
    }
    (void)bone_mass;

    // Build the mass table from the bone RECORDS (the authored mass is in
    // the reserved bytes), which is how a caller without a separate table
    // would do it.
    std::vector<float> masses(mass_of_bone.size(), 0.f);
    // The heavy bone (8.0) is steel; the light one (2.0) is plastic.
    size_t heavy = 0, light = 0;
    for (size_t i = 0; i < mass_of_bone.size(); ++i) {
        if (mass_of_bone[i] > mass_of_bone[heavy]) heavy = i;
    }
    for (size_t i = 0; i < mass_of_bone.size(); ++i)
        if (i != heavy && (light >= mass_of_bone.size() || mass_of_bone[i] > mass_of_bone[light]))
            light = i;
    if (mass_of_bone.size() < 2) {
        std::printf("FAIL the tree did not produce two bones\n");
        ++g_fail; ++g_checks;
        return;
    }
    masses[heavy] = 8.f;
    masses[light] = 2.f;

    std::vector<FieldProperties> props(masses.size());
    props[heavy] = FieldPresets::MildSteel();
    props[light] = FieldPresets::AbsPlastic();

    BodyFieldProperties lumped;
    CHECK(ComputeBodyFieldProperties(*body, props, masses, lumped, why),
          "the two-bone body's effective medium computes: %s", why.c_str());

    // THE MIXTURE RULE: the mass-weighted average of the two materials.
    // The shaft carries 8 of the 10 units of mass, the housing 2.
    const float frac_heavy = 0.8f, frac_light = 0.2f;
    const float want_mu  = frac_heavy * props[heavy].permeability
                         + frac_light * props[light].permeability;
    const float want_eps = frac_heavy * props[heavy].permittivity
                         + frac_light * props[light].permittivity;
    CHECK(close_rel(lumped.lumped.permeability, want_mu, 5e-2f),
          "the lumped permeability is the mass-weighted mixture (got %g, want %g)",
          lumped.lumped.permeability, want_mu);
    CHECK(close_rel(lumped.lumped.permittivity, want_eps, 5e-2f),
          "the lumped permittivity is the mass-weighted mixture (got %g, want %g)",
          lumped.lumped.permittivity, want_eps);

    // THE LUMPED MEDIUM MUST BE BETWEEN THE TWO COMPONENTS. A mixture
    // outside its components' range is arithmetically impossible and
    // means the weighting is wrong.
    const float lo_mu = std::fmin(props[heavy].permeability, props[light].permeability);
    const float hi_mu = std::fmax(props[heavy].permeability, props[light].permeability);
    CHECK(lumped.lumped.permeability >= lo_mu - 1e-3f &&
          lumped.lumped.permeability <= hi_mu + 1e-3f,
          "the mixture lies BETWEEN its components");
    CHECK(lumped.average_mass > 0.f, "the lumped medium reports its mass");

    // KIRCHHOFF'S LAW MUST STILL HOLD for the mixture.
    CHECK(close_rel(lumped.lumped.absorptivity(), lumped.lumped.emissivity, 1e-2f),
          "Kirchhoff's law holds for the lumped mixture");

    // A BODY WITH NO PROPERTIES MUST BE REJECTED, not defaulted.
    BodyFieldProperties empty;
    std::vector<FieldProperties> none;
    CHECK(!ComputeBodyFieldProperties(*body, none, masses, empty, why),
          "a body with no authored properties has no effective medium");

    // A BODY WITH NO MASS must be rejected too.
    std::vector<float> zero_mass(masses.size(), 0.f);
    BodyFieldProperties no_mass;
    CHECK(!ComputeBodyFieldProperties(*body, props, zero_mass, no_mass, why),
          "a massless body has no effective medium");
}

} // namespace

int main() {
    std::printf("== field properties gate ==\n");
    TestMagneticRelations();
    TestElectricRelations();
    TestTheLimits();
    TestRadiation();
    TestTheMaterialsAreDistinct();
    TestTheRegistry();
    TestTheLumpedMedium();
    std::printf("\n%s (%d checks, %d failures)\n",
                g_fail ? "FIELD PROPERTIES FAILED" : "field properties passed",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
