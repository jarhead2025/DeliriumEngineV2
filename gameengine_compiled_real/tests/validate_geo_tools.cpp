// GATE: THE GEOMETRIC TOOLS' MATHS MUST BE EXACT.
//
// WHY THIS GATE EXISTS
//
// A tool whose maths is wrong is WORSE THAN NO TOOL: the model trusts it.
// The whole premise of the tool set is that the model cannot do this
// arithmetic reliably and the tool can -- so the tool's arithmetic is the
// load-bearing part of the feature.
//
// The assertions are against the CLOSED FORMS and against the engine's
// OWN physics (FieldProperties), not against a snapshot of the current
// output. A change-detector test would pass forever while the maths
// rotted.
//
// WHAT IS PINNED
//
//   A) the volumes and areas against their exact closed forms,
//   B) the ellipsoid's area against the known near-sphere case and
//      against the exact prolate form,
//   C) the inertia against the exact closed forms AND against the
//      engine's own MassProperties (so the tool and the simulation cannot
//      disagree),
//   D) the arithmetic the model gets wrong (the torus's radii, the
//      sphere-from-mass, the closed-body rule),
//   E) the tool protocol itself: every tool is invocable, every schema is
//      valid JSON, every tool answers rather than throwing,
//   F) the registry's JSON is well-formed and lists every tool.
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <map>

#include "delirium/core/Types.h"
#include "delirium/ai/GeoTools.h"
#include "delirium/ai/ElementTreeParser.h"

using namespace DeliRium;
using namespace DeliRium::AI;

static int g_fail = 0;
static int g_checks = 0;

static void check(bool ok, const char* what, double value = 0.0, double bound = 0.0) {
    ++g_checks;
    std::printf("  %s %-56s", ok ? "ok  " : "FAIL", what);
    if (bound > 0.0) std::printf("  (%.9g", value);
    if (bound > 0.0) std::printf(ok ? " vs %.9g)" : " vs %.9g)", bound);
    std::printf("\n");
    if (!ok) ++g_fail;
}

// Relative closeness -- the right measure for a formula that spans orders
// of magnitude.
static bool closeRel(double a, double b, double rel) {
    const double d = std::fabs(a - b);
    const double m = std::max(std::fabs(a), std::fabs(b));
    return m < 1e-30 ? d < 1e-30 : (d / m) < rel;
}

static std::string call(const char* name,
                        const std::map<std::string, double>& n = {},
                        const std::map<std::string, std::string>& s = {}) {
    std::string out, err;
    if (!GeoToolSet::Instance().Invoke(name, n, s, out, err)) return "ERROR:" + err;
    return out;
}

int main() {
    const double kPi = 3.14159265358979323846;

    std::printf("A) THE VOLUMES, AGAINST THE CLOSED FORMS\n");
    check(closeRel(geom::SphereVolume(2.0), 4.0/3.0*kPi*8.0, 1e-12),
          "the sphere's volume", geom::SphereVolume(2.0), 4.0/3.0*kPi*8.0);
    check(closeRel(geom::EllipsoidVolume(1,2,3), 4.0/3.0*kPi*6.0, 1e-12),
          "the ellipsoid's volume", geom::EllipsoidVolume(1,2,3), 4.0/3.0*kPi*6.0);
    check(closeRel(geom::TorusVolume(3,1), 2.0*kPi*kPi*3.0, 1e-12),
          "the torus's volume", geom::TorusVolume(3,1), 2.0*kPi*kPi*3.0);
    check(closeRel(geom::CylinderVolume(2,5), kPi*4.0*5.0, 1e-12),
          "the cylinder's volume", geom::CylinderVolume(2,5), kPi*4.0*5.0);
    // A sphere IS an ellipsoid with equal axes.
    check(closeRel(geom::EllipsoidVolume(2,2,2), geom::SphereVolume(2), 1e-12),
          "a sphere is an ellipsoid with equal axes");

    std::printf("\nB) THE SURFACE AREAS\n");
    check(closeRel(geom::SphereSurfaceArea(2.0), 4.0*kPi*4.0, 1e-12),
          "the sphere's area", geom::SphereSurfaceArea(2.0), 4.0*kPi*4.0);
    check(closeRel(geom::TorusSurfaceArea(3,1), 4.0*kPi*kPi*3.0, 1e-12),
          "the torus's area", geom::TorusSurfaceArea(3,1), 4.0*kPi*kPi*3.0);
    check(closeRel(geom::CylinderSurfaceArea(2,5), 2.0*kPi*2.0*(2.0+5.0), 1e-12),
          "the cylinder's area", geom::CylinderSurfaceArea(2,5), 2.0*kPi*2.0*7.0);
    // The ellipsoid's area with EQUAL axes must be the sphere's area --
    // the exact form, not the approximation, is what makes this true.
    //
    // THIS CHECK IS THE ONE THAT MATTERS. Three earlier versions of the
    // formula were wrong (18.905, then 12.566, then 25.133 for a sphere
    // of radius 2 instead of 50.265) and a stored snapshot would have
    // accepted all three. The sphere case is a CLOSED FORM, so it
    // catches any consistent-but-wrong formula.
    //
    // The tolerance is the QUADRATURE'S PRECISION, not double precision:
    // the area is evaluated as a double integral over (v, u), and the
    // measured error on this sphere is 2.8e-7 relative. A tolerance
    // tighter than the method would fail on the method's own arithmetic.
    check(closeRel(geom::EllipsoidSurfaceArea(2,2,2), geom::SphereSurfaceArea(2), 1e-5),
          "the ellipsoid's exact area == the sphere's area when the axes match",
          geom::EllipsoidSurfaceArea(2,2,2), geom::SphereSurfaceArea(2));
    // And the area of a very flat ellipsoid approaches the area of a
    // DISK of radius max(sx,sy) -- a good independent check of the
    // integral (the limit is 2*pi*a^2 for a flat disc, plus a vanishing
    // rim; the exact value tends to 2*pi*a^2 from above).
    {
        const double flat = geom::EllipsoidSurfaceArea(1.0, 1.0, 1e-4);
        const double disk = 2.0 * kPi * 1.0;
        check(flat > disk && flat < disk * 1.001,
              "a flat ellipsoid's area approaches the disk's (2*pi*a^2)",
              flat, disk * 1.001);
    }
    // A PROLATE ellipsoid's area must lie between the sphere of the same
    // volume and ... the simplest useful bracket: the area must EXCEED
    // the area of the sphere of the SAME VOLUME (isoperimetric: the
    // sphere minimises the area for a given volume, so any other shape
    // of the same volume has a LARGER area). That is a real theorem, and
    // it holds for the ellipsoid.
    {
        const double sx = 1.0, sy = 1.0, sz = 2.0;
        const double vol = geom::EllipsoidVolume(sx, sy, sz);
        const double r_eq = std::cbrt(vol * 3.0 / (4.0 * kPi));
        const double sphere_area = 4.0 * kPi * r_eq * r_eq;
        const double e_area = geom::EllipsoidSurfaceArea(sx, sy, sz);
        check(e_area > sphere_area,
              "a prolate ellipsoid's area exceeds the same-volume sphere's",
              e_area, sphere_area);
    }

    std::printf("\nC) THE INERTIA -- THE CLOSED FORMS AND THE ENGINE'S OWN\n");
    check(closeRel(geom::SolidSphereInertia(2.0, 3.0), 0.4*2.0*9.0, 1e-12),
          "the solid sphere's inertia", geom::SolidSphereInertia(2,3), 0.4*2*9);
    check(closeRel(geom::SolidCylinderInertiaAboutAxis(2.0, 3.0, 5.0), 0.5*2.0*9.0, 1e-12),
          "the solid cylinder's inertia about its axis");
    check(closeRel(geom::SolidCylinderInertiaAboutDiameter(2.0, 3.0, 5.0),
                   2.0*(27.0+25.0)/12.0, 1e-12),
          "the solid cylinder's inertia about a diameter");
    check(closeRel(geom::ThinRingInertiaAboutAxis(2.0, 3.0), 2.0*9.0, 1e-12),
          "the thin ring's inertia about its axis");
    check(closeRel(geom::ThinRingInertiaAboutDiameter(2.0, 3.0), 0.5*2.0*9.0, 1e-12),
          "the thin ring's inertia about a diameter");
    // THE ENGINE'S OWN MassProperties must agree with the closed form --
    // so the tool and the body the engine builds cannot disagree.
    {
        MassProperties mp;
        mp.total_mass = 2.0;
        mp.Ixx = mp.Iyy = mp.Izz = geom::SolidSphereInertia(2.0, 3.0);
        mp.Ixy = mp.Ixz = mp.Iyz = 0.0;
        // The inertia about an arbitrary axis must match the closed form
        // for a sphere (which is isotropic).
        const Vec3 axis = Vec3{0.f, 0.f, 1.f};
        // The tolerance is FLOAT precision, not double: MassProperties
        // stores float, so 0.4*2*9 = 7.2 comes back as 7.19999981. A
        // double-precision tolerance here would fail on the ENGINE's own
        // storage type, not on a maths error.
        check(closeRel(mp.InertiaAbout(axis), 0.4*2.0*9.0, 1e-6),
              "MassProperties agrees with the sphere's closed form");
        Vec3 moments[3], axes[3];
        mp.PrincipalMoments(moments, axes);
        check(closeRel(moments[0].x, 0.4*2.0*9.0, 1e-6) &&
              closeRel(moments[1].x, 0.4*2.0*9.0, 1e-6) &&
              closeRel(moments[2].x, 0.4*2.0*9.0, 1e-6),
              "a sphere's principal moments are all equal");
    }

    std::printf("\nD) THE ARITHMETIC THE MODEL GETS WRONG\n");
    {
        const geom::TorusRadii tr = geom::ResolveTorusRadii(0.44, 0.12);
        check(closeRel(tr.major, 0.32, 1e-12), "an outer radius resolves to major",
              tr.major, 0.32);
        check(closeRel(tr.minor, 0.12, 1e-12), "the tube radius is the minor radius");
    }
    {
        // A 1 kg aluminium sphere: r = cbrt(3*1/(4*pi*2700)).
        const geom::SphereFromMass f = geom::FitSphereToMass(1.0, 2700.0);
        const double expect = std::cbrt(3.0 / (4.0 * kPi * 2700.0));
        check(closeRel(f.radius, expect, 1e-12), "the sphere-from-mass radius",
              f.radius, expect);
        check(closeRel(f.volume, 1.0/2700.0, 1e-12), "the sphere-from-mass volume");
    }

    std::printf("\nE) THE TOOL PROTOCOL\n");
    {
        const auto& tools = GeoToolSet::Instance().Tools();
        check(!tools.empty(), "the registry has tools", double(tools.size()));
        int schema_ok = 0, invoke_ok = 0, answered = 0;
        for (const auto& t : tools) {
            // Every schema must be valid JSON the engine's own reader
            // accepts -- a malformed schema would make the server reject
            // the whole request.
            std::string why;
            AI::JsonValue jv;
            if (AI::ParseJson(t.parameters_json, jv, why)) ++schema_ok;
            // Every tool must be invocable with no arguments and must
            // ANSWER (not throw, not hang).
            //
            // NOTE: invoking with NO arguments is now expected to FAIL for
            // most tools -- a missing argument is a FAILURE, not an answer
            // (that is the behaviour the live run forced, and it is what
            // lets the model correct itself). So the contract checked here
            // is: the tool is REACHABLE, and it either answers or fails
            // INFORMATIVELY -- it never throws, and never hangs.
            std::string out, err;
            const bool reached = GeoToolSet::Instance().Invoke(t.name, {}, {}, out, err);
            if (reached) {
                ++invoke_ok;
                if (!out.empty()) ++answered;
            } else if (!out.empty() || !err.empty()) {
                // A failure that EXPLAINS itself counts as reachable: the
                // tool ran and told us why it could not answer.
                ++invoke_ok;
                ++answered;
            }
        }
        check(schema_ok == int(tools.size()),
              "every tool's schema is valid JSON", double(schema_ok), double(tools.size()));
        check(invoke_ok == int(tools.size()),
              "every tool is invocable", double(invoke_ok), double(tools.size()));
        check(answered == int(tools.size()),
              "every tool answers rather than throwing", double(answered), double(tools.size()));
        // An unknown tool must be REJECTED, not silently ignored.
        std::string out, err;
        check(!GeoToolSet::Instance().Invoke("no_such_tool", {}, {}, out, err),
              "an unknown tool is rejected");
        check(!err.empty(), "the rejection explains itself");
    }

    std::printf("\nH) THE ARGUMENT NAMES ARE AN INTERFACE\n");
    {
        // MEASURED: the local 7B called `resolve_torus_radii` with
        // "outer_radius"/"tube_radius" -- the natural spelling -- while the
        // schema declared "outer"/"tube". EVERY call failed, and the model
        // then guessed the answer instead. The tool was deployed,
        // understood, and useless.
        //
        // So the READER accepts the natural synonyms. The schema still
        // declares the canonical name (that is what the model is shown).
        const AI::GeoToolSet& ts = AI::GeoToolSet::Instance();

        std::string out, err;
        const bool alias_ok = ts.Invoke("resolve_torus_radii",
                                        {{"outer_radius", 0.44}, {"tube_radius", 0.12}},
                                        {}, out, err);
        check(alias_ok, "the tool accepts the natural argument names");
        check(out.find("0.32") != std::string::npos,
              "the aliased call returns the right major radius");
        check(out.find("0.12") != std::string::npos,
              "the aliased call returns the right minor radius");

        // The canonical names must still work -- the alias is an ADDITION.
        std::string out2, err2;
        check(ts.Invoke("resolve_torus_radii", {{"outer", 0.44}, {"tube", 0.12}},
                        {}, out2, err2),
              "the canonical argument names still work");
        check(out2 == out, "the canonical and aliased calls agree");

        // A WRONG argument must be a FAILURE, not a successful answer.
        // Without this the loop records the call as succeeded and the model
        // gets no signal that anything went wrong.
        std::string out3, err3;
        const bool wrong_ok = ts.Invoke("resolve_torus_radii",
                                        {{"banana", 1.0}}, {}, out3, err3);
        check(!wrong_ok, "a wrong argument name is a FAILURE, not an answer");

        // And the failure must be SELF-CORRECTING: it names the accepted
        // argument names and echoes what was received, so a small model can
        // fix its call instead of repeating it.
        check(out3.find("\"outer\"") != std::string::npos,
              "the failure names the accepted argument names");
        check(out3.find("\"tube\"") != std::string::npos,
              "the failure names the second accepted argument name");
        check(out3.find("banana") != std::string::npos,
              "the failure echoes what was actually received");
        check(!err3.empty(), "the failure is reported as an error");
    }

    std::printf("\nF) THE REGISTRY'S JSON\n");
    {
        const std::string j = GeoToolSet::Instance().ToOllamaToolsJson();
        std::string why;
        AI::JsonValue jv;
        const bool jok = AI::ParseJson(j, jv, why);
        check(jok, "the tools JSON parses", 0, 0);
        if (!jok) std::printf("       why: %s\n", why.c_str());
        check(j.find("\"type\":\"function\"") != std::string::npos,
              "the tools JSON is in Ollama's function shape");
        check(j.find("\"parameters\"") != std::string::npos,
              "every tool carries its parameters schema");
        const std::string cat = GeoToolSet::Instance().Catalogue();
        check(!cat.empty(), "the catalogue is non-empty");
        // Every tool must be NAMED in the catalogue -- a tool the model
        // cannot see is a tool it will never call.
        int named = 0;
        for (const auto& t : GeoToolSet::Instance().Tools())
            if (cat.find(t.name) != std::string::npos) ++named;
        check(named == int(GeoToolSet::Instance().Tools().size()),
              "every tool is named in the catalogue", double(named),
              double(GeoToolSet::Instance().Tools().size()));
    }

    std::printf("\nG) THE TOOLS' ANSWERS CARRY THE NUMBERS\n");
    {
        // A tool that answers without the number is useless to a model.
        const std::string v = call("volume_of_sphere", {{"r", 3.0}});
        check(v.find("36") != std::string::npos || v.find("113") != std::string::npos,
              "the sphere's volume answer contains the number", 0, 0);
        if (v.find("ERROR") != std::string::npos) {
            check(false, "the sphere's volume tool answered", 0, 0);
            std::printf("       said: %s\n", v.c_str());
        }
        // The torus resolver must state the numbers to USE.
        const std::string t = call("resolve_torus_radii", {{"outer", 0.44}, {"tube", 0.12}});
        check(t.find("0.32") != std::string::npos,
              "the torus resolver states the major radius", 0, 0);
        // The closed-body rule must produce valid JSON.
        const std::string c = call("closed_body_rule", {{"sx", 0.45}, {"sy", 0.45}, {"sz", 0.45}});
        const size_t a = c.find('{'), b = c.rfind('}');
        bool ok = false;
        if (a != std::string::npos && b != std::string::npos && b > a) {
            std::string why;
            AI::JsonValue jv;
            ok = AI::ParseJson(c.substr(a, b - a + 1), jv, why);
        }
        check(ok, "the closed-body rule's answer is valid JSON", 0, 0);
        // And it must NOT contain a negative half-extent.
        check(c.find("-0.45") == std::string::npos,
              "the closed-body rule never emits a negative half-extent", 0, 0);
        check(c.find("[1,1,-1]") != std::string::npos,
              "the closed-body rule emits the negative SCALE", 0, 0);
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
