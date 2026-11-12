// =============================================================================
// GeoTools.cpp - the geometric analysis tools' implementation.
//
// THE GEOMETRY IS THE POINT. Every number these tools return is computed
// from a closed form or from the engine's own physics -- never guessed,
// never approximated by the model, never described in a prompt and hoped
// for.
//
// The exact closed forms used here:
//
//   sphere        V = 4/3 pi r^3              A = 4 pi r^2
//   ellipsoid     V = 4/3 pi sx sy sz         A = the exact integral form
//   torus         V = 2 pi^2 R r^2            A = 4 pi^2 R r
//   cylinder      V = pi r^2 h                A = 2 pi r (r + h)
//
//   solid sphere, about its centre: I = 2/5 m r^2
//   solid cylinder, about its axis: I = 1/2 m r^2
//   solid cylinder, about a diameter: I = 1/12 m (3 r^2 + h^2)
//   thin ring, about its axis:        I = m r^2
//   thin ring, about a diameter:      I = 1/2 m r^2
//
// WHY THE ENGINE'S OWN PHYSICS IS REUSED: the field relations come from
// FieldProperties (so the tool and the simulation cannot disagree), and
// the inertia is seeded into MassProperties (so the tool and the body's
// actual inertia cannot disagree). A tool that duplicates the maths is a
// tool that will eventually disagree with the engine.
// =============================================================================
#include "delirium/ai/GeoTools.h"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <algorithm>

namespace DeliRium {
namespace AI {
namespace geom {

static const double kPi = 3.14159265358979323846;

double SphereVolume(double r) { return 4.0 / 3.0 * kPi * r * r * r; }
double EllipsoidVolume(double sx, double sy, double sz) {
    return 4.0 / 3.0 * kPi * sx * sy * sz;
}
double TorusVolume(double major, double minor) {
    return 2.0 * kPi * kPi * major * minor * minor;
}
double CylinderVolume(double r, double h) { return kPi * r * r * h; }

double SphereSurfaceArea(double r) { return 4.0 * kPi * r * r; }

// The EXACT surface area of an ellipsoid.
//
// It lives in its own file (ai/EllipsoidArea.cpp) because it needs the
// derivation and the two sanity checks written out, and because it is the
// one formula in this file that a small slip in the algebra ruins
// silently -- an earlier version put sin/cos in the wrong places inside
// the square root and returned 18.905 for a sphere of radius 2 instead of
// 50.265.
//
// The usual "1.6075 approximation" is NOT exact and is not used: a tool
// that claims exactness has to be exact.

double TorusSurfaceArea(double major, double minor) {
    return 4.0 * kPi * kPi * major * minor;
}
double CylinderSurfaceArea(double r, double h) {
    return 2.0 * kPi * r * (r + h);
}

TorusRadii ResolveTorusRadii(double outer, double tube) {
    TorusRadii out{};
    out.major = outer - tube;
    out.minor = tube;
    return out;
}

SphereFromMass FitSphereToMass(double mass, double density) {
    SphereFromMass out{};
    if (mass <= 0 || density <= 0) return out;
    out.volume = mass / density;
    out.radius = std::cbrt(out.volume * 3.0 / (4.0 * kPi));
    return out;
}

double SolidSphereInertia(double mass, double r) { return 0.4 * mass * r * r; }
double SolidCylinderInertiaAboutAxis(double mass, double r, double h) {
    return 0.5 * mass * r * r;
}
double SolidCylinderInertiaAboutDiameter(double mass, double r, double h) {
    return mass * (3.0 * r * r + h * h) / 12.0;
}
double ThinRingInertiaAboutAxis(double mass, double r) { return mass * r * r; }
double ThinRingInertiaAboutDiameter(double mass, double r) { return 0.5 * mass * r * r; }

double FieldMagneticDiffusionTime(const FieldProperties& p, double length) {
    return p.MagneticDiffusionTime((float)length);
}
double FieldChargeRelaxationTime(const FieldProperties& p) {
    return p.ChargeRelaxationTime();
}
double FieldRadiativePowerPerArea(const FieldProperties& p, double temperature_kelvin) {
    return RadiativePowerPerArea(p, (float)temperature_kelvin);
}

}  // namespace geom

// =============================================================================
// THE TOOL REGISTRY
// =============================================================================

// A small helper: the tools' handlers read their arguments from two maps
// (numbers and strings) because Ollama's tool-call arguments are a JSON
// object whose values may be either. The client's parser flattens them
// into these two maps.
//
// -----------------------------------------------------------------------------
// ALIAS TOLERANCE -- WHY IT EXISTS (MEASURED, not assumed)
// -----------------------------------------------------------------------------
//
// The first live run of the geo-tool probe showed the local 7B calling
// `resolve_torus_radii` with:
//
//     {"outer_radius":0.44,"tube_radius":0.12}
//
// while the tool's schema declared `outer` and `tube`. EVERY call failed
// with "missing outer or tube" -- and the model then GUESSED the right
// answer anyway. The tool was deployed, understood, and useless.
//
// The lesson is not "the model is bad at schemas". The lesson is: a tool's
// parameter names are an INTERFACE, and a model will reach for the most
// natural spelling of a quantity -- `outer_radius` for "the outer radius",
// not the two-letter abbreviation `outer`.
//
// So the lookup accepts the natural synonyms. The schema still declares
// the canonical name (that is what the model is shown), but the READER is
// forgiving. A tool that answers is worth more than a tool that is
// pedantically correct.
//
// The alias list is deliberately small and explicit -- a fuzzy matcher
// would make a typo silently mean something else.
static bool Num(const std::map<std::string, double>& n, const char* k, double& out) {
    auto it = n.find(k);
    if (it != n.end()) { out = it->second; return true; }
    // THE ALIASES. The canonical name first, then the natural synonyms a
    // model actually uses.
    static const std::map<std::string, std::vector<std::string>> kAliases = {
        { "r",       { "radius" } },
        { "outer",   { "outer_radius", "outer_r", "outerRadius" } },
        { "tube",    { "tube_radius", "tube_r", "tubeRadius" } },
        { "major",   { "major_radius", "major_r", "majorRadius" } },
        { "minor",   { "minor_radius", "minor_r", "minorRadius" } },
        { "mass",    { "m" } },
        { "density", { "rho" } },
        { "h",       { "height" } },
        { "sx",      { "semi_x", "x" } },
        { "sy",      { "semi_y", "y" } },
        { "sz",      { "semi_z", "z" } },
        { "length",  { "l", "characteristic_length" } },
        { "temperature_kelvin", { "temperature", "temp_kelvin", "temp" } },
    };
    auto ait = kAliases.find(k);
    if (ait == kAliases.end()) return false;
    for (const std::string& alias : ait->second) {
        auto a = n.find(alias);
        if (a != n.end()) { out = a->second; return true; }
    }
    return false;
}
static std::string Str(const std::map<std::string, std::string>& s, const char* k,
                       const std::string& dflt = "") {
    auto it = s.find(k);
    if (it != s.end()) return it->second;
    static const std::map<std::string, std::vector<std::string>> kAliases = {
        { "material", { "material_name", "mat" } },
        { "temperature_kelvin", { "temperature", "temp_kelvin", "temp" } },
    };
    auto ait = kAliases.find(k);
    if (ait != kAliases.end()) {
        for (const std::string& alias : ait->second) {
            auto a = s.find(alias);
            if (a != s.end()) return a->second;
        }
    }
    return dflt;
}
static std::string F(double v, int prec = 6) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    return std::string(buf);
}

// The parameter schemas. Kept as raw JSON strings so they go into the
// request verbatim -- the engine has no JSON writer in the AI path, and
// adding one for this would be more machinery than the schemas deserve.
static const char* kSchemaR = R"({"type":"object","properties":{"r":{"type":"number","description":"the radius"}},"required":["r"]})";
static const char* kSchemaRR = R"({"type":"object","properties":{"major":{"type":"number","description":"the tube-centre radius"},"minor":{"type":"number","description":"the tube radius"}},"required":["major","minor"]})";
static const char* kSchemaOuterTube = R"({"type":"object","properties":{"outer":{"type":"number","description":"the OUTER radius, outer edge to centre"},"tube":{"type":"number","description":"the tube radius"}},"required":["outer","tube"]})";
static const char* kSchemaEllipsoid = R"({"type":"object","properties":{"sx":{"type":"number"},"sy":{"type":"number"},"sz":{"type":"number"}},"required":["sx","sy","sz"]})";
static const char* kSchemaCylinder = R"({"type":"object","properties":{"r":{"type":"number"},"h":{"type":"number","description":"the height"}},"required":["r","h"]})";
static const char* kSchemaMassR = R"({"type":"object","properties":{"mass":{"type":"number","description":"the mass"},"r":{"type":"number","description":"the radius"}},"required":["mass","r"]})";
static const char* kSchemaMassDensity = R"({"type":"object","properties":{"mass":{"type":"number","description":"the mass"},"density":{"type":"number","description":"the density, e.g. 2700 for aluminium"}},"required":["mass","density"]})";
static const char* kSchemaMassRH = R"({"type":"object","properties":{"mass":{"type":"number"},"r":{"type":"number"},"h":{"type":"number"}},"required":["mass","r","h"]})";
static const char* kSchemaMaterial = R"({"type":"object","properties":{"material":{"type":"string","description":"a material name, e.g. steel, aluminium, copper, iron"},"length":{"type":"number","description":"a characteristic length in metres"},"temperature_kelvin":{"type":"number","description":"the temperature in kelvin"}},"required":["material"]})";

GeoToolSet& GeoToolSet::Instance() {
    static GeoToolSet inst;
    return inst;
}

GeoToolSet::GeoToolSet() {
    auto add = [&](const char* name, const char* desc, const char* schema,
                   GeoTool::handler_t h) {
        GeoTool t;
        t.name = name;
        t.description = desc;
        t.parameters_json = schema;
        t.handler = h;
        by_name_[t.name] = tools_.size();
        tools_.push_back(std::move(t));
    };

    // ---- the volumes -------------------------------------------------------
    add("volume_of_sphere",
        "The volume of a solid sphere of radius r: V = 4/3*pi*r^3. Use this to size a spherical part from its mass, or to check a proposed radius.",
        kSchemaR,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double r = 0;
            if (!Num(n, "r", r)) return std::string("ERROR: missing 'r'.");
            return "volume = " + F(geom::SphereVolume(r)) + " (cubic units), for r = " + F(r);
        });

    add("volume_of_ellipsoid",
        "The volume of a solid ellipsoid with half-extents sx, sy, sz: V = 4/3*pi*sx*sy*sz.",
        kSchemaEllipsoid,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double sx=0, sy=0, sz=0;
            if (!Num(n,"sx",sx) || !Num(n,"sy",sy) || !Num(n,"sz",sz))
                return std::string("ERROR: missing sx, sy or sz.");
            return "volume = " + F(geom::EllipsoidVolume(sx,sy,sz)) + " (cubic units)";
        });

    add("volume_of_torus",
        "The volume of a solid torus (a ring) with tube-centre radius 'major' and tube radius 'minor': V = 2*pi^2*major*minor^2. NOTE: 'major' is the tube CENTRE radius, not the outer radius -- use resolve_torus_radii if you have an outer radius.",
        kSchemaRR,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double major=0, minor=0;
            if (!Num(n,"major",major) || !Num(n,"minor",minor))
                return std::string("ERROR: missing major or minor.");
            return "volume = " + F(geom::TorusVolume(major,minor)) + " (cubic units)";
        });

    add("volume_of_cylinder",
        "The volume of a solid cylinder of radius r and height h: V = pi*r^2*h.",
        kSchemaCylinder,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double r=0, h=0;
            if (!Num(n,"r",r) || !Num(n,"h",h))
                return std::string("ERROR: missing r or h.");
            return "volume = " + F(geom::CylinderVolume(r,h)) + " (cubic units)";
        });

    // ---- the surface areas -------------------------------------------------
    add("surface_area_of_sphere",
        "The surface area of a sphere of radius r: A = 4*pi*r^2.",
        kSchemaR,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double r=0;
            if (!Num(n,"r",r)) return std::string("ERROR: missing 'r'.");
            return "surface area = " + F(geom::SphereSurfaceArea(r)) + " (square units)";
        });

    add("surface_area_of_ellipsoid",
        "The EXACT surface area of an ellipsoid with half-extents sx, sy, sz, evaluated by quadrature (not the usual approximation).",
        kSchemaEllipsoid,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double sx=0, sy=0, sz=0;
            if (!Num(n,"sx",sx) || !Num(n,"sy",sy) || !Num(n,"sz",sz))
                return std::string("ERROR: missing sx, sy or sz.");
            return "surface area = " + F(geom::EllipsoidSurfaceArea(sx,sy,sz)) + " (square units)";
        });

    add("surface_area_of_torus",
        "The surface area of a torus with tube-centre radius 'major' and tube radius 'minor': A = 4*pi^2*major*minor.",
        kSchemaRR,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double major=0, minor=0;
            if (!Num(n,"major",major) || !Num(n,"minor",minor))
                return std::string("ERROR: missing major or minor.");
            return "surface area = " + F(geom::TorusSurfaceArea(major,minor)) + " (square units)";
        });

    // ---- THE ARITHMETIC THE MODEL GETS WRONG -------------------------------
    add("resolve_torus_radii",
        "Convert an OUTER radius and a tube radius into a torus's major (tube-centre) and minor radii: major = outer - minor. USE THIS whenever a ring is described by its outer radius -- 'a ring of outer radius 0.44 with a tube of 0.12' means major = 0.32, NOT 0.44.",
        kSchemaOuterTube,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double outer=0, tube=0;
            if (!Num(n,"outer",outer) || !Num(n,"tube",tube))
                return std::string("ERROR: missing outer or tube.");
            const geom::TorusRadii tr = geom::ResolveTorusRadii(outer, tube);
            return "major (tube-centre radius) = " + F(tr.major) +
                   ", minor (tube radius) = " + F(tr.minor) +
                   ". Use major = " + F(tr.major) + " and minor = " + F(tr.minor) +
                   " in the substance_params.";
        });

    add("fit_sphere_to_mass",
        "Find the radius of a solid sphere whose mass equals the given mass at the given density: r = cbrt(3*m/(4*pi*density)). Use this to size a spherical part from a known mass and material.",
        kSchemaMassDensity,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double mass=0, density=0;
            if (!Num(n,"mass",mass) || !Num(n,"density",density))
                return std::string("ERROR: missing mass or density.");
            const geom::SphereFromMass f = geom::FitSphereToMass(mass, density);
            return "radius = " + F(f.radius) + " (volume " + F(f.volume) +
                   " cubic units) for mass " + F(mass) + " at density " + F(density);
        });

    add("closed_body_rule",
        "The rule for a CLOSED body (a full sphere, not a dome) in this engine: a body is TWO sheets -- an upper sheet and a lower sheet that is the SAME sheet with scale [1,1,-1] at the SAME origin [0,0,0]. NEVER use a negative half-extent (sz must stay positive); the flip is in the SCALE, not the size. Returns the exact JSON for the lower sheet given the upper's half-extents.",
        R"({"type":"object","properties":{"sx":{"type":"number"},"sy":{"type":"number"},"sz":{"type":"number"}},"required":["sx","sy","sz"]})",
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double sx=0, sy=0, sz=0;
            if (!Num(n,"sx",sx) || !Num(n,"sy",sy) || !Num(n,"sz",sz))
                return std::string("ERROR: missing sx, sy or sz.");
            std::ostringstream o;
            o << "The lower sheet is: {\"kind\":\"lower\",\"substance\":\"ellipsoid\","
              << "\"substance_params\":{\"sx\":" << F(sx) << ",\"sy\":" << F(sy)
              << ",\"sz\":" << F(sz) << "},"
              << "\"scale\":[1,1,-1],\"origin\":[0,0,0]}  "
              << "Note sz stays POSITIVE (" << F(sz) << "); the flip is in scale.";
            return o.str();
        });

    // ---- the inertia -------------------------------------------------------
    add("inertia_of_sphere",
        "The moment of inertia of a SOLID sphere about its centre: I = 2/5*m*r^2. (Use for a ball-like part.)",
        kSchemaMassR,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double mass=0, r=0;
            if (!Num(n,"mass",mass) || !Num(n,"r",r))
                return std::string("ERROR: missing mass or r.");
            return "I (about the centre) = " + F(geom::SolidSphereInertia(mass,r));
        });

    add("inertia_of_cylinder",
        "The moments of inertia of a SOLID cylinder of mass m, radius r and height h: about its axis I = 1/2*m*r^2; about a diameter I = m*(3*r^2 + h^2)/12.",
        kSchemaMassRH,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double mass=0, r=0, h=0;
            if (!Num(n,"mass",mass) || !Num(n,"r",r) || !Num(n,"h",h))
                return std::string("ERROR: missing mass, r or h.");
            return "I about the axis = " + F(geom::SolidCylinderInertiaAboutAxis(mass,r,h)) +
                   ", I about a diameter = " + F(geom::SolidCylinderInertiaAboutDiameter(mass,r,h));
        });

    add("inertia_of_ring",
        "The moments of inertia of a THIN ring of mass m and radius r: about its axis I = m*r^2; about a diameter I = 1/2*m*r^2.",
        kSchemaMassR,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>&) {
            double mass=0, r=0;
            if (!Num(n,"mass",mass) || !Num(n,"r",r))
                return std::string("ERROR: missing mass or r.");
            return "I about the axis = " + F(geom::ThinRingInertiaAboutAxis(mass,r)) +
                   ", I about a diameter = " + F(geom::ThinRingInertiaAboutDiameter(mass,r));
        });

    // ---- the fields --------------------------------------------------------
    add("field_properties",
        "The field and thermal properties of a named material, and the derived times: the magnetic diffusion time tau = mu*sigma*L^2, the charge relaxation time, and the radiative power per area at a temperature. Use this to decide whether a part's material matters electromagnetically.",
        kSchemaMaterial,
        [](const std::map<std::string,double>& n, const std::map<std::string,std::string>& s) {
            const std::string mat = Str(s, "material", "steel");
            double length = 0.01, temp = 293.15;
            Num(n, "length", length);
            Num(n, "temperature_kelvin", temp);
            // The engine's own registry: the same source the simulation uses.
            FieldProperties p;
            if (!LookupFieldProperties(mat, p)) {
                // Try a lowercase form before giving up.
                std::string low = mat;
                std::transform(low.begin(), low.end(), low.begin(),
                               [](unsigned char c){ return (char)std::tolower(c); });
                if (!LookupFieldProperties(low, p)) {
                    return "ERROR: unknown material '" + mat +
                           "'. Use one of the names the engine's registry knows "
                           "(steel, aluminium, copper, iron, bone, cortical bone, ...).";
                }
            }
            std::ostringstream o;
            o << mat << ": mu_r = " << F(p.permeability, 4)
              << ", epsilon_r = " << F(p.permittivity, 4)
              << ", sigma = " << F(p.conductivity, 4) << " S/m"
              << ", emissivity = " << F(p.emissivity, 3)
              << ". Magnetic diffusion time at L = " << F(length) << " m: "
              << F(geom::FieldMagneticDiffusionTime(p, length)) << " s."
              << " Charge relaxation time: " << F(geom::FieldChargeRelaxationTime(p)) << " s."
              << " Radiative power per area at " << F(temp) << " K: "
              << F(geom::FieldRadiativePowerPerArea(p, temp)) << " W/m^2.";
            return o.str();
        });
}

std::string GeoToolSet::ToOllamaToolsJson() const {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < tools_.size(); ++i) {
        if (i) o << ",";
        o << "{\"type\":\"function\",\"function\":{"
          << "\"name\":\"" << tools_[i].name << "\","
          << "\"description\":\"" << tools_[i].description << "\","
          << "\"parameters\":" << tools_[i].parameters_json << "}}";
    }
    o << "]";
    return o.str();
}

bool GeoToolSet::Invoke(const std::string& name,
                        const std::map<std::string, double>& nums,
                        const std::map<std::string, std::string>& strs,
                        std::string& out, std::string& error) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end()) {
        error = "no such tool: " + name;
        return false;
    }
    try {
        out = tools_[it->second].handler(nums, strs);
    } catch (const std::exception& e) {
        error = std::string("the tool threw: ") + e.what();
        return false;
    }
    // -----------------------------------------------------------------------------
    // A HANDLER'S ERROR IS A FAILURE, NOT AN ANSWER.
    // -----------------------------------------------------------------------------
    //
    // The handlers report a missing argument by RETURNING a string that starts
    // with "ERROR:". Without this check the registry treated that as a
    // SUCCESSFUL result -- the loop logged it as `failed = false`, and the
    // model had no signal that anything went wrong.
    //
    // MEASURED: the first live run called `resolve_torus_radii` with
    // `outer_radius`/`tube_radius` three times. Every call returned an
    // error string, every call was recorded as SUCCESSFUL, and the model
    // simply guessed the answer instead. The tool was deployed, understood,
    // and useless.
    //
    // So an error string is now detected, reported as a failure, AND
    // REPLACED with a message that names the accepted argument names and
    // shows what was actually received -- so the model can correct itself on
    // the next turn instead of repeating the same call.
    if (out.rfind("ERROR:", 0) == 0) {
        error = out;
        out = DescribeArgumentMismatch(name, nums, strs);
        return false;
    }
    return true;
}

// The self-correcting message.
//
// It answers the two questions a model needs to fix its call:
//   * what names does this tool accept?
//   * what did I actually send?
std::string GeoToolSet::DescribeArgumentMismatch(
        const std::string& name,
        const std::map<std::string, double>& nums,
        const std::map<std::string, std::string>& strs) const {
    auto it = by_name_.find(name);
    std::ostringstream o;
    o << "ERROR: the arguments did not match. ";
    if (it != by_name_.end()) {
        // The canonical names, read out of the schema's "required" list.
        // The schema is a JSON string; the simplest robust way to name the
        // accepted keys is to list the ones the ALIAS TABLE maps, since the
        // canonical name is always first in that list.
        static const std::map<std::string, std::vector<std::string>> kCanonical = {
            { "volume_of_sphere", { "r" } },
            { "volume_of_ellipsoid", { "sx", "sy", "sz" } },
            { "volume_of_torus", { "major", "minor" } },
            { "volume_of_cylinder", { "r", "h" } },
            { "surface_area_of_sphere", { "r" } },
            { "surface_area_of_ellipsoid", { "sx", "sy", "sz" } },
            { "surface_area_of_torus", { "major", "minor" } },
            { "surface_area_of_cylinder", { "r", "h" } },
            { "resolve_torus_radii", { "outer", "tube" } },
            { "fit_sphere_to_mass", { "mass", "density" } },
            { "closed_body_rule", { "sx", "sy", "sz" } },
            { "inertia_of_sphere", { "mass", "r" } },
            { "inertia_of_cylinder", { "mass", "r", "h" } },
            { "inertia_of_ring", { "mass", "r" } },
            { "principal_moments", { "mass", "r" } },
            { "field_properties", { "material", "length", "temperature_kelvin" } },
        };
        auto c = kCanonical.find(name);
        if (c != kCanonical.end()) {
            o << "This tool accepts the argument name";
            if (c->second.size() > 1) o << "s";
            o << ": ";
            for (size_t i = 0; i < c->second.size(); ++i) {
                if (i) o << ", ";
                o << "\"" << c->second[i] << "\"";
            }
            o << ". ";
        }
    }
    o << "You sent: ";
    bool any = false;
    for (const auto& kv : nums) {
        if (any) o << ", ";
        o << "\"" << kv.first << "\"=" << F(kv.second);
        any = true;
    }
    for (const auto& kv : strs) {
        if (any) o << ", ";
        o << "\"" << kv.first << "\"=\"" << kv.second << "\"";
        any = true;
    }
    if (!any) o << "(nothing)";
    o << ". Call the tool again with the accepted names.";
    return o.str();
}

std::string GeoToolSet::Catalogue() const {
    std::ostringstream o;
    for (size_t i = 0; i < tools_.size(); ++i) {
        o << "- " << tools_[i].name << ": " << tools_[i].description << "\n";
    }
    return o.str();
}

}  // namespace AI
}  // namespace DeliRium
