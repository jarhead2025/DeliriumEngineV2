// =============================================================================
// GeoTools.h - GEOMETRIC ANALYSIS TOOLS for the local model.
//
// THE IDEA (Josias, 2026-11-12):
//   "try training the model to use geometrical analysis math tools it should
//    help alot more to have them as small pseudo-skills in compartments it
//    can learn to deploy"
//
// WHY THIS EXISTS
//
// The model's geometric output is wrong in SPECIFIC, REPEATABLE ways, and
// the failures are not "the model is bad at geometry" -- they are "the
// model cannot do arithmetic reliably enough to get a shape's PARAMETERS
// right". Measured failures, all from real runs against the local 7B:
//
//   1. the closed-body rule: the model emitted `"sz": -0.5` (a negative
//      HALF-EXTENT) where the rule is `"scale": [1,1,-1]` (a negative
//      SCALE). It confused a size with an orientation.
//   2. the torus's radii: the model had to work out that a torus quoted as
//      "outer radius 0.44, tube radius 0.12" means major = 0.32, and it
//      got it wrong.
//   3. the scale of a part: the model guessed a radius that made the
//      assembly's parts the wrong relative size.
//   4. `"substance": ""` on grouping nodes -- a construct the schema
//      already had, under a different spelling.
//
// Every one of those is arithmetic or a vocabulary lookup. A 7B model
// cannot reliably do arithmetic in its head; it CAN reliably CALL a
// function that does it.
//
// WHAT THIS IS
//
// A TOOL SET the model can invoke mid-reasoning, over Ollama's native
// tool-calling (MEASURED: the local qwen2.5:7b emits a proper
// `tool_calls` block with structured arguments -- `{"name": ..., 
// "arguments": {...}}` -- so no prompt-parsing hack is needed).
//
// The tools are NOT prompt instructions dressed as code. Each one is a
// REAL function over the engine's own mathematics:
//
//   volume_of_sphere / volume_of_ellipsoid / volume_of_torus /
//   volume_of_cylinder   -- the exact closed forms
//   surface_area_*        -- the exact closed forms
//   inertia_*             -- over the engine's own MassProperties
//   principal_moments     -- over the engine's own MassProperties
//   field_*               -- over the engine's own FieldProperties
//   fit_sphere_to_mass   -- the arithmetic the model gets wrong
//   resolve_torus_radii  -- the arithmetic the model gets wrong
//   closed_body_rule     -- the vocabulary lookup the model gets wrong
//
// WHY "PSEUDO-SKILLS IN COMPARTMENTS"
//
// Each tool is a COMPARTMENT: a small, named, self-describing unit with
// its own description and its own parameter schema. The model sees the
// compartment list, picks the ones it needs, and calls them. That is the
// same structure as a skill -- trigger conditions, a narrow procedure, a
// verifiable output -- except the "procedure" is executed rather than
// read.
//
// The compartment boundary matters: a tool that answers ONE question with
// ONE number is learnable. A tool that "does geometry" is not.
//
// THE CONTRACT
//
//   * a tool NEVER returns prose the model must parse -- it returns
//     numbers, and the model's next move is to use them;
//   * a tool NEVER mutates the scene -- they are PURE;
//   * a tool's result is echoed back into the conversation, so the model
//     can reason over it;
//   * a tool that cannot answer says so, with the reason.
//
// WHY THE TOOLS ARE IN THE ENGINE AND NOT IN THE PROMPT
//
// A prompt can say "remember that a torus's major radius is the outer
// radius minus the tube radius". A 7B will not reliably remember it, and
// it will not reliably apply it. A tool computes it. The difference is
// the difference between an instruction and a capability.
// =============================================================================
#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>

#include "delirium/core/Types.h"
#include "delirium/physics/Inertia.h"
#include "delirium/physics/FieldProperties.h"

namespace DeliRium {
namespace AI {

// -----------------------------------------------------------------------------
// A TOOL: a named compartment with a schema and a handler.
//
// The schema is deliberately in the SHAPE OLLAMA'S TOOL-CALLING EXPECTS
// (a JSON Schema object), because the client serialises it straight into
// the request. Keeping it here (rather than building it at request time)
// means the tool's contract is written once and lives next to its code.
// -----------------------------------------------------------------------------
struct GeoTool {
    std::string name;
    std::string description;
    // The JSON Schema for the arguments, as a JSON string. Kept as a
    // string because it is passed through verbatim, and because the
    // engine's own JSON writer is not available to every caller.
    std::string parameters_json;
    // The handler: the arguments (already parsed into a flat map of
    // numbers and strings) in, a result out. The result is a SHORT
    // human-readable string -- the model reads it and reasons over it.
    //
    // Returning a string (rather than a struct) is deliberate: the result
    // goes straight back into the conversation as the tool message's
    // content, and the model needs to READ it. A JSON blob would be
    // parsed; a short sentence with the numbers in it is easier for a
    // small model to use.
    using handler_t = std::function<std::string(const std::map<std::string, double>&,
                                                const std::map<std::string, std::string>&)>;
    handler_t handler;
};

// -----------------------------------------------------------------------------
// The registry.
//
// ONE registry, in the engine, so the tools are available to every path
// that talks to the model: the workspace's semantic prompt, the document
// learner, and the escalation ladder. A tool set that only exists in one
// path is a tool set that gets forgotten.
// -----------------------------------------------------------------------------
class GeoToolSet {
public:
    static GeoToolSet& Instance();

    // The tools, in the order the model sees them.
    const std::vector<GeoTool>& Tools() const { return tools_; }

    // The JSON array of tool specifications, ready for Ollama's request.
    std::string ToOllamaToolsJson() const;

    // Invoke a tool by name. Returns false if the name is unknown (and
    // sets `error` to say so -- the model should be told, not left
    // hanging).
    //
    // A tool whose ARGUMENTS do not match ALSO returns false: the handler
    // reports the mismatch by returning an "ERROR:" string, and the
    // registry converts that into a failure whose message names the
    // ACCEPTED argument names and echoes what was actually received -- so
    // a small model can correct itself on the next turn instead of
    // repeating the same failing call.
    bool Invoke(const std::string& name,
                const std::map<std::string, double>& nums,
                const std::map<std::string, std::string>& strs,
                std::string& out,
                std::string& error) const;

    // The self-correcting message for an argument mismatch.
    std::string DescribeArgumentMismatch(
            const std::string& name,
            const std::map<std::string, double>& nums,
            const std::map<std::string, std::string>& strs) const;

    // The catalogue the model is told about, as a compact list -- the
    // "compartments" it can choose from.
    std::string Catalogue() const;

private:
    GeoToolSet();
    std::vector<GeoTool> tools_;
    std::map<std::string, size_t> by_name_;
};

// -----------------------------------------------------------------------------
// THE GEOMETRY, as pure functions.
//
// These are the engine's OWN mathematics, exposed as the tools' bodies.
// They are separated from the registry so they can be unit-tested
// directly -- a tool whose maths is wrong is worse than no tool, because
// the model TRUSTS it.
// -----------------------------------------------------------------------------
namespace geom {

// The exact closed forms. (A "sphere" is an ellipsoid with equal axes.)
double SphereVolume(double r);
double EllipsoidVolume(double sx, double sy, double sz);
double TorusVolume(double major, double minor);
double CylinderVolume(double r, double h);
double SphereSurfaceArea(double r);
double EllipsoidSurfaceArea(double sx, double sy, double sz);  // the exact form
double TorusSurfaceArea(double major, double minor);
double CylinderSurfaceArea(double r, double h);

// The arithmetic the model gets wrong.
//
// A torus quoted by its OUTER radius (what people and documents mean by
// "a ring of radius 0.44 with a tube of 0.12") has major = outer - minor.
// The model gets this wrong by using outer as major.
//
// A closed body's lower half is the SAME sheet with the SCALE negated, at
// the SAME origin -- NOT a negative half-extent. The model gets this
// wrong by negating the size.
struct TorusRadii { double major; double minor; };
TorusRadii ResolveTorusRadii(double outer, double tube);
struct SphereFromMass { double radius; double volume; };
SphereFromMass FitSphereToMass(double mass, double density);

// The engine's own physics, exposed.
//
// The inertia of a solid sphere/cylinder/torus about its centre -- the
// exact closed forms, which the engine's MassProperties can then be
// seeded with so the WHOLE pipeline (bones, anchors, fields) stays
// consistent.
double SolidSphereInertia(double mass, double r);
double SolidCylinderInertiaAboutAxis(double mass, double r, double h);
double SolidCylinderInertiaAboutDiameter(double mass, double r, double h);
double ThinRingInertiaAboutAxis(double mass, double r);
double ThinRingInertiaAboutDiameter(double mass, double r);

// The field relations, over the engine's FieldProperties.
double FieldMagneticDiffusionTime(const FieldProperties& p, double length);
double FieldChargeRelaxationTime(const FieldProperties& p);
double FieldRadiativePowerPerArea(const FieldProperties& p, double temperature_kelvin);

}  // namespace geom
}  // namespace AI
}  // namespace DeliRium
