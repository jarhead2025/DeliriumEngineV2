// =============================================================================
// ElementTreeParser.h - JSON element tree -> Element (the AI -> geometry seam).
//
// WHY THIS EXISTS: the element-system baker (ai/ElementSystem.h) already turns
// an arbitrary recursive element tree into valid, live 64-byte PSF records.
// What was missing was the step BEFORE it: turning something a MODEL wrote
// into an Element tree. A local LLM answers with JSON; this parser validates
// that JSON against the baker's actual vocabulary and builds the tree.
//
// THE CONTRACT:
//   input  : a JSON object (or a JSON array of objects) describing a tree
//   output : an Element tree, ready for ElementSystem::Bake
//   failure: a human-readable reason. NEVER a partially-built tree -- the
//            parser is all-or-nothing so a malformed model response can
//            never poison the scene.
//
// THE SCHEMA (every key optional except where noted):
//   {
//     "kind":     "body",                  // free-form category tag
//     "origin":   [x, y, z],               // local offset
//     "scale":    [x, y, z],               // local scale (default 1)
//     "weight":   1.0,                     // contribution at an overlap
//     "color":    [r, g, b],               // 0..1 (default 0.8)
//     "substance": "ellipsoid",            // leaf material (see below)
//     "substance_params": { ... },         // per-substance parameters
//     "presence": { "radius": 0.5 },       // optional spatial mask
//     "children": [ ... ]                  // recursive
//   }
//
// SUBSTANCES (the leaf materials the baker knows):
//   "plane"      -- z = 0, infinite domain
//   "ellipsoid"  -- upper hemisphere, params {sx, sy, sz} (default 0.5)
//   "gaussian"   -- localized bump, params {amp, sigma} (default 0.2, 0.35)
//   "heightfield"-- analytic field, params {amp, freq_x, freq_y, phase}
//   "wave"       -- sugar for heightfield with amp/freq
//   "sphere"     -- sugar for ellipsoid with sx=sy=sz
//
// The parser deliberately REJECTS an unknown substance rather than silently
// falling back to a plane: a model that invented "dragon_scale" must be
// told, not quietly given a flat card.
//
// NO THIRD-PARTY JSON DEPENDENCY: the engine's JSON handling is already
// nlohmann/json, but it is an OPTIONAL system package (CMake only wires it
// when found). This parser therefore ships a small, strict JSON reader of
// its own so the AI path works on a machine without nlohmann. It is a
// SUBSET reader -- enough for the schema above, and it rejects trailing
// garbage -- not a general JSON library.
// =============================================================================
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "ElementSystem.h"

namespace DeliRium {
namespace AI {

// A minimal JSON value. Only what the element schema needs.
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool   boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<JsonValue> items;
    std::vector<std::pair<std::string, JsonValue>> members;

    bool IsNull()   const { return type == Type::Null; }
    bool IsBool()   const { return type == Type::Bool; }
    bool IsNumber() const { return type == Type::Number; }
    bool IsString() const { return type == Type::String; }
    bool IsArray()  const { return type == Type::Array; }
    bool IsObject() const { return type == Type::Object; }

    // Object member lookup (nullptr if absent).
    const JsonValue* Find(const std::string& key) const;
    // Convenience accessors with defaults.
    double GetNumber(double fallback) const { return IsNumber() ? number : fallback; }
    std::string GetString(const std::string& fallback) const {
        return IsString() ? text : fallback;
    }
};

// Parse a JSON document. Returns false with `why` on any malformed input.
// `allow_trailing` accepts trailing whitespace only (never trailing data).
bool ParseJson(const std::string& src, JsonValue& out, std::string& why);

// Find the FIRST balanced {...} block in `src` (ignoring braces inside
// strings). A model often wraps its JSON in prose or a markdown fence; this
// extracts the object without requiring a perfectly clean response.
bool ExtractJsonObject(const std::string& src, std::string& out);

// Parse a JSON element tree. On success `out` owns the tree. On failure
// `why` explains why and `out` is untouched.
bool ParseElementTree(const std::string& json, std::unique_ptr<Element>& out,
                      std::string& why);

// The system prompt that teaches a local model the schema above. Kept here
// (not in the workspace) so the contract has exactly one definition.
std::string ElementTreeSystemPrompt();

// A short human-readable summary of a tree ("3 systems, 2 leaves"), for the
// console panel.
std::string DescribeTree(const Element& root);

} // namespace AI
} // namespace DeliRium
