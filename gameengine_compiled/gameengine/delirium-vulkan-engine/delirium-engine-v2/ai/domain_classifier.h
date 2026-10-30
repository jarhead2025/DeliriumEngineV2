#pragma once
/* ── AI Domain Classification ────────────────────────────────────────
 *
 * Every semantic input is classified as ORGANIC or INORGANIC.
 * This split determines which framework interprets and generates the
 * mathematical representation.
 *
 * Organic:   living structures, biological materials, growth patterns
 * Inorganic: engineered structures, geological formations, mechanical parts
 *
 * The model learns to "craft" parts through math tools — reproducing
 * the assembly of elements where math is the boundary (no molecular
 * structure, only material science and blueprints).
 * ──────────────────────────────────────────────────────────────────── */

#include "core/types.h"
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace delirium {

/* ── Domain ────────────────────────────────────────────────────────── */
enum class Domain : std::uint8_t {
    Organic   = 0,  // biological, living, anatomical
    Inorganic = 1,  // engineered, geological, mechanical
};

/* ── Semantic tag ──────────────────────────────────────────────────── */
struct SemanticTag {
    std::string_view word;    // original token
    Domain           domain;  // classified domain
    float            confidence; // 0..1 classification confidence
};

/* ── Part description ──────────────────────────────────────────────── */
struct PartDescription {
    std::string        name;         // "thorax", "gear", "leaf"
    Domain             domain;
    std::string        material;     // "cartilage", "steel", "cellulose"
    Vec3               scale{1,1,1};// relative scale hint
    Vec3               position{};   // assembly position
    std::vector<std::string> children; // child part names
    std::vector<std::string> features; // "articulated", "porous", "rigid"
};

/* ── Parsed semantic ──────────────────────────────────────────────── */
struct ParsedSemantic {
    Domain primary_domain;
    std::vector<PartDescription> parts;
    std::vector<SemanticTag> tags;
    std::string raw_input;
    bool complex_assembly; // needs blueprint system
};

/* ── Domain classifier ────────────────────────────────────────────── */
class DomainClassifier {
public:
    /* Classify a semantic string into organic/inorganic.
     * Returns the dominant domain based on keyword analysis. */
    Domain classify(std::string_view semantic) const;

    /* Parse semantic into structured parts and tags.
     * Handles multi-part descriptions: "dragon with wings and tail" */
    ParsedSemantic parse(std::string_view semantic) const;

    /* Register a custom keyword → domain mapping. */
    void register_keyword(std::string_view word, Domain d, float confidence = 1.0f);

private:
    std::unordered_map<std::string, std::pair<Domain, float>> keywords_;
    void init_builtin_keywords();
};

} // namespace delirium