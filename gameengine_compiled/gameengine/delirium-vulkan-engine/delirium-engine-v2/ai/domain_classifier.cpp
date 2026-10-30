#include "domain_classifier.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace delirium {

namespace {

bool is_alpha(char c) { return std::isalpha(static_cast<unsigned char>(c)); }
bool is_alnum(char c) { return std::isalnum(static_cast<unsigned char>(c)); }

std::string to_lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return out;
}

/* Tokenize on whitespace/punctuation, keep alphanumeric runs. */
std::vector<std::string> tokenize(std::string_view s) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : s) {
        if (is_alnum(c)) cur += std::tolower(c);
        else {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

/* Split on "with", "and", "plus", "has", "having" for multi-part parsing. */
std::vector<std::string> split_parts(std::string_view s) {
    static const char* seps[] = {" with ", " and ", " plus ", " has ", " having "};
    std::string str(s);
    std::vector<std::string> parts;
    std::string cur = str;
    bool found = true;
    while (found) {
        found = false;
        std::string best;
        std::size_t best_pos = std::string::npos;
        for (const char* sep : seps) {
            auto pos = cur.find(sep);
            if (pos != std::string::npos && (best_pos == std::string::npos || pos < best_pos)) {
                best = sep;
                best_pos = pos;
            }
        }
        if (best_pos != std::string::npos) {
            parts.push_back(cur.substr(0, best_pos));
            cur = cur.substr(best_pos + best.size());
            found = true;
        }
    }
    parts.push_back(cur);
    return parts;
}

} // anonymous namespace

/* ── Builtin keywords ─────────────────────────────────────────────── */
void DomainClassifier::init_builtin_keywords() {
    // Organic keywords (biological, living structures)
    auto reg = [this](std::string_view w, Domain d, float c) {
        register_keyword(w, d, c);
    };

    // Core organic terms
    reg("organism", Domain::Organic, 1.0f);
    reg("animal", Domain::Organic, 1.0f);
    reg("plant", Domain::Organic, 1.0f);
    reg("creature", Domain::Organic, 1.0f);
    reg("body", Domain::Organic, 0.9f);
    reg("limb", Domain::Organic, 0.9f);
    reg("bone", Domain::Organic, 0.85f);
    reg("muscle", Domain::Organic, 0.9f);
    reg("skin", Domain::Organic, 0.85f);
    reg("organ", Domain::Organic, 0.9f);
    reg("cell", Domain::Organic, 0.9f);
    reg("tissue", Domain::Organic, 0.9f);
    reg("cartilage", Domain::Organic, 0.9f);
    reg("ligament", Domain::Organic, 0.9f);
    reg("tendon", Domain::Organic, 0.9f);
    reg("leaf", Domain::Organic, 0.9f);
    reg("root", Domain::Organic, 0.7f);
    reg("stem", Domain::Organic, 0.7f);
    reg("flower", Domain::Organic, 0.9f);
    reg("seed", Domain::Organic, 0.9f);
    reg("spore", Domain::Organic, 0.9f);
    reg("fungus", Domain::Organic, 0.9f);
    reg("mushroom", Domain::Organic, 0.9f);
    reg("algae", Domain::Organic, 0.9f);
    reg("bacteria", Domain::Organic, 0.9f);
    reg("virus", Domain::Organic, 0.8f);
    reg("dna", Domain::Organic, 0.9f);
    reg("protein", Domain::Organic, 0.85f);
    reg("enzyme", Domain::Organic, 0.85f);
    reg("blood", Domain::Organic, 0.9f);
    reg("nerve", Domain::Organic, 0.9f);
    reg("brain", Domain::Organic, 0.9f);
    reg("heart", Domain::Organic, 0.9f);
    reg("lung", Domain::Organic, 0.9f);
    reg("liver", Domain::Organic, 0.9f);
    reg("kidney", Domain::Organic, 0.9f);
    reg("eye", Domain::Organic, 0.85f);
    reg("wing", Domain::Organic, 0.85f);
    reg("fin", Domain::Organic, 0.85f);
    reg("tail", Domain::Organic, 0.8f);
    reg("head", Domain::Organic, 0.75f);
    reg("torso", Domain::Organic, 0.85f);
    reg("thorax", Domain::Organic, 0.9f);
    reg("abdomen", Domain::Organic, 0.9f);
    reg("skull", Domain::Organic, 0.85f);
    reg("jaw", Domain::Organic, 0.85f);
    reg("tooth", Domain::Organic, 0.8f);
    reg("claw", Domain::Organic, 0.85f);
    reg("scale", Domain::Organic, 0.7f);
    reg("feather", Domain::Organic, 0.85f);
    reg("fur", Domain::Organic, 0.85f);
    reg("antenna", Domain::Organic, 0.85f);
    reg("mandible", Domain::Organic, 0.9f);
    reg("chitin", Domain::Organic, 0.9f);
    reg("exoskeleton", Domain::Organic, 0.9f);
    reg("endoskeleton", Domain::Organic, 0.9f);
    reg("growth", Domain::Organic, 0.8f);
    reg("evolve", Domain::Organic, 0.8f);
    reg("mutate", Domain::Organic, 0.85f);
    reg("reproduce", Domain::Organic, 0.85f);
    reg("biomass", Domain::Organic, 0.85f);
    reg("cellulose", Domain::Organic, 0.9f);
    reg("keratin", Domain::Organic, 0.9f);
    reg("collagen", Domain::Organic, 0.9f);
    reg("melanin", Domain::Organic, 0.85f);
    reg("hemoglobin", Domain::Organic, 0.9f);
    reg("chlorophyll", Domain::Organic, 0.9f);

    // Core inorganic terms
    reg("metal", Domain::Inorganic, 0.9f);
    reg("steel", Domain::Inorganic, 0.9f);
    reg("iron", Domain::Inorganic, 0.9f);
    reg("copper", Domain::Inorganic, 0.9f);
    reg("aluminum", Domain::Inorganic, 0.9f);
    reg("gold", Domain::Inorganic, 0.9f);
    reg("silver", Domain::Inorganic, 0.9f);
    reg("tungsten", Domain::Inorganic, 0.9f);
    reg("titanium", Domain::Inorganic, 0.9f);
    reg("plastic", Domain::Inorganic, 0.85f);
    reg("glass", Domain::Inorganic, 0.85f);
    reg("ceramic", Domain::Inorganic, 0.85f);
    reg("concrete", Domain::Inorganic, 0.85f);
    reg("stone", Domain::Inorganic, 0.85f);
    reg("rock", Domain::Inorganic, 0.85f);
    reg("crystal", Domain::Inorganic, 0.85f);
    reg("mineral", Domain::Inorganic, 0.85f);
    reg("gear", Domain::Inorganic, 0.9f);
    reg("wheel", Domain::Inorganic, 0.9f);
    reg("lever", Domain::Inorganic, 0.9f);
    reg("spring", Domain::Inorganic, 0.8f);
    reg("hinge", Domain::Inorganic, 0.9f);
    reg("bolt", Domain::Inorganic, 0.9f);
    reg("screw", Domain::Inorganic, 0.9f);
    reg("beam", Domain::Inorganic, 0.8f);
    reg("plate", Domain::Inorganic, 0.75f);
    reg("shell", Domain::Inorganic, 0.7f);
    reg("frame", Domain::Inorganic, 0.85f);
    reg("structure", Domain::Inorganic, 0.75f);
    reg("machine", Domain::Inorganic, 0.9f);
    reg("engine", Domain::Inorganic, 0.85f);
    reg("motor", Domain::Inorganic, 0.9f);
    reg("piston", Domain::Inorganic, 0.9f);
    reg("blade", Domain::Inorganic, 0.8f);
    reg("shaft", Domain::Inorganic, 0.8f);
    reg("axle", Domain::Inorganic, 0.9f);
    reg("circuit", Domain::Inorganic, 0.9f);
    reg("conductor", Domain::Inorganic, 0.85f);
    reg("insulator", Domain::Inorganic, 0.85f);
    reg("semiconductor", Domain::Inorganic, 0.9f);
    reg("alloy", Domain::Inorganic, 0.9f);
    reg("composite", Domain::Inorganic, 0.85f);
    reg("polymer", Domain::Inorganic, 0.85f);
    reg("resin", Domain::Inorganic, 0.85f);
    reg("carbon", Domain::Inorganic, 0.75f);
    reg("graphite", Domain::Inorganic, 0.85f);
    reg("diamond", Domain::Inorganic, 0.85f);
    reg("silicon", Domain::Inorganic, 0.85f);
    reg("sand", Domain::Inorganic, 0.8f);
    reg("clay", Domain::Inorganic, 0.85f);
    reg("brick", Domain::Inorganic, 0.85f);
    reg("mortar", Domain::Inorganic, 0.85f);
    reg("bridge", Domain::Inorganic, 0.85f);
    reg("tower", Domain::Inorganic, 0.85f);
    reg("wall", Domain::Inorganic, 0.8f);
    reg("foundation", Domain::Inorganic, 0.85f);
    reg("blueprint", Domain::Inorganic, 0.9f);
    reg("assembly", Domain::Inorganic, 0.85f);
    reg("fabrication", Domain::Inorganic, 0.9f);
    reg("welding", Domain::Inorganic, 0.9f);
    reg("casting", Domain::Inorganic, 0.9f);
    reg("forging", Domain::Inorganic, 0.9f);
    reg("machining", Domain::Inorganic, 0.9f);
    reg("extrusion", Domain::Inorganic, 0.9f);
    reg("molding", Domain::Inorganic, 0.85f);
}

/* ── Public API ───────────────────────────────────────────────────── */
void DomainClassifier::register_keyword(std::string_view word, Domain d, float confidence) {
    keywords_[to_lower(word)] = {d, confidence};
}

Domain DomainClassifier::classify(std::string_view semantic) const {
    if (keywords_.empty()) {
        // Lazy init — not thread-safe but fine for single-threaded demo
        const_cast<DomainClassifier*>(this)->init_builtin_keywords();
    }

    auto tokens = tokenize(semantic);
    float organic_score = 0, inorganic_score = 0;

    for (const auto& tok : tokens) {
        auto it = keywords_.find(tok);
        if (it != keywords_.end()) {
            if (it->second.first == Domain::Organic)
                organic_score += it->second.second;
            else
                inorganic_score += it->second.second;
        }
    }

    // Default: if no keywords matched, lean organic (more common in creative contexts)
    if (organic_score == 0 && inorganic_score == 0)
        return Domain::Organic;

    return organic_score >= inorganic_score ? Domain::Organic : Domain::Inorganic;
}

ParsedSemantic DomainClassifier::parse(std::string_view semantic) const {
    if (keywords_.empty()) {
        const_cast<DomainClassifier*>(this)->init_builtin_keywords();
    }

    ParsedSemantic parsed;
    parsed.raw_input = std::string(semantic);
    parsed.primary_domain = classify(semantic);
    parsed.complex_assembly = false;

    // Tokenize and tag
    auto tokens = tokenize(semantic);
    for (const auto& tok : tokens) {
        SemanticTag tag;
        tag.word = tok;
        tag.domain = Domain::Organic; // default
        tag.confidence = 0.3f;

        auto it = keywords_.find(tok);
        if (it != keywords_.end()) {
            tag.domain = it->second.first;
            tag.confidence = it->second.second;
        }
        parsed.tags.push_back(tag);
    }

    // Split into parts (handles "dragon with wings and tail")
    auto parts = split_parts(semantic);
    parsed.complex_assembly = parts.size() > 1;

    for (std::size_t i = 0; i < parts.size(); ++i) {
        PartDescription part;
        part.domain = classify(parts[i]);
        part.name = std::string(parts[i]);

        // Trim whitespace
        while (!part.name.empty() && std::isspace(static_cast<unsigned char>(part.name.front())))
            part.name.erase(part.name.begin());
        while (!part.name.empty() && std::isspace(static_cast<unsigned char>(part.name.back())))
            part.name.pop_back();

        // Infer material from keywords in this part
        auto part_tokens = tokenize(parts[i]);
        for (const auto& tok : part_tokens) {
            auto it = keywords_.find(tok);
            if (it != keywords_.end() && it->second.second > 0.8f) {
                part.material = tok;
                break;
            }
        }
        if (part.material.empty()) {
            part.material = part.domain == Domain::Organic ? "tissue" : "steel";
        }

        parsed.parts.push_back(part);
    }

    return parsed;
}

} // namespace delirium