#pragma once
/* ── AI Generator — Main entry point ─────────────────────────────────
 *
 * Takes a semantic description, classifies it into organic/inorganic,
 * routes to the appropriate framework, and produces a BodyRuntime.
 *
 * Usage:
 *   AIGenerator gen;
 *   auto body = gen.generate("a dragon with wings and tail", mat_science, phys_reg);
 *   auto bridge = gen.generate("steel bridge with arch supports", mat_science, phys_reg);
 *
 * The model learns to interpret "crafting" parts through math tools —
 * reproducing the assembly of elements where the math is as far as it
 * goes without going into molecular structure.
 * ──────────────────────────────────────────────────────────────────── */

#include "core/body_runtime.h"
#include "core/material_registry.h"
#include "ai/domain_classifier.h"
#include "ai/material_science.h"
#include "ai/organic_generator.h"
#include "ai/inorganic_generator.h"
#include <memory>
#include <string>
#include <string_view>

namespace delirium {

class AIGenerator {
public:
    /* Generate a body from a semantic description.
     * Automatically classifies domain and routes to the right framework. */
    std::unique_ptr<BodyRuntime> generate(
        std::string_view semantic,
        MaterialScienceRegistry& mat_science,
        MaterialRegistry& phys_reg);

    /* Generate with explicit domain override. */
    std::unique_ptr<BodyRuntime> generate_as(
        std::string_view semantic,
        Domain domain,
        MaterialScienceRegistry& mat_science,
        MaterialRegistry& phys_reg);

    /* Classify a semantic string. */
    Domain classify(std::string_view semantic) const {
        return classifier_.classify(semantic);
    }

    /* Parse semantic into structured parts. */
    ParsedSemantic parse(std::string_view semantic) const {
        return classifier_.parse(semantic);
    }

    /* Access to sub-generators for direct control. */
    OrganicGenerator& organic() { return organic_gen_; }
    InorganicGenerator& inorganic() { return inorganic_gen_; }
    const OrganicGenerator& organic() const { return organic_gen_; }
    const InorganicGenerator& inorganic() const { return inorganic_gen_; }

private:
    DomainClassifier classifier_;
    OrganicGenerator organic_gen_;
    InorganicGenerator inorganic_gen_;
};

} // namespace delirium