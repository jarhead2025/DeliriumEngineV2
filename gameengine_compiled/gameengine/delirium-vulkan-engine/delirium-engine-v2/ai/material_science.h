#pragma once
/* ── Material Science Registry ───────────────────────────────────────
 *
 * Extended material system that bridges organic and inorganic domains.
 * Each material defines physical properties (Young's modulus, density,
 * etc.) that drive both physics simulation and AI generation decisions.
 *
 * Organic materials:   tissue, cartilage, bone, chitin, cellulose, keratin
 * Inorganic materials: steel, aluminum, carbon_fiber, ceramic, glass, etc.
 *
 * Materials also define "crafting rules" — how the AI should approach
 * building structures from this material (growth patterns, assembly
 * methods, deformation behavior).
 * ──────────────────────────────────────────────────────────────────── */

#include "core/types.h"
#include "core/material_registry.h"
#include "ai/domain_classifier.h"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace delirium {

/* ── Crafting method ──────────────────────────────────────────────── */
enum class CraftingMethod : std::uint8_t {
    Growth,        // biological growth pattern (organic)
    Assembly,      // bolted/welded parts (inorganic)
    Casting,       // poured into mold (inorganic)
    Forging,       // shaped by force (inorganic)
    Weaving,       // fiber interlocking (both)
    Layering,      // deposited layer by layer (both)
    Fusion,        // melted and joined (inorganic)
    Regeneration,  // self-healing/regrowth (organic)
    Molding,       // shaped in mold (inorganic)
};

/* ── Deformation behavior ─────────────────────────────────────────── */
enum class DeformationBehavior : std::uint8_t {
    Elastic,       // returns to original shape
    Plastic,       // permanent deformation
    Viscoelastic,  // time-dependent (rubber, tissue)
    Brittle,       // fractures before yielding
    Fluid,         // flows under stress
    Hyperelastic,  // large elastic strains (biological tissue)
};

/* ── Extended material definition ─────────────────────────────────── */
struct MaterialDefinition {
    std::string name;
    Domain domain;
    CraftingMethod crafting;
    DeformationBehavior deformation;

    // Physical properties (SI units)
    MaterialProperties physics;

    // Generation hints
    float typical_thickness = 0.1f;   // meters, for patch sizing
    float min_curvature = 0.0f;       // 1/meters, how tightly it can bend
    float max_strain = 0.02f;         // yield strain
    float anisotropy = 0.0f;          // 0=isotropic, 1=fully directional

    // Visual properties
    float translucency = 0.0f;
    float subsurface_scatter = 0.0f;
    float iridescence = 0.0f;
};

/* ── Material science registry ────────────────────────────────────── */
class MaterialScienceRegistry {
public:
    /* Get or create a material by name.
     * Returns a registered MaterialId and populates the physics registry. */
    MaterialId get_material(std::string_view name, MaterialRegistry& phys_reg);

    /* Get the full material definition. */
    const MaterialDefinition& definition(MaterialId id) const;

    /* List all materials in a domain. */
    std::vector<MaterialId> list_domain(Domain d) const;

    /* Resolve a material name from semantic text.
     * Returns the best matching MaterialId. */
    MaterialId resolve_from_text(std::string_view text, MaterialRegistry& phys_reg) const;

    /* Get default material for a domain. */
    MaterialId domain_default(Domain d) const;

private:
    void init_builtin_materials();
    std::unordered_map<std::string, MaterialId> name_to_id_;
    std::vector<MaterialDefinition> definitions_;
    MaterialId next_id_ = 0;
};

} // namespace delirium