#include "material_science.h"
#include <algorithm>
#include <cctype>

namespace delirium {

namespace {

std::string to_lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return out;
}

} // anonymous namespace

/* ── Builtin materials ────────────────────────────────────────────── */
void MaterialScienceRegistry::init_builtin_materials() {
    auto reg = [this](const MaterialDefinition& def) {
        MaterialId id = next_id_++;
        name_to_id_[def.name] = id;
        definitions_.push_back(def);
        return id;
    };

    // ═══════════════════════════════════════════════════════════════
    // ORGANIC MATERIALS
    // ═══════════════════════════════════════════════════════════════

    // Living tissue (muscle, organ parenchyma)
    { MaterialProperties p;
      p.youngs_modulus = 20e3f; p.poisson_ratio = 0.45f; p.density = 1060.0f;
      p.yield_strength = 1e5f; p.r = 180; p.g = 100; p.b = 100;
      p.roughness = 0.8f;
      reg({ "tissue", Domain::Organic, CraftingMethod::Growth, DeformationBehavior::Hyperelastic,
            p, 0.05f, 5.0f, 0.15f, 0.3f, 0.4f, 0.6f, 0.0f }); }

    // Cartilage (joint surfaces, flexible connective)
    { MaterialProperties p;
      p.youngs_modulus = 1e6f; p.poisson_ratio = 0.45f; p.density = 1100.0f;
      p.yield_strength = 5e5f; p.r = 220; p.g = 210; p.b = 180;
      p.roughness = 0.6f;
      reg({ "cartilage", Domain::Organic, CraftingMethod::Growth, DeformationBehavior::Viscoelastic,
            p, 0.02f, 8.0f, 0.08f, 0.2f, 0.3f, 0.5f, 0.0f }); }

    // Bone (rigid structural element)
    { MaterialProperties p;
      p.youngs_modulus = 17e9f; p.poisson_ratio = 0.3f; p.density = 1900.0f;
      p.yield_strength = 130e6f; p.r = 240; p.g = 230; p.b = 210;
      p.roughness = 0.7f;
      reg({ "bone", Domain::Organic, CraftingMethod::Growth, DeformationBehavior::Brittle,
            p, 0.03f, 3.0f, 0.007f, 0.4f, 0.0f, 0.0f, 0.0f }); }

    // Chitin (exoskeleton, insect shells)
    { MaterialProperties p;
      p.youngs_modulus = 10e9f; p.poisson_ratio = 0.35f; p.density = 1300.0f;
      p.yield_strength = 80e6f; p.r = 100; p.g = 80; p.b = 40;
      p.roughness = 0.5f; p.metalness = 0.1f;
      reg({ "chitin", Domain::Organic, CraftingMethod::Layering, DeformationBehavior::Elastic,
            p, 0.005f, 10.0f, 0.03f, 0.5f, 0.0f, 0.0f, 0.3f }); }

    // Keratin (hair, nails, feathers, claws)
    { MaterialProperties p;
      p.youngs_modulus = 4e9f; p.poisson_ratio = 0.35f; p.density = 1300.0f;
      p.yield_strength = 300e6f; p.r = 60; p.g = 40; p.b = 20;
      p.roughness = 0.6f;
      reg({ "keratin", Domain::Organic, CraftingMethod::Growth, DeformationBehavior::Elastic,
            p, 0.002f, 20.0f, 0.05f, 0.6f, 0.0f, 0.0f, 0.1f }); }

    // Cellulose (plant cell walls, wood)
    { MaterialProperties p;
      p.youngs_modulus = 10e9f; p.poisson_ratio = 0.35f; p.density = 700.0f;
      p.yield_strength = 50e6f; p.r = 120; p.g = 100; p.b = 50;
      p.roughness = 0.9f;
      reg({ "cellulose", Domain::Organic, CraftingMethod::Growth, DeformationBehavior::Elastic,
            p, 0.1f, 2.0f, 0.04f, 0.7f, 0.0f, 0.0f, 0.0f }); }

    // Collagen (tendons, ligaments, skin structure)
    { MaterialProperties p;
      p.youngs_modulus = 500e6f; p.poisson_ratio = 0.45f; p.density = 1200.0f;
      p.yield_strength = 20e6f; p.r = 240; p.g = 220; p.b = 200;
      p.roughness = 0.85f;
      reg({ "collagen", Domain::Organic, CraftingMethod::Weaving, DeformationBehavior::Viscoelastic,
            p, 0.01f, 15.0f, 0.1f, 0.8f, 0.2f, 0.3f, 0.0f }); }

    // ═══════════════════════════════════════════════════════════════
    // INORGANIC MATERIALS
    // ═══════════════════════════════════════════════════════════════

    // Steel (structural metal)
    { MaterialProperties p;
      p.youngs_modulus = 200e9f; p.poisson_ratio = 0.3f; p.density = 7850.0f;
      p.yield_strength = 250e6f; p.r = 160; p.g = 160; p.b = 170;
      p.roughness = 0.4f; p.metalness = 0.8f;
      reg({ "steel", Domain::Inorganic, CraftingMethod::Forging, DeformationBehavior::Plastic,
            p, 0.01f, 50.0f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f }); }

    // Aluminum (lightweight metal)
    { MaterialProperties p;
      p.youngs_modulus = 70e9f; p.poisson_ratio = 0.33f; p.density = 2700.0f;
      p.yield_strength = 95e6f; p.r = 190; p.g = 195; p.b = 200;
      p.roughness = 0.3f; p.metalness = 0.7f;
      reg({ "aluminum", Domain::Inorganic, CraftingMethod::Casting, DeformationBehavior::Plastic,
            p, 0.005f, 100.0f, 0.015f, 0.0f, 0.0f, 0.0f, 0.0f }); }

    // Carbon fiber composite
    { MaterialProperties p;
      p.youngs_modulus = 150e9f; p.poisson_ratio = 0.3f; p.density = 1600.0f;
      p.yield_strength = 1500e6f; p.r = 40; p.g = 40; p.b = 45;
      p.roughness = 0.2f; p.metalness = 0.3f;
      reg({ "carbon_fiber", Domain::Inorganic, CraftingMethod::Layering, DeformationBehavior::Brittle,
            p, 0.003f, 200.0f, 0.005f, 0.9f, 0.0f, 0.0f, 0.2f }); }

    // Ceramic (hard, brittle)
    { MaterialProperties p;
      p.youngs_modulus = 300e9f; p.poisson_ratio = 0.25f; p.density = 2500.0f;
      p.yield_strength = 300e6f; p.r = 220; p.g = 220; p.b = 225;
      p.roughness = 0.15f;
      reg({ "ceramic", Domain::Inorganic, CraftingMethod::Casting, DeformationBehavior::Brittle,
            p, 0.01f, 500.0f, 0.002f, 0.0f, 0.0f, 0.0f, 0.0f }); }

    // Glass (transparent brittle)
    { MaterialProperties p;
      p.youngs_modulus = 72e9f; p.poisson_ratio = 0.2f; p.density = 2500.0f;
      p.yield_strength = 33e6f; p.r = 200; p.g = 220; p.b = 240;
      p.roughness = 0.05f;
      reg({ "glass", Domain::Inorganic, CraftingMethod::Casting, DeformationBehavior::Brittle,
            p, 0.005f, 1000.0f, 0.001f, 0.0f, 0.9f, 0.0f, 0.5f }); }

    // Concrete (construction material)
    { MaterialProperties p;
      p.youngs_modulus = 30e9f; p.poisson_ratio = 0.2f; p.density = 2400.0f;
      p.yield_strength = 20e6f; p.r = 180; p.g = 175; p.b = 165;
      p.roughness = 0.95f;
      reg({ "concrete", Domain::Inorganic, CraftingMethod::Casting, DeformationBehavior::Brittle,
            p, 0.3f, 1.0f, 0.005f, 0.0f, 0.0f, 0.0f, 0.0f }); }

    // Titanium (strong, lightweight)
    { MaterialProperties p;
      p.youngs_modulus = 110e9f; p.poisson_ratio = 0.34f; p.density = 4500.0f;
      p.yield_strength = 880e6f; p.r = 170; p.g = 175; p.b = 185;
      p.roughness = 0.35f; p.metalness = 0.75f;
      reg({ "titanium", Domain::Inorganic, CraftingMethod::Forging, DeformationBehavior::Plastic,
            p, 0.005f, 200.0f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f }); }

    // Rubber (highly elastic)
    { MaterialProperties p;
      p.youngs_modulus = 10e6f; p.poisson_ratio = 0.49f; p.density = 1100.0f;
      p.yield_strength = 15e6f; p.r = 50; p.g = 50; p.b = 50;
      p.roughness = 0.9f;
      reg({ "rubber", Domain::Inorganic, CraftingMethod::Molding, DeformationBehavior::Hyperelastic,
            p, 0.01f, 50.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f }); }

    // Tungsten (extremely dense, hard)
    { MaterialProperties p;
      p.youngs_modulus = 400e9f; p.poisson_ratio = 0.28f; p.density = 19250.0f;
      p.yield_strength = 750e6f; p.r = 130; p.g = 130; p.b = 140;
      p.roughness = 0.3f; p.metalness = 0.85f;
      reg({ "tungsten", Domain::Inorganic, CraftingMethod::Forging, DeformationBehavior::Plastic,
            p, 0.01f, 300.0f, 0.003f, 0.0f, 0.0f, 0.0f, 0.0f }); }

    // Gold (soft, conductive)
    { MaterialProperties p;
      p.youngs_modulus = 79e9f; p.poisson_ratio = 0.42f; p.density = 19300.0f;
      p.yield_strength = 120e6f; p.r = 255; p.g = 215; p.b = 0;
      p.roughness = 0.1f; p.metalness = 0.95f;
      reg({ "gold", Domain::Inorganic, CraftingMethod::Casting, DeformationBehavior::Plastic,
            p, 0.001f, 500.0f, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f }); }
}

/* ── Public API ───────────────────────────────────────────────────── */
MaterialId MaterialScienceRegistry::get_material(std::string_view name, MaterialRegistry& phys_reg) {
    auto key = to_lower(name);
    if (name_to_id_.empty()) init_builtin_materials();

    auto it = name_to_id_.find(key);
    if (it != name_to_id_.end()) {
        MaterialId id = it->second;
        // Register in physics registry if not already
        if (!phys_reg.has(id))
            phys_reg.register_material(id, definitions_[id].physics);
        return id;
    }

    // Not found — register a default
    Domain domain = (key.find("organ") != std::string::npos ||
                     key.find("bio") != std::string::npos) ? Domain::Organic : Domain::Inorganic;
    MaterialProperties p;
    if (domain == Domain::Organic) {
        p.youngs_modulus = 1e6f; p.poisson_ratio = 0.45f; p.density = 1100.0f;
        p.r = 180; p.g = 140; p.b = 120;
    } else {
        p.youngs_modulus = 200e9f; p.poisson_ratio = 0.3f; p.density = 7800.0f;
        p.r = 160; p.g = 160; p.b = 170; p.metalness = 0.7f;
    }
    MaterialId new_id = next_id_++;
    MaterialDefinition def;
    def.name = std::string(name);
    def.domain = domain;
    def.crafting = domain == Domain::Organic ? CraftingMethod::Growth : CraftingMethod::Assembly;
    def.deformation = domain == Domain::Organic ? DeformationBehavior::Viscoelastic : DeformationBehavior::Elastic;
    def.physics = p;
    name_to_id_[key] = new_id;
    definitions_.push_back(def);
    phys_reg.register_material(new_id, p);
    return new_id;
}

const MaterialDefinition& MaterialScienceRegistry::definition(MaterialId id) const {
    return definitions_[id];
}

std::vector<MaterialId> MaterialScienceRegistry::list_domain(Domain d) const {
    if (name_to_id_.empty()) const_cast<MaterialScienceRegistry*>(this)->init_builtin_materials();
    std::vector<MaterialId> result;
    for (const auto& def : definitions_)
        if (def.domain == d)
            result.push_back(&def - &definitions_[0]);
    return result;
}

MaterialId MaterialScienceRegistry::resolve_from_text(std::string_view text, MaterialRegistry& phys_reg) const {
    if (name_to_id_.empty()) const_cast<MaterialScienceRegistry*>(this)->init_builtin_materials();

    // Try exact match first, then substring
    std::string lower = to_lower(text);
    for (const auto& [name, id] : name_to_id_) {
        if (lower.find(name) != std::string::npos)
            return const_cast<MaterialScienceRegistry*>(this)->get_material(name, phys_reg);
    }
    return domain_default(Domain::Inorganic);
}

MaterialId MaterialScienceRegistry::domain_default(Domain d) const {
    if (name_to_id_.empty()) const_cast<MaterialScienceRegistry*>(this)->init_builtin_materials();
    auto it = name_to_id_.find(d == Domain::Organic ? "tissue" : "steel");
    return it != name_to_id_.end() ? it->second : 0;
}

} // namespace delirium