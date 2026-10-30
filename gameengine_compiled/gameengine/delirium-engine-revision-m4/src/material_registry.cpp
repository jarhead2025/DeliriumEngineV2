#include "delirium_m4/material_registry.hpp"

namespace delirium::m4 {

const MaterialProperties& MaterialRegistry::default_material() {
    static const MaterialProperties default_props = {
        1.0e6f,    // youngs_modulus
        0.45f,     // poisson_ratio
        1000.0f,   // density
        5.0e5f,    // yield_strength
        200, 200, 200, // r, g, b
        0.5f,      // roughness
        0.0f       // metalness
    };
    return default_props;
}

MaterialRegistry::MaterialRegistry() : default_props_(default_material()) {}

void MaterialRegistry::register_material(MaterialId id, const MaterialProperties& props) {
    materials_[id] = props;
}

const MaterialProperties& MaterialRegistry::get(MaterialId id) const {
    auto it = materials_.find(id);
    if (it != materials_.end()) {
        return it->second;
    }
    return default_props_;
}

bool MaterialRegistry::has(MaterialId id) const {
    return materials_.find(id) != materials_.end();
}

std::vector<MaterialId> MaterialRegistry::all_ids() const {
    std::vector<MaterialId> ids;
    ids.reserve(materials_.size());
    for (const auto& [id, _] : materials_) {
        ids.push_back(id);
    }
    return ids;
}

void MaterialRegistry::get_color(MaterialId id, std::uint8_t& r, std::uint8_t& g,
                                  std::uint8_t& b, float& roughness, float& metalness) const {
    const MaterialProperties& props = get(id);
    r = props.r;
    g = props.g;
    b = props.b;
    roughness = props.roughness;
    metalness = props.metalness;
}

} // namespace delirium::m4