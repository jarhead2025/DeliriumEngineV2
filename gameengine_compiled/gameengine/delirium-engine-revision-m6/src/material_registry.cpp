#include "delirium_m6/material_registry.hpp"
namespace delirium::m6 {
const MaterialProperties& MaterialRegistry::default_material() {
    static const MaterialProperties p = {1.0e6f,0.45f,1000.0f,5.0e5f,0.5f,1e-5f,200,200,200,0.5f,0.0f};
    return p;
}
MaterialRegistry::MaterialRegistry() : default_props_(default_material()) {}
void MaterialRegistry::register_material(MaterialId id, const MaterialProperties& props) { materials_[id] = props; }
const MaterialProperties& MaterialRegistry::get(MaterialId id) const {
    auto it = materials_.find(id);
    return it != materials_.end() ? it->second : default_props_;
}
bool MaterialRegistry::has(MaterialId id) const { return materials_.find(id) != materials_.end(); }
std::vector<MaterialId> MaterialRegistry::all_ids() const {
    std::vector<MaterialId> ids; ids.reserve(materials_.size());
    for (const auto& [id, _] : materials_) ids.push_back(id);
    return ids;
}
void MaterialRegistry::get_color(MaterialId id, std::uint8_t& r, std::uint8_t& g,
                                  std::uint8_t& b, float& roughness, float& metalness) const {
    const auto& p = get(id); r = p.r; g = p.g; b = p.b; roughness = p.roughness; metalness = p.metalness;
}
} // namespace delirium::m6