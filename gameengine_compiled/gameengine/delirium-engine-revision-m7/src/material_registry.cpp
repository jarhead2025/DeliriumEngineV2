#include "delirium_m7/material_registry.hpp"
#include <stdexcept>

namespace delirium::m7 {

void MaterialRegistry::register_material(MaterialId id, MaterialProperties props) {
    materials_[id] = std::move(props);
}

const MaterialProperties& MaterialRegistry::get(MaterialId id) const {
    if (!has(id)) throw std::runtime_error("material not found");
    return materials_.at(id);
}

std::vector<MaterialId> MaterialRegistry::ids() const {
    std::vector<MaterialId> result;
    result.reserve(materials_.size());
    for (const auto& [id, _] : materials_) result.push_back(id);
    return result;
}

} // namespace delirium::m7