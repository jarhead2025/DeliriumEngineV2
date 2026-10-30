#include "material_registry.h"
#include <stdexcept>

namespace delirium {

void MaterialRegistry::register_material(MaterialId id, MaterialProperties props) {
    materials_[id] = std::move(props);
}

const MaterialProperties& MaterialRegistry::get(MaterialId id) const {
    auto it = materials_.find(id);
    if (it == materials_.end())
        throw std::runtime_error("missing material " + std::to_string(id));
    return it->second;
}

std::vector<MaterialId> MaterialRegistry::ids() const {
    std::vector<MaterialId> result;
    for (const auto& [id, _] : materials_)
        result.push_back(id);
    return result;
}

} // namespace delirium