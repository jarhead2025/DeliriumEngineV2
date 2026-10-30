#pragma once
#include "delirium_m8/material_registry.hpp"
#include <unordered_map>
namespace delirium::m8 {
class SharedMaterialPool {
public:
    MaterialId register_material(MaterialProperties props);
    const MaterialProperties& get(MaterialId id) const;
    std::uint32_t reference_count(MaterialId id) const;
    void add_reference(MaterialId id);
    void remove_reference(MaterialId id);
    std::size_t material_count() const { return materials_.size(); }
private:
    std::unordered_map<MaterialId, MaterialProperties> materials_;
    std::unordered_map<MaterialId, std::uint32_t> refs_;
    MaterialId next_id_ = 0;
};
} // namespace delirium::m8