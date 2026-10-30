#pragma once

#include "delirium_m7/types.hpp"
#include <unordered_map>
#include <optional>

namespace delirium::m7 {

class MaterialRegistry {
public:
    void register_material(MaterialId id, MaterialProperties props);
    bool has(MaterialId id) const { return materials_.count(id) > 0; }
    const MaterialProperties& get(MaterialId id) const;
    std::vector<MaterialId> ids() const;
private:
    std::unordered_map<MaterialId, MaterialProperties> materials_;
};

} // namespace delirium::m7