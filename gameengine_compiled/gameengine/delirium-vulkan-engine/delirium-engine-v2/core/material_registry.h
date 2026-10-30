#pragma once

#include "types.h"
#include "body_topology.h"
#include "body_state.h"
#include <string>
#include <unordered_map>

namespace delirium {

class MaterialRegistry {
public:
    void register_material(MaterialId id, MaterialProperties props);
    bool has(MaterialId id) const { return materials_.count(id) > 0; }
    const MaterialProperties& get(MaterialId id) const;
    std::vector<MaterialId> ids() const;
private:
    std::unordered_map<MaterialId, MaterialProperties> materials_;
};

} // namespace delirium