#pragma once

#include "delirium_m6/types.hpp"
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace delirium::m6 {

class MaterialRegistry {
public:
    MaterialRegistry();
    void register_material(MaterialId id, const MaterialProperties& props);
    const MaterialProperties& get(MaterialId id) const;
    bool has(MaterialId id) const;
    std::vector<MaterialId> all_ids() const;
    static const MaterialProperties& default_material();
    void get_color(MaterialId id, std::uint8_t& r, std::uint8_t& g,
                   std::uint8_t& b, float& roughness, float& metalness) const;
private:
    std::map<MaterialId, MaterialProperties> materials_;
    MaterialProperties default_props_;
};

} // namespace delirium::m6