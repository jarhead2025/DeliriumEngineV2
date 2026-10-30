#pragma once

#include "delirium_m4/types.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace delirium::m4 {

// ---------------------------------------------------------------------------
// MaterialRegistry — M4: per-region mechanical behavior lookup
// ---------------------------------------------------------------------------
//
// Maps MaterialId to MaterialProperties. Provides defaults for unknown
// materials and ensures physics/rendering consistency.
// ---------------------------------------------------------------------------
class MaterialRegistry {
public:
    MaterialRegistry();

    // Register material properties for a material ID
    void register_material(MaterialId id, const MaterialProperties& props);

    // Get material properties (returns default for unknown IDs)
    const MaterialProperties& get(MaterialId id) const;

    // Check if a material is registered
    bool has(MaterialId id) const;

    // Get all registered material IDs
    std::vector<MaterialId> all_ids() const;

    // Get default material properties
    static const MaterialProperties& default_material();

    // Get material color for rendering
    void get_color(MaterialId id, std::uint8_t& r, std::uint8_t& g,
                   std::uint8_t& b, float& roughness, float& metalness) const;

private:
    std::map<MaterialId, MaterialProperties> materials_;
    MaterialProperties default_props_;
};

} // namespace delirium::m4