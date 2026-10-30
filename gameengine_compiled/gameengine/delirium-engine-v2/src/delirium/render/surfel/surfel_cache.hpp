#pragma once
#include "delirium/geometry/patch/patch.hpp"
#include <vector>
#include <unordered_map>

namespace delirium {

/// Surfel — an oriented disk representing a surface sample
struct Surfel {
    vec3 center;
    vec3 normal;
    vec3 color;
    float radius;
    uint32_t patch_id;
    uint32_t material_id;
};

/// SurfelCache — derived render cache from body state
///
/// NOT the authoritative geometry. Regenerated when patches are dirty.
class SurfelCache {
public:
    std::vector<Surfel> surfels;
    uint32_t generation_id; // matches body generation

    SurfelCache() : generation_id(0) {}

    /// Generate surfels from patches
    void generate(const std::vector<Patch>& patches,
                  const std::vector<AnchorRecord>& anchors,
                  float density = 1.0f);

    /// Clear and increment generation
    void clear();

    /// Is this cache up-to-date with the given generation?
    bool is_current(uint32_t gen) const { return generation_id == gen; }
};

} // namespace delirium