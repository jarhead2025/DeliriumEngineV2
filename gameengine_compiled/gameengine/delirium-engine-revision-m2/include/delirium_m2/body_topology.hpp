#pragma once

#include "delirium_m2/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m2 {

struct PatchTopology {
    std::array<AnchorId, 4> anchors{};
    MaterialId material = kInvalidMaterial;
    PatchCacheRange cache_range{};
};

class BodyTopology {
public:
    static constexpr std::size_t kPatchArity = 4;

    AnchorId add_anchor();
    PatchId add_patch(const std::array<AnchorId, kPatchArity>& anchors,
                      MaterialId material,
                      PatchCacheRange cache_range);

    // Finalize builds adjacency lists. Calling finalize again clears and
    // rebuilds — this is the deliberate invalidation path for topology changes.
    bool finalize(std::string* error = nullptr);
    bool finalized() const { return finalized_; }

    // When topology changes are made (after a re-finalize), the topology
    // generation increments. Cache consumers must check this.
    TopologyGeneration topology_generation() const { return topology_generation_; }

    std::size_t anchor_count() const { return anchor_to_patches_.size(); }
    std::size_t patch_count() const { return patches_.size(); }
    const PatchTopology& patch(PatchId id) const;
    const std::vector<PatchId>& patches_for_anchor(AnchorId id) const;

private:
    std::vector<PatchTopology> patches_;
    std::vector<std::vector<PatchId>> anchor_to_patches_;
    bool finalized_ = false;
    TopologyGeneration topology_generation_ = 0;
};

} // namespace delirium::m2