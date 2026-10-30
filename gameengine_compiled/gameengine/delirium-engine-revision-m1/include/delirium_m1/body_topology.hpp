#pragma once

#include "delirium_m1/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace delirium::m1 {

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

    bool finalize(std::string* error = nullptr);
    bool finalized() const { return finalized_; }

    std::size_t anchor_count() const { return anchor_to_patches_.size(); }
    std::size_t patch_count() const { return patches_.size(); }
    const PatchTopology& patch(PatchId id) const;
    const std::vector<PatchId>& patches_for_anchor(AnchorId id) const;

private:
    std::vector<PatchTopology> patches_;
    std::vector<std::vector<PatchId>> anchor_to_patches_;
    bool finalized_ = false;
};

} // namespace delirium::m1
