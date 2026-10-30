#include "body_topology.h"

namespace delirium {

AnchorId BodyTopology::add_anchor() {
    AnchorId id = static_cast<AnchorId>(anchor_to_patches_.size());
    anchor_to_patches_.push_back({});
    return id;
}

PatchId BodyTopology::add_patch(const std::array<AnchorId, kPatchArity>& anchors,
                                 MaterialId material,
                                 BasisType basis,
                                 PatchCoefficients coeffs) {
    PatchId id = static_cast<PatchId>(patches_.size());
    patches_.push_back({anchors, material, basis, coeffs, 0, 0});
    for (AnchorId a : anchors) {
        if (a < anchor_to_patches_.size())
            anchor_to_patches_[a].push_back(id);
    }
    return id;
}

bool BodyTopology::finalize(std::string* error) {
    if (finalized_) return true;
    if (patches_.empty()) {
        if (error) *error = "no patches";
        return false;
    }
    finalized_ = true;
    topo_gen_++;
    return true;
}

const PatchTopology& BodyTopology::patch(PatchId id) const {
    return patches_[id];
}

const std::vector<PatchId>& BodyTopology::patches_for_anchor(AnchorId id) const {
    return anchor_to_patches_[id];
}

} // namespace delirium