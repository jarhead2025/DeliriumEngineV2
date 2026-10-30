#include "delirium_m8/body_topology.hpp"
#include <algorithm>

namespace delirium::m8 {

AnchorId BodyTopology::add_anchor() {
    anchor_to_patches_.push_back({});
    return static_cast<AnchorId>(anchor_to_patches_.size() - 1);
}

PatchId BodyTopology::add_patch(const std::array<AnchorId, kPatchArity>& anchors,
                                 MaterialId material, BasisType basis,
                                 PatchCoefficients coeffs) {
    PatchTopology p;
    p.anchors = anchors;
    p.material = material;
    p.basis = basis;
    p.coefficients = coeffs;
    patches_.push_back(std::move(p));
    for (auto a : anchors) {
        if (a < anchor_to_patches_.size())
            anchor_to_patches_[a].push_back(static_cast<PatchId>(patches_.size() - 1));
    }
    return static_cast<PatchId>(patches_.size() - 1);
}

void BodyTopology::add_welded_seam(const WeldedSeam& seam) {
    welded_seams_.push_back(seam);
}

bool BodyTopology::finalize(std::string* error) {
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

} // namespace delirium::m8