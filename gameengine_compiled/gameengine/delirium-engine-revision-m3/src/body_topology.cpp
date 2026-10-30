#include "delirium_m3/body_topology.hpp"

#include <stdexcept>

namespace delirium::m3 {

AnchorId BodyTopology::add_anchor() {
    if (finalized_) return kInvalidAnchor;
    anchor_to_patches_.emplace_back();
    return static_cast<AnchorId>(anchor_to_patches_.size() - 1);
}

PatchId BodyTopology::add_patch(const std::array<AnchorId, kPatchArity>& anchors,
                                MaterialId material,
                                BasisType basis,
                                PatchCoefficients coeffs) {
    if (finalized_) return kInvalidPatch;
    patches_.push_back({anchors, material, basis, coeffs, 0, 0});
    return static_cast<PatchId>(patches_.size() - 1);
}

void BodyTopology::add_welded_seam(const WeldedSeam& seam) {
    if (!seam.is_valid()) return;
    welded_seams_.push_back(seam);
}

bool BodyTopology::finalize(std::string* error) {
    if (finalized_) {
        topo_gen_++;
        return true;
    }
    for (std::vector<PatchId>& incident : anchor_to_patches_) incident.clear();
    for (PatchId patch_id = 0; patch_id < patches_.size(); ++patch_id) {
        const PatchTopology& patch = patches_[patch_id];
        if (patch.material == kInvalidMaterial) {
            if (error) *error = "patch has no material assignment";
            return false;
        }
        for (AnchorId anchor : patch.anchors) {
            if (anchor >= anchor_to_patches_.size()) {
                if (error) *error = "patch references an anchor outside topology";
                return false;
            }
            anchor_to_patches_[anchor].push_back(patch_id);
        }
    }
    finalized_ = true;
    topo_gen_ = 1;
    return true;
}

const PatchTopology& BodyTopology::patch(PatchId id) const {
    if (id >= patches_.size()) throw std::out_of_range("invalid patch ID");
    return patches_[id];
}

const std::vector<PatchId>& BodyTopology::patches_for_anchor(AnchorId id) const {
    if (id >= anchor_to_patches_.size()) throw std::out_of_range("invalid anchor ID");
    return anchor_to_patches_[id];
}

} // namespace delirium::m3