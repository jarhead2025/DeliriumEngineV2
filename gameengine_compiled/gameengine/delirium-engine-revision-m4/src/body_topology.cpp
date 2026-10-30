#include "delirium_m4/body_topology.hpp"

#include <stdexcept>
#include <utility>

namespace delirium::m4 {

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
    PatchTopology patch;
    patch.anchors = anchors;
    patch.material = material;
    patch.basis = basis;
    patch.coefficients = std::move(coeffs);
    patches_.push_back(patch);

    for (AnchorId a : anchors) {
        if (a < anchor_to_patches_.size()) {
            anchor_to_patches_[a].push_back(id);
        }
    }

    return id;
}

void BodyTopology::add_welded_seam(const WeldedSeam& seam) {
    if (!seam.is_valid()) return;
    welded_seams_.push_back(seam);
}

bool BodyTopology::finalize(std::string* error) {
    if (patches_.empty()) {
        if (error) *error = "cannot finalize empty topology";
        return false;
    }

    for (const auto& patch : patches_) {
        for (AnchorId a : patch.anchors) {
            if (a >= anchor_to_patches_.size()) {
                if (error) *error = "patch references invalid anchor";
                return false;
            }
        }
    }

    finalized_ = true;
    topo_gen_++;
    return true;
}

const PatchTopology& BodyTopology::patch(PatchId id) const {
    if (id >= patches_.size()) {
        static const PatchTopology empty;
        return empty;
    }
    return patches_[id];
}

const std::vector<PatchId>& BodyTopology::patches_for_anchor(AnchorId id) const {
    if (id >= anchor_to_patches_.size()) {
        static const std::vector<PatchId> empty;
        return empty;
    }
    return anchor_to_patches_[id];
}

} // namespace delirium::m4