#pragma once

#include "delirium_m4/types.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace delirium::m4 {

// ---------------------------------------------------------------------------
// PatchTopology — with basis type and coefficients
// ---------------------------------------------------------------------------
struct PatchTopology {
    std::array<AnchorId, 4> anchors{};
    MaterialId material = kInvalidMaterial;

    BasisType basis = BasisType::Bilinear;
    PatchCoefficients coefficients{};

    std::uint32_t cache_first = 0;
    std::uint32_t cache_count = 0;
};

// ---------------------------------------------------------------------------
// BodyTopology — with welded seam support
// ---------------------------------------------------------------------------
class BodyTopology {
public:
    static constexpr std::size_t kPatchArity = 4;

    AnchorId add_anchor();
    PatchId add_patch(const std::array<AnchorId, kPatchArity>& anchors,
                      MaterialId material,
                      BasisType basis = BasisType::Bilinear,
                      PatchCoefficients coeffs = {});

    void add_welded_seam(const WeldedSeam& seam);

    bool finalize(std::string* error = nullptr);
    bool finalized() const { return finalized_; }

    std::size_t anchor_count() const { return anchor_to_patches_.size(); }
    std::size_t patch_count() const { return patches_.size(); }
    const PatchTopology& patch(PatchId id) const;
    const std::vector<PatchId>& patches_for_anchor(AnchorId id) const;

    const std::vector<WeldedSeam>& welded_seams() const { return welded_seams_; }
    TopologyGeneration topology_generation() const { return topo_gen_; }

private:
    std::vector<PatchTopology> patches_;
    std::vector<std::vector<PatchId>> anchor_to_patches_;
    std::vector<WeldedSeam> welded_seams_;
    bool finalized_ = false;
    TopologyGeneration topo_gen_ = 0;
};

} // namespace delirium::m4