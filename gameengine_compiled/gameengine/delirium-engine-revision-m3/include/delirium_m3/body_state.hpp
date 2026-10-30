#pragma once

#include "delirium_m3/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace delirium::m3 {

// ---------------------------------------------------------------------------
// DirtySet
// ---------------------------------------------------------------------------
class DirtySet {
public:
    explicit DirtySet(std::size_t count = 0) : bits_(count, false) {}

    void resize(std::size_t count) { bits_.assign(count, false); }
    void mark(std::size_t index) { if (index < bits_.size()) bits_[index] = true; }
    bool marked(std::size_t index) const { return index < bits_.size() && bits_[index]; }
    void clear() { std::fill(bits_.begin(), bits_.end(), false); }
    std::size_t size() const { return bits_.size(); }
    std::vector<std::uint32_t> indices() const {
        std::vector<std::uint32_t> result;
        for (std::size_t i = 0; i < bits_.size(); ++i)
            if (bits_[i]) result.push_back(static_cast<std::uint32_t>(i));
        return result;
    }

private:
    std::vector<bool> bits_;
};

// ---------------------------------------------------------------------------
// AnchorState
// ---------------------------------------------------------------------------
struct AnchorState {
    Vec3 rest_position{};
    Vec3 position{};
    Vec3 velocity{};
    float temperature = 0.0f;
    float damage = 0.0f;
};

// ---------------------------------------------------------------------------
// PatchState — with vector residuals
// ---------------------------------------------------------------------------
struct PatchState {
    // Scalar coefficients (from topology, mutable for runtime deformation)
    std::array<float, 8> coefficients{};

    // Vector residuals — M3: sub-anchor geometric detail
    std::array<Vec3, 4> residuals{};

    float strain_energy = 0.0f;
    float temperature = 0.0f;
    float damage = 0.0f;
};

// ---------------------------------------------------------------------------
// BodyState — double-buffered, generation-tagged
// ---------------------------------------------------------------------------
struct BodyState {
    Generation generation = 0;
    TopologyGeneration topology_generation = 0;

    std::vector<AnchorState> anchors;
    std::vector<PatchState> patches;
    DirtySet dirty_anchors;
    DirtySet dirty_patches;

    void resize(std::size_t anchor_count, std::size_t patch_count) {
        anchors.resize(anchor_count);
        patches.resize(patch_count);
        dirty_anchors.resize(anchor_count);
        dirty_patches.resize(patch_count);
    }

    void clear_dirty() {
        dirty_anchors.clear();
        dirty_patches.clear();
    }
};

} // namespace delirium::m3