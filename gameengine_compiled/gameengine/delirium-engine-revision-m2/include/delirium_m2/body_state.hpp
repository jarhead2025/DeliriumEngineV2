#pragma once

#include "delirium_m2/types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace delirium::m2 {

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

struct AnchorState {
    Vec3 rest_position{};
    Vec3 position{};
    Vec3 velocity{};
    float temperature = 0.0f;
    float damage = 0.0f;
};

struct PatchState {
    std::array<float, 8> coefficients{};
    std::array<std::int8_t, 8> stitch{};
    float strain_energy = 0.0f;
    float temperature = 0.0f;
    float damage = 0.0f;
};

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

} // namespace delirium::m2