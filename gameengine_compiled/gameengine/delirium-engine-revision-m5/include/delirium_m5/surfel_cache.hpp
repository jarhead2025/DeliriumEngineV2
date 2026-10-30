#pragma once

#include "delirium_m5/types.hpp"
#include "delirium_m5/body_topology.hpp"
#include "delirium_m5/body_state.hpp"
#include "delirium_m5/material_registry.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace delirium::m5 {

class SurfelCache {
public:
    Generation source_generation() const { return source_generation_; }
    void set_source_generation(Generation g) { source_generation_ = g; }
    TopologyGeneration topology_generation() const { return topology_generation_; }
    void set_topology_generation(TopologyGeneration g) { topology_generation_ = g; }
    std::size_t surfel_count() const { return surfels_.size(); }
    const Surfel& surfel(std::size_t index) const;
    Surfel& surfel(std::size_t index);
    void reserve(std::size_t capacity);
    void clear();
    void push_back(const Surfel& s);
    bool matches_generation(Generation body_gen, TopologyGeneration topo_gen) const;
    void mark_patch_updated(PatchId patch);
    bool patch_was_updated(PatchId patch) const;
    void clear_patch_updates();
private:
    std::vector<Surfel> surfels_;
    Generation source_generation_ = 0;
    TopologyGeneration topology_generation_ = 0;
    std::vector<bool> patch_updated_;
};

class CacheUpdater {
public:
    explicit CacheUpdater(std::uint32_t grid_resolution = 8);
    bool update(SurfelCache& cache, const BodyState& state, const BodyTopology& topology,
                const MaterialRegistry& materials, std::string* error = nullptr);
    bool rebuild(SurfelCache& cache, const BodyState& state, const BodyTopology& topology,
                 const MaterialRegistry& materials, std::string* error = nullptr);
    bool rebuild_with_lod(SurfelCache& cache, const BodyState& state, const BodyTopology& topology,
                          const MaterialRegistry& materials, float camera_distance,
                          std::string* error = nullptr);
    std::uint32_t grid_resolution() const { return grid_resolution_; }
private:
    std::uint32_t grid_resolution_;
};

} // namespace delirium::m5