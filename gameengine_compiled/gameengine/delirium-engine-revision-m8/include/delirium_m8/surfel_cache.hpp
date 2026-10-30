#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_topology.hpp"
#include "delirium_m8/body_state.hpp"
#include "delirium_m8/material_registry.hpp"
#include <string>
namespace delirium::m8 {
class SurfelCache {
public:
    void set_topology_generation(TopologyGeneration g) { topo_gen_ = g; }
    TopologyGeneration topology_generation() const { return topo_gen_; }
    const std::vector<Surfel>& surfels() const { return surfels_; }
    std::vector<Surfel>& surfels_mut() { return surfels_; }
    std::size_t count() const { return surfels_.size(); }
private:
    TopologyGeneration topo_gen_ = 0;
    std::vector<Surfel> surfels_;
};
class CacheUpdater {
public:
    bool rebuild(SurfelCache& cache, const BodyState& state, const BodyTopology& topology,
                 const MaterialRegistry& materials, std::string* error = nullptr);
    bool rebuild_with_lod(SurfelCache& cache, const BodyState& state, const BodyTopology& topology,
                          const MaterialRegistry& materials, float cam_dist,
                          std::string* error = nullptr);
private:
    void emit_surfel(const BodyState& state, const PatchTopology& patch,
                     const MaterialProperties& mat, float u, float v, SurfelCache& cache);
};
} // namespace delirium::m8