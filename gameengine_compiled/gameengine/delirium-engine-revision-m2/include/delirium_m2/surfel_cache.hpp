#pragma once

#include "delirium_m2/types.hpp"
#include "delirium_m2/surface_evaluator.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// Surfel — a single surface element in the render cache
// ---------------------------------------------------------------------------
struct Surfel {
    Vec3 position{};
    Vec3 normal{};
    float radius = 0.0f;

    // Color (RGBA, normalized [0,1])
    std::uint8_t r = 255;
    std::uint8_t g = 255;
    std::uint8_t b = 255;
    std::uint8_t a = 255;

    // Source identity for composition safety
    MaterialId material = kInvalidMaterial;
    PatchId source_patch = kInvalidPatch;
};

// ---------------------------------------------------------------------------
// SurfelCache — generation-tagged render cache derived from BodyState
// ---------------------------------------------------------------------------
//
// The cache is the ONLY path from BodyState to the renderer. A renderer may
// use only a cache whose source_generation matches a complete BodyState
// generation.
// ---------------------------------------------------------------------------

class SurfelCache {
public:
    Generation source_generation() const { return source_generation_; }
    void set_source_generation(Generation g) { source_generation_ = g; }

    TopologyGeneration topology_generation() const { return topology_generation_; }
    void set_topology_generation(TopologyGeneration g) { topology_generation_ = g; }

    std::size_t surfel_count() const { return surfels_.size(); }
    const Surfel& surfel(std::size_t index) const;

    // Reserve capacity to avoid reallocations during updates
    void reserve(std::size_t capacity);

    // Clear all surfels (called before a full or incremental rebuild)
    void clear();

    // Append a surfel
    void push_back(const Surfel& s);

    // Check if this cache is valid for a given BodyState generation
    bool matches_generation(Generation body_gen,
                            TopologyGeneration topo_gen) const;

private:
    std::vector<Surfel> surfels_;
    Generation source_generation_ = 0;
    TopologyGeneration topology_generation_ = 0;
};

// ---------------------------------------------------------------------------
// CacheUpdater — consumes dirty patch ranges and emits surfel cache updates
// ---------------------------------------------------------------------------
//
// This is the bridge between the body runtime and the render cache. It reads
// one published BodyState, evaluates the dirty patches through the surface
// evaluator, and writes surfels into the cache tagged with the same generation.
// ---------------------------------------------------------------------------

class CacheUpdater {
public:
    explicit CacheUpdater(std::uint32_t grid_resolution = 8);

    // Update the cache for the dirty patches in the given state.
    // If topology_generation changed, a full rebuild is performed.
    // Otherwise, only dirty patches are re-evaluated and their surfels
    // replaced in-place.
    bool update(SurfelCache& cache,
                const BodyState& state,
                const BodyTopology& topology,
                std::string* error = nullptr);

    // Full rebuild: regenerate all surfels from all patches
    bool rebuild(SurfelCache& cache,
                 const BodyState& state,
                 const BodyTopology& topology,
                 std::string* error = nullptr);

    std::uint32_t grid_resolution() const { return grid_resolution_; }

private:
    std::uint32_t grid_resolution_;
};

} // namespace delirium::m2