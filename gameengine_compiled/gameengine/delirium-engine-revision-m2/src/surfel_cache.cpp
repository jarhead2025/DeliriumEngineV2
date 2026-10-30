#include "delirium_m2/surfel_cache.hpp"

#include <stdexcept>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// SurfelCache
// ---------------------------------------------------------------------------

const Surfel& SurfelCache::surfel(std::size_t index) const {
    if (index >= surfels_.size()) throw std::out_of_range("surfel index out of range");
    return surfels_[index];
}

void SurfelCache::reserve(std::size_t capacity) {
    surfels_.reserve(capacity);
}

void SurfelCache::clear() {
    surfels_.clear();
    source_generation_ = 0;
    topology_generation_ = 0;
}

void SurfelCache::push_back(const Surfel& s) {
    surfels_.push_back(s);
}

bool SurfelCache::matches_generation(Generation body_gen,
                                      TopologyGeneration topo_gen) const {
    return source_generation_ == body_gen &&
           topology_generation_ == topo_gen;
}

// ---------------------------------------------------------------------------
// CacheUpdater
// ---------------------------------------------------------------------------

CacheUpdater::CacheUpdater(std::uint32_t grid_resolution)
    : grid_resolution_(grid_resolution) {}

bool CacheUpdater::update(SurfelCache& cache,
                           const BodyState& state,
                           const BodyTopology& topology,
                           std::string* error) {
    // If topology changed, full rebuild is required
    if (cache.topology_generation() != topology.topology_generation()) {
        return rebuild(cache, state, topology, error);
    }

    // Only update if there are dirty patches
    if (state.dirty_patches.indices().empty()) {
        // No dirty patches — just update the generation tag
        cache.set_source_generation(state.generation);
        return true;
    }

    // For M2, rebuild on any dirty patches (M5 adds incremental range replacement)
    return rebuild(cache, state, topology, error);
}

bool CacheUpdater::rebuild(SurfelCache& cache,
                            const BodyState& state,
                            const BodyTopology& topology,
                            std::string* error) {
    SurfaceEvaluator evaluator;

    // Evaluate all patches
    std::vector<SurfacePoint> points;
    float step = 1.0f / grid_resolution_;
    std::uint32_t pts_per_patch = (grid_resolution_ + 1) * (grid_resolution_ + 1);
    points.reserve(topology.patch_count() * pts_per_patch);

    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        for (std::uint32_t i = 0; i <= grid_resolution_; ++i) {
            for (std::uint32_t j = 0; j <= grid_resolution_; ++j) {
                float u = i * step;
                float v = j * step;
                SurfacePoint sp = evaluator.evaluate_patch(state, topology, pid, u, v);
                points.push_back(sp);
            }
        }
    }

    // Convert surface points to surfels
    cache.clear();
    cache.reserve(points.size());

    // Estimate surfel radius from grid resolution
    float avg_radius = 1.0f / grid_resolution_;

    for (const auto& sp : points) {
        Surfel s;
        s.position = sp.position;
        s.normal = sp.normal;
        s.radius = avg_radius;
        s.material = sp.material;
        s.source_patch = sp.source_patch;
        // Default color (light gray) — M4 adds material-driven colors
        s.r = 200; s.g = 200; s.b = 200; s.a = 255;
        cache.push_back(s);
    }

    // Tag with the source generation — this is the generation invariant
    cache.set_source_generation(state.generation);
    cache.set_topology_generation(topology.topology_generation());

    return true;
}

} // namespace delirium::m2