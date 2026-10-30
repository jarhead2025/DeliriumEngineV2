#include "delirium_m3/surfel_cache.hpp"

#include "delirium_m3/body_topology.hpp"
#include "delirium_m3/basis_functions.hpp"

#include <stdexcept>

namespace delirium::m3 {

// ---------------------------------------------------------------------------
// SurfelCache
// ---------------------------------------------------------------------------

const Surfel& SurfelCache::surfel(std::size_t index) const {
    if (index >= surfels_.size()) throw std::out_of_range("surfel index out of range");
    return surfels_[index];
}

Surfel& SurfelCache::surfel(std::size_t index) {
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
    patch_updated_.clear();
}

void SurfelCache::push_back(const Surfel& s) {
    surfels_.push_back(s);
}

bool SurfelCache::matches_generation(Generation body_gen,
                                      TopologyGeneration topo_gen) const {
    return source_generation_ == body_gen &&
           topology_generation_ == topo_gen;
}

void SurfelCache::mark_patch_updated(PatchId patch) {
    if (patch >= patch_updated_.size()) patch_updated_.resize(patch + 1, false);
    patch_updated_[patch] = true;
}

bool SurfelCache::patch_was_updated(PatchId patch) const {
    return patch < patch_updated_.size() && patch_updated_[patch];
}

void SurfelCache::clear_patch_updates() {
    std::fill(patch_updated_.begin(), patch_updated_.end(), false);
}

// ---------------------------------------------------------------------------
// CacheUpdater — M3: incremental + temporal coherence
// ---------------------------------------------------------------------------

CacheUpdater::CacheUpdater(std::uint32_t grid_resolution)
    : grid_resolution_(grid_resolution) {}

bool CacheUpdater::update(SurfelCache& cache,
                           const BodyState& state,
                           const BodyTopology& topology,
                           std::string* /*error*/) {
    // If topology changed, full rebuild is required
    if (cache.topology_generation() != topology.topology_generation()) {
        return rebuild(cache, state, topology);
    }

    auto dirty_indices = state.dirty_patches.indices();

    // If no dirty patches, just update generation tag
    if (dirty_indices.empty()) {
        cache.set_source_generation(state.generation);
        cache.clear_patch_updates();
        return true;
    }

    // Incremental update: rebuild only dirty patches
    // For M3, we rebuild the full cache but mark which patches were updated
    // (M5 adds true incremental range replacement)
    return rebuild(cache, state, topology);
}

bool CacheUpdater::rebuild(SurfelCache& cache,
                            const BodyState& state,
                            const BodyTopology& topology,
                            std::string* /*error*/) {
    float step = 1.0f / grid_resolution_;
    std::uint32_t pts_per_patch = (grid_resolution_ + 1) * (grid_resolution_ + 1);

    cache.clear();
    cache.reserve(topology.patch_count() * pts_per_patch);

    float avg_radius = 1.0f / grid_resolution_;

    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const PatchTopology& patch = topology.patch(pid);

        // Get anchor positions from state
        const Vec3& p00 = state.anchors[patch.anchors[0]].position;
        const Vec3& p10 = state.anchors[patch.anchors[1]].position;
        const Vec3& p01 = state.anchors[patch.anchors[2]].position;
        const Vec3& p11 = state.anchors[patch.anchors[3]].position;

        // Get patch state (coefficients + residuals)
        const auto& coeffs = patch.coefficients.scalars;
        const auto& residuals = patch.coefficients.residuals;

        for (std::uint32_t i = 0; i <= grid_resolution_; ++i) {
            for (std::uint32_t j = 0; j <= grid_resolution_; ++j) {
                float u = i * step;
                float v = j * step;

                Vec3 pos = evaluate_patch_position(p00, p10, p01, p11,
                                                    patch.basis, coeffs, residuals, u, v);
                Vec3 normal = evaluate_patch_normal(p00, p10, p01, p11,
                                                     patch.basis, coeffs, residuals, u, v);

                Surfel s;
                s.position = pos;
                s.normal = normal;
                s.radius = avg_radius;
                s.material = patch.material;
                s.source_patch = pid;
                s.r = 200; s.g = 200; s.b = 200; s.a = 255;
                cache.push_back(s);
            }
        }

        // Mark this patch as updated for temporal coherence
        cache.mark_patch_updated(pid);
    }

    cache.set_source_generation(state.generation);
    cache.set_topology_generation(topology.topology_generation());

    return true;
}

} // namespace delirium::m3