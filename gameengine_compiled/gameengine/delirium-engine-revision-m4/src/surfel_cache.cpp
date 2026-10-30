#include "delirium_m4/surfel_cache.hpp"
#include "delirium_m4/basis_functions.hpp"

#include <stdexcept>

namespace delirium::m4 {

// ---------------------------------------------------------------------------
// SurfelCache
// ---------------------------------------------------------------------------

const Surfel& SurfelCache::surfel(std::size_t index) const {
    if (index >= surfels_.size()) {
        static const Surfel empty;
        return empty;
    }
    return surfels_[index];
}

Surfel& SurfelCache::surfel(std::size_t index) {
    if (index >= surfels_.size()) {
        static Surfel empty;
        return empty;
    }
    return surfels_[index];
}

void SurfelCache::reserve(std::size_t capacity) {
    surfels_.reserve(capacity);
    patch_updated_.resize(capacity, false);
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
    while (patch_updated_.size() <= patch) {
        patch_updated_.push_back(false);
    }
    patch_updated_[patch] = true;
}

bool SurfelCache::patch_was_updated(PatchId patch) const {
    if (patch >= patch_updated_.size()) return false;
    return patch_updated_[patch];
}

void SurfelCache::clear_patch_updates() {
    std::fill(patch_updated_.begin(), patch_updated_.end(), false);
}

// ---------------------------------------------------------------------------
// CacheUpdater
// ---------------------------------------------------------------------------

CacheUpdater::CacheUpdater(std::uint32_t grid_resolution)
    : grid_resolution_(grid_resolution) {}

bool CacheUpdater::update(SurfelCache& cache,
                           const BodyState& state,
                           const BodyTopology& topology,
                           const MaterialRegistry& materials,
                           std::string* error) {
    if (!topology.finalized()) {
        if (error) *error = "topology must be finalized";
        return false;
    }

    // Check generation mismatch — full rebuild needed
    if (!cache.matches_generation(state.generation, state.topology_generation)) {
        return rebuild(cache, state, topology, materials, error);
    }

    // Incremental update: only dirty patches
    auto dirty_indices = state.dirty_patches.indices();
    if (dirty_indices.empty()) {
        cache.set_source_generation(state.generation);
        cache.set_topology_generation(state.topology_generation);
        return true;
    }

    // For dirty patches, mark as updated
    for (PatchId pid : dirty_indices) {
        cache.mark_patch_updated(pid);
    }

    cache.set_source_generation(state.generation);
    cache.set_topology_generation(state.topology_generation);

    return true;
}

bool CacheUpdater::rebuild(SurfelCache& cache,
                            const BodyState& state,
                            const BodyTopology& topology,
                            const MaterialRegistry& materials,
                            std::string* error) {
    if (!topology.finalized()) {
        if (error) *error = "topology must be finalized";
        return false;
    }

    cache.clear();
    cache.clear_patch_updates();

    float step = 1.0f / grid_resolution_;

    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const PatchTopology& patch = topology.patch(pid);
        const MaterialProperties& mat = materials.get(patch.material);

        for (std::uint32_t i = 0; i <= grid_resolution_; ++i) {
            for (std::uint32_t j = 0; j <= grid_resolution_; ++j) {
                float u = i * step;
                float v = j * step;

                const Vec3& p00 = state.anchors[patch.anchors[0]].position;
                const Vec3& p10 = state.anchors[patch.anchors[1]].position;
                const Vec3& p01 = state.anchors[patch.anchors[2]].position;
                const Vec3& p11 = state.anchors[patch.anchors[3]].position;

                Vec3 pos = evaluate_patch_position(p00, p10, p01, p11,
                                                    patch.basis,
                                                    patch.coefficients.scalars,
                                                    patch.coefficients.residuals,
                                                    u, v);
                Vec3 normal = evaluate_patch_normal(p00, p10, p01, p11,
                                                     patch.basis,
                                                     patch.coefficients.scalars,
                                                     patch.coefficients.residuals,
                                                     u, v);

                Surfel s;
                s.position = pos;
                s.normal = normal;
                s.radius = step * 0.5f;
                s.r = mat.r;
                s.g = mat.g;
                s.b = mat.b;
                s.a = 255;
                s.roughness = mat.roughness;
                s.metalness = mat.metalness;
                s.material = patch.material;
                s.source_patch = pid;

                cache.push_back(s);
            }
        }
    }

    cache.set_source_generation(state.generation);
    cache.set_topology_generation(state.topology_generation);

    return true;
}

} // namespace delirium::m4