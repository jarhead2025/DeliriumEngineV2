#include "delirium_m6/surfel_cache.hpp"
#include "delirium_m6/basis_functions.hpp"
namespace delirium::m6 {
const Surfel& SurfelCache::surfel(std::size_t i) const {
    if (i >= surfels_.size()) { static const Surfel e; return e; }
    return surfels_[i];
}
Surfel& SurfelCache::surfel(std::size_t i) {
    if (i >= surfels_.size()) { static Surfel e; return e; }
    return surfels_[i];
}
void SurfelCache::reserve(std::size_t c) { surfels_.reserve(c); patch_updated_.resize(c, false); }
void SurfelCache::clear() { surfels_.clear(); source_generation_ = 0; topology_generation_ = 0; patch_updated_.clear(); }
void SurfelCache::push_back(const Surfel& s) { surfels_.push_back(s); }
bool SurfelCache::matches_generation(Generation bg, TopologyGeneration tg) const {
    return source_generation_ == bg && topology_generation_ == tg;
}
void SurfelCache::mark_patch_updated(PatchId p) {
    while (patch_updated_.size() <= p) patch_updated_.push_back(false);
    patch_updated_[p] = true;
}
bool SurfelCache::patch_was_updated(PatchId p) const {
    return p < patch_updated_.size() && patch_updated_[p];
}
void SurfelCache::clear_patch_updates() { std::fill(patch_updated_.begin(), patch_updated_.end(), false); }

CacheUpdater::CacheUpdater(std::uint32_t r) : grid_resolution_(r) {}
bool CacheUpdater::rebuild(SurfelCache& cache, const BodyState& state, const BodyTopology& topology,
                            const MaterialRegistry& materials, std::string* error) {
    return rebuild_with_lod(cache, state, topology, materials, 1.0f, error);
}
bool CacheUpdater::rebuild_with_lod(SurfelCache& cache, const BodyState& state, const BodyTopology& topology,
                                     const MaterialRegistry& materials, float cam_dist, std::string* error) {
    if (!topology.finalized()) { if (error) *error = "topology not finalized"; return false; }
    cache.clear(); cache.clear_patch_updates();
    std::uint32_t res = grid_resolution_;
    if (cam_dist > 10.0f) res = std::max(1u, grid_resolution_ / 4);
    else if (cam_dist > 5.0f) res = std::max(2u, grid_resolution_ / 2);
    float step = 1.0f / res;
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        const auto& mat = materials.get(patch.material);
        for (std::uint32_t i = 0; i <= res; ++i) {
            for (std::uint32_t j = 0; j <= res; ++j) {
                float u = i*step, v = j*step;
                const Vec3& p00 = state.anchors[patch.anchors[0]].position;
                const Vec3& p10 = state.anchors[patch.anchors[1]].position;
                const Vec3& p01 = state.anchors[patch.anchors[2]].position;
                const Vec3& p11 = state.anchors[patch.anchors[3]].position;
                Vec3 pos = evaluate_patch_position(p00,p10,p01,p11,patch.basis,patch.coefficients.scalars,patch.coefficients.residuals,u,v);
                Vec3 normal = evaluate_patch_normal(p00,p10,p01,p11,patch.basis,patch.coefficients.scalars,patch.coefficients.residuals,u,v);
                Surfel s; s.position = pos; s.normal = normal; s.radius = step*0.5f;
                s.r = mat.r; s.g = mat.g; s.b = mat.b; s.a = 255;
                s.roughness = mat.roughness; s.metalness = mat.metalness;
                s.material = patch.material; s.source_patch = pid;
                cache.push_back(s);
            }
        }
    }
    cache.set_source_generation(state.generation);
    cache.set_topology_generation(state.topology_generation);
    return true;
}
} // namespace delirium::m6