#pragma once

#include "types.h"
#include "body_topology.h"
#include "body_state.h"
#include "material_registry.h"
#include <string>
#include <vector>

namespace delirium {

/* ── Sparse lateral raycasting surfel cache ────────────────────────
 *
 * Surfels are NOT body-fixed.  They are regenerated every frame from
 * the camera's perspective: rays march laterally across the viewport,
 * hit the surface via distance sampling, and emit surfels at those
 * hit points.  Back-to-front sort for correct blending.
 *
 * When surfels are sparse (small radius), adjacent surfels average
 * color/normal to fill visual gaps — no hard edges between points.
 * ────────────────────────────────────────────────────────────────── */
struct RayCamera {
    Vec3 origin{}, forward{}, right{}, up{};
    float fov_y = 1.0f;       // tan(fov/2)
    float aspect = 1.33f;
};

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
    /* Legacy: uniform patch sampling (for backward compat). */
    bool rebuild(SurfelCache& cache, const BodyState& state,
                 const BodyTopology& topology,
                 const MaterialRegistry& materials,
                 std::string* error = nullptr);

    /* Camera-dependent lateral raycasting surfel generation.
     * Rays are cast from camera, march through scene, emit surfels
     * at surface hits.  Sparse grid with color averaging. */
    bool rebuild_with_lod(SurfelCache& cache, const BodyState& state,
                          const BodyTopology& topology,
                          const MaterialRegistry& materials,
                          const RayCamera& cam,
                          std::string* error = nullptr);

    /* Legacy overload (delegates to uniform sampling). */
    bool rebuild_with_lod(SurfelCache& cache, const BodyState& state,
                          const BodyTopology& topology,
                          const MaterialRegistry& materials,
                          float cam_dist,
                          std::string* error = nullptr);

private:
    /* Evaluate patch surface point (bilinear + 8-term deformation). */
    Vec3 eval_patch(const BodyState& state, const PatchTopology& patch,
                    float u, float v) const;

    /* Normal via bilinear tangents. */
    Vec3 eval_normal(const BodyState& state, const PatchTopology& patch,
                     float u, float v) const;

    /* Distance from point P to patch surface (grid search). */
    float patch_distance(const BodyState& state, const PatchTopology& patch,
                         Vec3 P, float samples,
                         Vec3& closest, float& cu, float& cv) const;

    /* Bounding sphere culling for patch. */
    float patch_bbox_dist(Vec3 P, const Vec3& A, const Vec3& B,
                          const Vec3& C, const Vec3& D) const;
};

} // namespace delirium