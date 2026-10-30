#include "delirium_m7/surfel_cache.hpp"
#include <cmath>

namespace delirium::m7 {

void CacheUpdater::emit_surfel(const BodyState& state, const PatchTopology& patch,
                                const MaterialProperties& mat, float u, float v,
                                SurfelCache& cache) {
    Surfel s;
    s.r = mat.r; s.g = mat.g; s.b = mat.b; s.a = 255;
    s.roughness = mat.roughness; s.metalness = mat.metalness;
    s.material = patch.material; s.source_patch = static_cast<PatchId>(cache.surfels().size());

    float bu00 = (1.0f - u) * (1.0f - v);
    float bu10 = u * (1.0f - v);
    float bu01 = (1.0f - u) * v;
    float bu11 = u * v;

    const auto& a00 = state.anchors[patch.anchors[0]];
    const auto& a10 = state.anchors[patch.anchors[1]];
    const auto& a01 = state.anchors[patch.anchors[2]];
    const auto& a11 = state.anchors[patch.anchors[3]];

    s.position = a00.position * bu00 + a10.position * bu10 +
                 a01.position * bu01 + a11.position * bu11;

    Vec3 du = (a10.position - a00.position) * (1.0f - v) +
              (a11.position - a01.position) * v;
    Vec3 dv = (a01.position - a00.position) * (1.0f - u) +
              (a11.position - a10.position) * u;
    s.normal = normalize_vec3(cross(du, dv));
    s.radius = length(du) * 0.5f;

    cache.surfels_mut().push_back(s);
}

bool CacheUpdater::rebuild(SurfelCache& cache, const BodyState& state,
                            const BodyTopology& topology,
                            const MaterialRegistry& materials, std::string* error) {
    cache.set_topology_generation(state.topology_generation);
    cache.surfels_mut().clear();

    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        if (!materials.has(patch.material)) {
            if (error) *error = "missing material";
            return false;
        }
        const auto& mat = materials.get(patch.material);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float u = (i + 0.5f) / 4.0f;
                float v = (j + 0.5f) / 4.0f;
                emit_surfel(state, patch, mat, u, v, cache);
            }
        }
    }
    return true;
}

bool CacheUpdater::rebuild_with_lod(SurfelCache& cache, const BodyState& state,
                                     const BodyTopology& topology,
                                     const MaterialRegistry& materials,
                                     float cam_dist, std::string* error) {
    cache.set_topology_generation(state.topology_generation);
    cache.surfels_mut().clear();

    int samples = cam_dist < 10.0f ? 4 : (cam_dist < 50.0f ? 2 : 1);

    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        if (!materials.has(patch.material)) {
            if (error) *error = "missing material";
            return false;
        }
        const auto& mat = materials.get(patch.material);

        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                float u = (i + 0.5f) / samples;
                float v = (j + 0.5f) / samples;
                emit_surfel(state, patch, mat, u, v, cache);
            }
        }
    }
    return true;
}

} // namespace delirium::m7