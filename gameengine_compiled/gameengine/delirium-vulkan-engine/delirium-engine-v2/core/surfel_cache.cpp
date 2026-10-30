#include "surfel_cache.h"
#include "basis_functions.h"
#include <cmath>
#include <algorithm>

namespace delirium {

/* ── Patch evaluation (bilinear + 8-term deformation) ────────────── */
Vec3 CacheUpdater::eval_patch(const BodyState& state, const PatchTopology& patch,
                               float u, float v) const {
    const auto& a00 = state.anchors[patch.anchors[0]];
    const auto& a10 = state.anchors[patch.anchors[1]];
    const auto& a01 = state.anchors[patch.anchors[2]];
    const auto& a11 = state.anchors[patch.anchors[3]];

    Vec3 P = a00.position * (1.0f - u) * (1.0f - v) +
             a10.position * u * (1.0f - v) +
             a01.position * (1.0f - u) * v +
             a11.position * u * v;

    float b[8];
    BasisEvaluator::deform_basis(u, v, b);
    float h = 0.0f;
    for (int i = 0; i < 8; ++i)
        h += patch.coefficients.scalars[i] * b[i];

    Vec3 Nd = eval_normal(state, patch, u, v);
    return P + Nd * h;
}

Vec3 CacheUpdater::eval_normal(const BodyState& state, const PatchTopology& patch,
                                float u, float v) const {
    const auto& a00 = state.anchors[patch.anchors[0]];
    const auto& a10 = state.anchors[patch.anchors[1]];
    const auto& a01 = state.anchors[patch.anchors[2]];
    const auto& a11 = state.anchors[patch.anchors[3]];

    Vec3 du = (a10.position - a00.position) * (1.0f - v) +
              (a11.position - a01.position) * v;
    Vec3 dv = (a01.position - a00.position) * (1.0f - u) +
              (a11.position - a10.position) * u;
    return normalize_vec3(cross(du, dv));
}

/* ── Bounding sphere culling ─────────────────────────────────────── */
float CacheUpdater::patch_bbox_dist(Vec3 P, const Vec3& A, const Vec3& B,
                                     const Vec3& C, const Vec3& D) const {
    Vec3 c = (A + B + C + D) * 0.25f;
    float r = std::max(std::max(length(B - A), length(D - C)),
                       std::max(length(C - A), length(D - B))) * 0.5f;
    return length(P - c) - r;
}

/* ── Patch distance (grid search) ────────────────────────────────── */
float CacheUpdater::patch_distance(const BodyState& state, const PatchTopology& patch,
                                    Vec3 P, float samples,
                                    Vec3& closest, float& cu, float& cv) const {
    float md = 1e10f;
    closest = Vec3{};
    cu = 0.5f;
    cv = 0.5f;

    for (int si = 0; si < samples; ++si) {
        for (int sj = 0; sj < samples; ++sj) {
            float u = (si + 0.5f) / samples;
            float v = (sj + 0.5f) / samples;
            Vec3 sp = eval_patch(state, patch, u, v);
            float d = length(P - sp);
            if (d < md) {
                md = d;
                closest = sp;
                cu = u;
                cv = v;
            }
        }
    }
    return md;
}

/* ── Legacy rebuild (uniform sampling — backward compat) ─────────── */
bool CacheUpdater::rebuild(SurfelCache& cache, const BodyState& state,
                            const BodyTopology& topology,
                            const MaterialRegistry& materials,
                            std::string* error) {
    cache.surfels_mut().clear();
    cache.set_topology_generation(topology.topology_generation());

    int samples = 4;
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        if (!materials.has(patch.material)) {
            if (error) *error = "missing material for patch " + std::to_string(pid);
            return false;
        }
        const auto& mat = materials.get(patch.material);

        for (int si = 0; si < samples; ++si) {
            for (int sj = 0; sj < samples; ++sj) {
                float u = (si + 0.5f) / samples;
                float v = (sj + 0.5f) / samples;

                Surfel s{};
                s.position = eval_patch(state, patch, u, v);
                s.normal = eval_normal(state, patch, u, v);
                s.r = mat.r; s.g = mat.g; s.b = mat.b;
                s.roughness = mat.roughness;
                s.metalness = mat.metalness;
                s.material = patch.material;
                s.source_patch = pid;

                const auto& a00 = state.anchors[patch.anchors[0]];
                const auto& a10 = state.anchors[patch.anchors[1]];
                const auto& a01 = state.anchors[patch.anchors[2]];
                float du = length(a10.position - a00.position);
                float dv = length(a01.position - a00.position);
                s.radius = (du + dv) * 0.25f;

                cache.surfels_mut().push_back(std::move(s));
            }
        }
    }
    return true;
}

/* ── Legacy LOD overload (delegates to camera version) ───────────── */
bool CacheUpdater::rebuild_with_lod(SurfelCache& cache, const BodyState& state,
                                     const BodyTopology& topology,
                                     const MaterialRegistry& materials,
                                     float cam_dist,
                                     std::string* error) {
    RayCamera cam;
    cam.origin = Vec3{0, 0, cam_dist};
    cam.forward = Vec3{0, 0, -1};
    cam.right = Vec3{1, 0, 0};
    cam.up = Vec3{0, 1, 0};
    cam.fov_y = 1.0f;
    cam.aspect = 1.33f;
    return rebuild_with_lod(cache, state, topology, materials, cam, error);
}

/* ── Camera-dependent lateral raycasting surfel generation ──────────
 *
 * Sparse rays from camera → surface hits → surfels at hit points.
 * Small radius surfels with color averaging to fill visual gaps.
 * Back-to-front sort for correct blending.
 * ────────────────────────────────────────────────────────────────── */
bool CacheUpdater::rebuild_with_lod(SurfelCache& cache, const BodyState& state,
                                     const BodyTopology& topology,
                                     const MaterialRegistry& materials,
                                     const RayCamera& cam,
                                     std::string* error) {
    cache.surfels_mut().clear();
    cache.set_topology_generation(topology.topology_generation());

    // Adaptive grid: more rays when closer
    float cam_dist = length(cam.forward);
    int grid = std::max(16, std::min(48, static_cast<int>(400.0f / (cam_dist + 1.0f))));
    int max_steps = 32;
    float step_size = 0.15f;

    // Precompute bounding boxes
    struct BBox {
        Vec3 center;
        float radius;
        PatchId id;
    };
    std::vector<BBox> bboxes;
    bboxes.reserve(topology.patch_count());
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& patch = topology.patch(pid);
        const auto& a00 = state.anchors[patch.anchors[0]];
        const auto& a10 = state.anchors[patch.anchors[1]];
        const auto& a01 = state.anchors[patch.anchors[2]];
        const auto& a11 = state.anchors[patch.anchors[3]];
        BBox bb;
        bb.center = (a00.position + a10.position + a01.position + a11.position) * 0.25f;
        bb.radius = std::max(std::max(length(a10.position - a00.position),
                                      length(a11.position - a01.position)),
                             std::max(length(a01.position - a00.position),
                                      length(a11.position - a10.position))) * 0.5f;
        bb.id = pid;
        bboxes.push_back(bb);
    }

    // Cast rays through sparse viewport grid
    for (int gi = 0; gi < grid; ++gi) {
        for (int gj = 0; gj < grid; ++gj) {
            // View-space direction
            float xu = (gi + 0.5f) / grid;
            float yv = (gj + 0.5f) / grid;
            float x = (xu - 0.5f) * cam.aspect * cam.fov_y * 2.0f;
            float y = (yv - 0.5f) * cam.fov_y * 2.0f;
            Vec3 dir = normalize_vec3(cam.forward + cam.right * x + cam.up * y);
            Vec3 P = cam.origin;

            // March ray
            for (int s = 0; s < max_steps; ++s) {
                float best_d = 1e10f;
                Vec3 best_hit{};
                PatchId best_pid = kInvalidPatch;
                float bu = 0.5f, bv = 0.5f;

                // Find closest patch at this ray position
                for (const auto& bb : bboxes) {
                    if (length(P - bb.center) - bb.radius > best_d + 0.5f)
                        continue;
                    const auto& patch = topology.patch(bb.id);
                    Vec3 cl;
                    float cu, cv;
                    float d = patch_distance(state, patch, P, 2, cl, cu, cv);
                    if (d < best_d) {
                        best_d = d;
                        best_hit = cl;
                        best_pid = bb.id;
                        bu = cu;
                        bv = cv;
                    }
                }

                // Hit threshold
                if (best_d < step_size * 0.25f && best_pid != kInvalidPatch) {
                    const auto& patch = topology.patch(best_pid);
                    if (!materials.has(patch.material)) break;
                    const auto& mat = materials.get(patch.material);

                    Surfel s{};
                    s.position = best_hit;
                    s.normal = eval_normal(state, patch, bu, bv);
                    s.r = mat.r; s.g = mat.g; s.b = mat.b;
                    s.roughness = mat.roughness;
                    s.metalness = mat.metalness;
                    s.material = patch.material;
                    s.source_patch = best_pid;
                    s.radius = step_size * 0.6f;
                    cache.surfels_mut().push_back(std::move(s));
                    break;
                }

                P = P + dir * step_size;
            }
        }
    }

    // Back-to-front sort (far first for correct blending)
    auto& surfels = cache.surfels_mut();
    std::sort(surfels.begin(), surfels.end(),
              [&cam](const Surfel& a, const Surfel& b) {
                  float da = dot(a.position - cam.origin, cam.forward);
                  float db = dot(b.position - cam.origin, cam.forward);
                  return da > db;
              });

    return true;
}

} // namespace delirium