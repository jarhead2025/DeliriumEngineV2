#include "delirium/geometry/patch/patch.hpp"
#include <algorithm>
#include <cmath>

namespace delirium {

/// --- Bilinear interpolation of anchor positions ---
vec3 Patch::evaluate_bilinear(float u, float v,
                               const std::span<const AnchorRecord>& anchors) const {
    // Clamp to [0,1] to avoid extrapolation artifacts
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);

    // Current positions (rest + displacement)
    const auto& a00 = anchors[anchor_indices[0]];
    const auto& a01 = anchors[anchor_indices[1]];
    const auto& a10 = anchors[anchor_indices[2]];
    const auto& a11 = anchors[anchor_indices[3]];

    vec3 p00 = a00.current_position();
    vec3 p01 = a01.current_position();
    vec3 p10 = a10.current_position();
    vec3 p11 = a11.current_position();

    // Bilinear: (1-u)(1-v)·A₀ + u(1-v)·A₁ + (1-u)v·A₂ + uv·A₃
    float w00 = (1.0f - u) * (1.0f - v);
    float w01 = u * (1.0f - v);
    float w10 = (1.0f - u) * v;
    float w11 = u * v;

    return p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11;
}

/// --- Normal derivation from finite differences ---
vec3 Patch::evaluate_normal(float u, float v,
                             const std::span<const AnchorRecord>& anchors) const {
    // Sample tangent vectors via finite differences
    float eps = 0.001f;

    vec3 p_center = evaluate_bilinear(u, v, anchors);
    vec3 p_u = evaluate_bilinear(u + eps, v, anchors);
    vec3 p_v = evaluate_bilinear(u, v + eps, anchors);

    vec3 tangent_u = (p_u - p_center) / eps;
    vec3 tangent_v = (p_v - p_center) / eps;

    // Normal = cross(tangent_u, tangent_v), normalized
    return tangent_u.cross(tangent_v).normalized();
}

/// --- Full surface evaluation ---
SurfaceSample Patch::evaluate(float u, float v,
                               const std::span<const AnchorRecord>& anchors) const {
    SurfaceSample sample;

    // 1. Bilinear base position
    vec3 base_pos = evaluate_bilinear(u, v, anchors);

    // 2. Shape layer displacement (along surface normal)
    sample.shape_val = evaluate_shape(u, v, std::span(surface.shape_K, 8));

    // 3. Detail layer relief
    sample.detail_val = evaluate_detail(u, v, std::span(surface.detail_K, 8));

    // 4. Total displacement along normal
    float total_displacement = sample.shape_val + sample.detail_val;
    vec3 normal = evaluate_normal(u, v, anchors);
    sample.position = base_pos + normal * total_displacement;

    // 5. Normal (re-derived at displaced position for accuracy)
    sample.normal = normal;

    // 6. Color layer
    // color_K stores 8 bases × 3 channels = 24 floats, but we have 8 slots
    // Each slot is used as a scalar multiplier for the RGB base color
    sample.color = evaluate_color(u, v, std::span(surface.color_K, 8));

    return sample;
}

/// --- Hit test: is there surface at world_pos? ---
bool Patch::test_hit(const vec3& world_pos,
                     const std::span<const AnchorRecord>& anchors,
                     float& out_u, float& out_v, float& out_distance) const {
    // Brute-force: sample the patch on a grid, find closest point
    // TODO: replace with analytic inverse when basis functions allow it
    const int grid = 16; // 16×16 = 256 samples per patch
    float best_dist = 1e9f;
    float best_u = 0, best_v = 0;

    for (int i = 0; i <= grid; ++i) {
        for (int j = 0; j <= grid; ++j) {
            float u = (float)i / grid;
            float v = (float)j / grid;

            vec3 pos = evaluate(u, v, anchors).position;
            vec3 diff = world_pos - pos;
            float d = diff.length();

            if (d < best_dist) {
                best_dist = d;
                best_u = u;
                best_v = v;
            }
        }
    }

    out_u = best_u;
    out_v = best_v;
    out_distance = best_dist;

    // Hit if within threshold (patch radius / 10)
    AABB bounds = compute_bounds(anchors);
    vec3 extent = {bounds.max.x - bounds.min.x,
                   bounds.max.y - bounds.min.y,
                   bounds.max.z - bounds.min.z};
    float threshold = std::max({extent.x, extent.y, extent.z}) * 0.1f;

    return best_dist < threshold;
}

/// --- AABB bounds ---
Patch::AABB Patch::compute_bounds(const std::span<const AnchorRecord>& anchors) const {
    AABB box;
    box.min = anchors[anchor_indices[0]].current_position();
    box.max = box.min;

    for (int i = 0; i < 4; ++i) {
        vec3 p = anchors[anchor_indices[i]].current_position();
        box.min.x = std::min(box.min.x, p.x);
        box.min.y = std::min(box.min.y, p.y);
        box.min.z = std::min(box.min.z, p.z);
        box.max.x = std::max(box.max.x, p.x);
        box.max.y = std::max(box.max.y, p.y);
        box.max.z = std::max(box.max.z, p.z);
    }

    // Inflate by max displacement coefficient
    float max_disp = 0;
    for (int i = 0; i < 8; ++i) {
        max_disp += std::abs(surface.shape_K[i]);
        max_disp += std::abs(surface.detail_K[i]);
    }
    box.min -= vec3(max_disp, max_disp, max_disp);
    box.max += vec3(max_disp, max_disp, max_disp);

    return box;
}

} // namespace delirium