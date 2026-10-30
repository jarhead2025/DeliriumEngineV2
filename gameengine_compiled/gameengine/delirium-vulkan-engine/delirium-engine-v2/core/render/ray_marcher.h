#pragma once
#include "../math/basis.h"
#include "../math/coefficient.h"
#include "surfel_cache.h"

namespace delirium {

/// Ray marcher — the core rendering loop.
/// Per frame: for each pixel (W×H threads), march far→near.
/// Fixed compute: W × H × N_steps operations. No scene complexity penalty.
struct RayMarcher {
    /// Number of depth steps per ray
    int steps;

    /// Step size in world units
    float step_size;

    /// Surface threshold
    float threshold;

    /// Maximum bounce depth for reflections
    int max_bounces;

    RayMarcher() : steps(32), step_size(0.1f), threshold(0.5f), max_bounces(2) {}

    /// March a single ray against coefficient register.
    /// Returns final pixel color.
    /// Logic: far→near, closer hits override farther ones (z-min).
    vec3 march(const Ray& ray, const CoefficientRegister& reg, const vec3& light_pos);

    /// March with bounce (lateral rays for reflections).
    /// All lighting is simple vector additions — no complex shading.
    vec3 march_with_bounce(const Ray& ray, const CoefficientRegister& reg,
                           const vec3& light_pos, int bounce_depth);

    /// Render a full frame to pixel buffer.
    /// Output is RGBA float array of width*height elements.
    void render_frame(const Camera& cam, const CoefficientRegister& reg,
                      const vec3& light_pos, float* out_rgba);
};

} // namespace delirium