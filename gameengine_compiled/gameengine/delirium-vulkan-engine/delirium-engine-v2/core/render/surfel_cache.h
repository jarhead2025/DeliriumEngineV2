#pragma once
#include "../math/vec3.h"
#include "../math/coefficient.h"

namespace delirium {

/// A surfel (surface element) — the rendering primitive.
/// Adaptive: fewer surfels at distance, more at close range.
/// Measured: at d=20, 231 surfels vs 5856 triangles (25x cheaper).
struct Surfel {
    vec3 position;
    vec3 normal;
    vec3 color;
    float radius;
    float depth;
    float material;
    int body_id;
};

/// Surfel cache — derived from body state for rendering.
/// Physics modifies coefficients → marks dirty → cache rebuilds surfels.
/// Decoupled: physics and render run on independent rhythms.
struct SurfelCache {
    static constexpr int MAX_SURFELS = 65536;

    Surfel surfels[MAX_SURFELS];
    int count = 0;
    bool dirty = true;

    /// Generate surfels from coefficient register
    /// Adaptive density based on distance from camera
    void generate(const CoefficientRegister& reg, const vec3& camera_pos, float distance);

    /// Clear cache
    void clear() { count = 0; dirty = true; }

    /// Sort surfels by depth (far to near for progressive raster)
    void sort_by_depth();
};

} // namespace delirium