#pragma once
#include "../../core/geometry/body.h"
#include "../../core/render/surfel_cache.h"

namespace delirium {

/// Scene — collection of bodies with spatial organization.
struct Scene {
    Body bodies[128];
    int body_count = 0;

    // Combined render register (all bodies merged)
    CoefficientRegister render_register;

    // Camera
    Camera camera;

    // Light position
    vec3 light_pos;

    Scene() {
        light_pos = vec3(5, 5, 5);
    }

    /// Add body to scene
    int add_body(const Body& b);

    /// Update combined render register from all bodies
    void update_render_register();

    /// Cull bodies outside camera frustum (basic)
    void frustum_cull();

    /// Get bounding box of entire scene
    void scene_bounding_box(vec3& min, vec3& max) const;
};

} // namespace delirium