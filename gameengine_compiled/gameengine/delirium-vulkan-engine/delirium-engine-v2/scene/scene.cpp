#include "scene.h"
#include <cmath>
#include <cstring>

namespace delirium {

// ---------------------------------------------------------------------------
// Scene implementation
// ---------------------------------------------------------------------------

int Scene::add_body(const Body& b) {
    if(body_count >= 128) return -1;
    bodies[body_count] = b;
    bodies[body_count].id = body_count;
    return body_count++;
}

void Scene::update_render_register() {
    render_register.reset();
    for(int i = 0; i < body_count; i++) {
        auto& body = bodies[i];
        body.update_render_register();
        // Merge body's render register into scene register
        for(int j = 0; j < body.render_coeffs.count; j++) {
            render_register.add_coefficient(body.render_coeffs.coeffs[j]);
        }
    }
    render_register.mark_all_dirty();
}

void Scene::frustum_cull() {
    // Basic frustum cull — check if body bounding box is in view
    // TODO: implement proper frustum test
}

void Scene::scene_bounding_box(vec3& min_out, vec3& max_out) const {
    min_out = vec3(1e30f); max_out = vec3(-1e30f);
    for(int i = 0; i < body_count; i++) {
        vec3 bmin, bmax;
        bodies[i].world_bounding_box(bmin, bmax);
        min_out = vec3(fminf(min_out.x, bmin.x), fminf(min_out.y, bmin.y), fminf(min_out.z, bmin.z));
        max_out = vec3(fmaxf(max_out.x, bmax.x), fmaxf(max_out.y, bmax.y), fmaxf(max_out.z, bmax.z));
    }
}

} // namespace delirium