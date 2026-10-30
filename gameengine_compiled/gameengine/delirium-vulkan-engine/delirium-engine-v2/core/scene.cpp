#include "scene.h"
#include <algorithm>

namespace delirium {

BodyId Scene::add_body(BodyRuntime* body) {
    BodyId id = next_id_++;
    if (body)
        bodies_.push_back(body);
    return id;
}

void Scene::remove_body(BodyId id) {
    // Remove by matching body_id() — not by index
    bodies_.erase(
        std::remove_if(bodies_.begin(), bodies_.end(),
                       [id](BodyRuntime* b) { return b && b->body_id() == id; }),
        bodies_.end());
}

BodyRuntime* Scene::get_body(BodyId id) {
    for (auto* b : bodies_)
        if (b && b->body_id() == id) return b;
    return nullptr;
}

const BodyRuntime* Scene::get_body(BodyId id) const {
    for (const auto* b : bodies_)
        if (b && b->body_id() == id) return b;
    return nullptr;
}

bool Scene::step(float dt, std::string* error) {
    for (auto* b : bodies_) {
        if (b && !b->step(dt, error))
            return false;
    }
    return true;
}

bool Scene::rebuild_cache(SurfelCache& cache, float cam_dist, std::string* error) {
    cache.surfels_mut().clear();
    for (auto* b : bodies_) {
        if (b && !b->rebuild_cache_with_lod(cache, cam_dist, error))
            return false;
    }
    return true;
}

Vec3 Scene::bounding_center() const {
    if (bodies_.empty()) return {};
    Vec3 sum{};
    for (const auto* b : bodies_)
        if (b) sum = sum + b->bounding_center();
    return sum * (1.0f / static_cast<float>(bodies_.size()));
}

float Scene::bounding_radius() const {
    Vec3 center = bounding_center();
    float max_r = 0.0f;
    for (const auto* b : bodies_) {
        if (!b) continue;
        float r = length(b->bounding_center() - center) + b->bounding_radius();
        if (r > max_r) max_r = r;
    }
    return max_r;
}

} // namespace delirium