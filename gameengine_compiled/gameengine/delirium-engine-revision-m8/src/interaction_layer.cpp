#include "delirium_m8/interaction_layer.hpp"
#include <algorithm>
#include <cmath>

namespace delirium::m8 {

PickResult InteractionLayer::pick(const std::vector<BodyRuntime*>& bodies,
                                   const Vec3& ray_origin, const Vec3& ray_dir,
                                   float max_dist)
{
    PickResult best{};
    best.distance = max_dist;

    for (auto* body : bodies) {
        if (!body) continue;
        PickResult r = pick_patch(body, ray_origin, ray_dir, max_dist);
        if (r.body_id != kInvalidBody && r.distance < best.distance) {
            best = r;
        }
    }
    return best;
}

PickResult InteractionLayer::pick_patch(const BodyRuntime* body,
                                         const Vec3& ray_origin, const Vec3& ray_dir,
                                         float max_dist)
{
    PickResult best{};
    best.distance = max_dist;
    const auto& state = body->read_state();

    // Iterate over anchors as hit points (each anchor is a point on the surface)
    for (std::size_t ai = 0; ai < state.anchors.size(); ++ai) {
        const auto& anchor = state.anchors[ai];
        Vec3 hit;
        float d = ray_surfel_intersect(ray_origin, ray_dir, anchor.position, 0.1f, hit);
        if (d > 0.0f && d < best.distance) {
            best.body_id = body->body_id();
            best.patch_id = 0;
            best.distance = d;
            best.hit_point = hit;
            best.normal = {0, 1, 0}; // default normal
        }
    }
    return best;
}

float InteractionLayer::ray_surfel_intersect(const Vec3& origin, const Vec3& dir,
                                              const Vec3& target, float radius,
                                              Vec3& hit_out) const
{
    Vec3 to_target = target - origin;
    float t = dot(to_target, dir);
    Vec3 closest = origin + dir * t;
    float dist = length(closest - target);
    if (dist < radius) {
        hit_out = closest;
        return t;
    }
    return -1.0f;
}

void InteractionLayer::grab(BodyRuntime* body, AnchorId anchor, const Vec3& target_pos) {
    if (!body) return;
    auto& state = body->write_state();
    if (anchor < state.anchors.size()) {
        state.anchors[anchor].position = target_pos;
        state.anchors[anchor].velocity = {0, 0, 0};
    }
}

void InteractionLayer::cut(BodyRuntime* body, const Vec3& cut_normal, const Vec3& cut_point) {
    if (!body) return;
    auto& state = body->write_state();
    for (auto& a : state.anchors) {
        Vec3 to_anchor = a.position - cut_point;
        float d = dot(to_anchor, cut_normal);
        if (d < 0.0f) {
            a.position = cut_point + cut_normal * 0.01f;
        }
    }
}

void InteractionLayer::weld(BodyRuntime* body_a, BodyRuntime* body_b,
                             AnchorId anchor_a, AnchorId anchor_b, float stiffness) {
    if (!body_a || !body_b) return;
    Constraint c{};
    c.type = ConstraintType::Fixed;
    c.anchor_a = anchor_a;
    c.anchor_b = anchor_b;
    c.stiffness = stiffness;
    body_a->add_constraint(c);
}

void InteractionLayer::heat(BodyRuntime* body, AnchorId anchor, float temperature) {
    if (!body) return;
    auto& state = body->write_state();
    if (anchor < state.anchors.size()) {
        state.anchors[anchor].temperature = temperature;
    }
}

} // namespace delirium::m8