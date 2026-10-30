#include "delirium_m7/collision_response.hpp"
#include <algorithm>

namespace delirium::m7 {

CollisionResponse::CollisionResponse(CollisionParams params) : params_(std::move(params)) {}

Vec3 CollisionResponse::compute_impulse(const ContactPair& pair, const BodyState& state_a,
                                         const BodyState& state_b, float restitution) const {
    AnchorId nearest_a = 0, nearest_b = 0;
    float min_dist_a = 1e30f, min_dist_b = 1e30f;

    for (std::size_t i = 0; i < state_a.anchors.size(); ++i) {
        float d = length(state_a.anchors[i].position - pair.contact_point);
        if (d < min_dist_a) { min_dist_a = d; nearest_a = static_cast<AnchorId>(i); }
    }
    for (std::size_t i = 0; i < state_b.anchors.size(); ++i) {
        float d = length(state_b.anchors[i].position - pair.contact_point);
        if (d < min_dist_b) { min_dist_b = d; nearest_b = static_cast<AnchorId>(i); }
    }

    Vec3 rel_vel = state_a.anchors[nearest_a].velocity - state_b.anchors[nearest_b].velocity;
    float rel_vel_n = dot(rel_vel, pair.normal);

    if (rel_vel_n < 0.0f) return Vec3{}; // Moving apart along normal

    float mass_a = 1.0f, mass_b = 1.0f;
    float inv_mass_sum = 1.0f / (mass_a + mass_b);

    float j = (1.0f + restitution) * rel_vel_n;
    j *= inv_mass_sum;

    j = std::max(-params_.impulse_clamp, std::min(params_.impulse_clamp, j));

    return pair.normal * j; // Impulse on body_b along normal
}

Vec3 CollisionResponse::compute_friction_impulse(const Vec3& normal_impulse, const Vec3& normal,
                                                   float friction) const {
    // Extract tangential component and clamp by Coulomb friction cone
    Vec3 tangent = normal_impulse - normal * dot(normal_impulse, normal);
    float tangent_len = length(tangent);
    float max_friction = friction * length(normal_impulse);

    if (tangent_len > max_friction) {
        tangent *= max_friction / tangent_len;
    }
    return tangent;
}

void CollisionResponse::apply_impulse(BodyState& state, AnchorId anchor, const Vec3& impulse) const {
    if (anchor < state.anchors.size()) {
        state.anchors[anchor].velocity += impulse;
    }
}

void CollisionResponse::resolve(BodyState& state_a, BodyState& state_b,
                                 const std::vector<ContactPair>& contacts,
                                 float /*fixed_dt*/) const {
    for (const auto& cp : contacts) {
        Vec3 impulse = compute_impulse(cp, state_a, state_b, params_.restitution);
        Vec3 friction = compute_friction_impulse(impulse, cp.normal, params_.friction);

        AnchorId nearest_a = 0, nearest_b = 0;
        float min_dist_a = 1e30f, min_dist_b = 1e30f;

        for (std::size_t i = 0; i < state_a.anchors.size(); ++i) {
            float d = length(state_a.anchors[i].position - cp.contact_point);
            if (d < min_dist_a) { min_dist_a = d; nearest_a = static_cast<AnchorId>(i); }
        }
        for (std::size_t i = 0; i < state_b.anchors.size(); ++i) {
            float d = length(state_b.anchors[i].position - cp.contact_point);
            if (d < min_dist_b) { min_dist_b = d; nearest_b = static_cast<AnchorId>(i); }
        }

        // Impulse on body_b along normal, opposite on body_a
        apply_impulse(state_b, nearest_b, impulse + friction);
        apply_impulse(state_a, nearest_a, -(impulse + friction));

        // Positional correction — push bodies apart along normal
        Vec3 correction = cp.normal * cp.penetration * 0.5f;
        state_b.anchors[nearest_b].position += correction;
        state_a.anchors[nearest_a].position -= correction;
    }
}

} // namespace delirium::m7