#include "delirium_m8/frustum_culler.hpp"
#include <cmath>

namespace delirium::m8 {

std::vector<BodyId> FrustumCuller::cull(
    const std::vector<BodyId>& active_bodies,
    const std::vector<Vec3>& body_centers,
    const Frustum& frustum) const
{
    std::vector<BodyId> visible;
    visible.reserve(active_bodies.size());

    for (std::size_t i = 0; i < active_bodies.size(); ++i) {
        if (is_visible(body_centers[i], 1.0f, frustum)) {
            visible.push_back(active_bodies[i]);
        }
    }
    return visible;
}

bool FrustumCuller::is_visible(const Vec3& center, float radius, const Frustum& frustum) const {
    Vec3 fwd = normalize_vec3(frustum.forward);
    Vec3 right = normalize_vec3(frustum.right);
    Vec3 up = normalize_vec3(frustum.up);

    Vec3 rel = center - frustum.origin;
    float z = dot(fwd, rel);

    // Near/far (conservative sphere test)
    if (z + radius < frustum.near_plane) return false;
    if (z - radius > frustum.far_plane) return false;

    float tan_hfov = std::tan(frustum.fov_y / 2.0f) * frustum.aspect;
    float tan_vfov = std::tan(frustum.fov_y / 2.0f);

    float x = std::abs(dot(right, rel));
    float y = std::abs(dot(up, rel));

    // At distance z, frustum half-width = z * tan_hfov
    // Sphere is outside if |offset| - radius > frustum_half_extent
    if (x - radius > z * tan_hfov) return false;
    if (y - radius > z * tan_vfov) return false;

    return true;
}

} // namespace delirium::m8