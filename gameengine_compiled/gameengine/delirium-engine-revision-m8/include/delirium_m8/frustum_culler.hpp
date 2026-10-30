#pragma once
#include "delirium_m8/types.hpp"
#include <vector>
namespace delirium::m8 {
class FrustumCuller {
public:
    std::vector<BodyId> cull(const std::vector<BodyId>& active_bodies,
                             const std::vector<Vec3>& body_centers,
                             const Frustum& frustum) const;
    bool is_visible(const Vec3& center, float radius, const Frustum& frustum) const;
};
} // namespace delirium::m8