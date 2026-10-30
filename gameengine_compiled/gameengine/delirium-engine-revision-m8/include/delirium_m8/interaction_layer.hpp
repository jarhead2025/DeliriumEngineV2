#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_runtime.hpp"
#include <vector>
namespace delirium::m8 {
class InteractionLayer {
public:
    PickResult pick(const std::vector<BodyRuntime*>& bodies,
                    const Vec3& ray_origin, const Vec3& ray_dir,
                    float max_dist = 100.0f);
    void grab(BodyRuntime* body, AnchorId anchor, const Vec3& target_pos);
    void cut(BodyRuntime* body, const Vec3& cut_normal, const Vec3& cut_point);
    void weld(BodyRuntime* body_a, BodyRuntime* body_b,
              AnchorId anchor_a, AnchorId anchor_b, float stiffness = 1.0f);
    void heat(BodyRuntime* body, AnchorId anchor, float temperature);
    PickResult pick_patch(const BodyRuntime* body,
                          const Vec3& ray_origin, const Vec3& ray_dir,
                          float max_dist = 100.0f);
private:
    float ray_surfel_intersect(const Vec3& origin, const Vec3& dir,
                               const Vec3& target, float radius,
                               Vec3& hit_out) const;
};
} // namespace delirium::m8