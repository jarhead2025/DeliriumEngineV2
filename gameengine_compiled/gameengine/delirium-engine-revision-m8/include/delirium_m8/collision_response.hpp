#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_state.hpp"
#include <vector>
namespace delirium::m8 {
class CollisionResponse {
public:
    explicit CollisionResponse(CollisionParams params = {});
    void resolve(BodyState& state_a, BodyState& state_b,
                 const std::vector<ContactPair>& contacts, float fixed_dt) const;
    void apply_impulse(BodyState& state, AnchorId anchor, const Vec3& impulse) const;
    Vec3 compute_impulse(const ContactPair& pair, const BodyState& state_a,
                         const BodyState& state_b, float restitution) const;
    Vec3 compute_friction_impulse(const Vec3& normal_impulse, const Vec3& normal,
                                  float friction) const;
private:
    CollisionParams params_;
};
} // namespace delirium::m8