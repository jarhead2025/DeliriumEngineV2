#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_state.hpp"
#include <vector>
namespace delirium::m8 {
class ExternalForces {
public:
    explicit ExternalForces(ExternalForce force = {});
    void set_force(ExternalForce f);
    void add_point_force(PointForce pf);
    void clear_point_forces();
    void apply(BodyState& state, float fixed_dt) const;
    void apply_gravity(BodyState& state, float fixed_dt) const;
    void apply_wind(BodyState& state, float fixed_dt) const;
    void apply_drag(BodyState& state) const;
    void apply_point_forces(BodyState& state, float fixed_dt) const;
private:
    ExternalForce force_;
    std::vector<PointForce> point_forces_;
};
} // namespace delirium::m8