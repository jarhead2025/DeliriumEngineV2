#include "delirium_m8/external_forces.hpp"

namespace delirium::m8 {

ExternalForces::ExternalForces(ExternalForce force) : force_(std::move(force)) {}

void ExternalForces::set_force(ExternalForce f) {
    force_ = std::move(f);
}

void ExternalForces::add_point_force(PointForce pf) {
    point_forces_.push_back(std::move(pf));
}

void ExternalForces::clear_point_forces() {
    point_forces_.clear();
}

void ExternalForces::apply(BodyState& state, float fixed_dt) const {
    apply_gravity(state, fixed_dt);
    apply_wind(state, fixed_dt);
    apply_drag(state);
    apply_point_forces(state, fixed_dt);
}

void ExternalForces::apply_gravity(BodyState& state, float fixed_dt) const {
    for (auto& anchor : state.anchors) {
        anchor.velocity += force_.gravity * fixed_dt;
    }
}

void ExternalForces::apply_wind(BodyState& state, float fixed_dt) const {
    for (auto& anchor : state.anchors) {
        anchor.velocity += force_.wind * fixed_dt;
    }
}

void ExternalForces::apply_drag(BodyState& state) const {
    if (force_.drag <= 0.0f) return;
    for (auto& anchor : state.anchors) {
        anchor.velocity *= (1.0f - force_.drag);
    }
}

void ExternalForces::apply_point_forces(BodyState& state, float fixed_dt) const {
    for (const auto& pf : point_forces_) {
        if (pf.anchor < state.anchors.size()) {
            state.anchors[pf.anchor].velocity += pf.force * fixed_dt;
        }
    }
}

} // namespace delirium::m8