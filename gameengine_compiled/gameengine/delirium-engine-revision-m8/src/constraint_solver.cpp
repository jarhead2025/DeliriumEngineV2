#include "delirium_m8/constraint_solver.hpp"
#include <cmath>

namespace delirium::m8 {

ConstraintSolver::ConstraintSolver(std::uint32_t iterations) : iterations_(iterations) {}

void ConstraintSolver::add_constraint(Constraint c) {
    constraints_.push_back(std::move(c));
}

void ConstraintSolver::clear_constraints() {
    constraints_.clear();
}

float ConstraintSolver::solve_distance(BodyState& state, const Constraint& c, float fixed_dt) const {
    auto& a = state.anchors[c.anchor_a];
    auto& b = state.anchors[c.anchor_b];

    Vec3 diff = a.position - b.position;
    float dist = length(diff);
    if (dist < 1e-8f) return 0.0f;

    Vec3 n = diff * (1.0f / dist);
    float error = dist - c.rest_distance;

    Vec3 rel_vel = a.velocity - b.velocity;
    float rel_vel_n = dot(rel_vel, n);

    float mass_a = 1.0f, mass_b = 1.0f;
    float inv_mass_sum = 1.0f / (mass_a + mass_b);

    float j = -(1.0f + c.damping) * rel_vel_n * inv_mass_sum;
    j += c.stiffness * error * inv_mass_sum;

    a.velocity += n * j * mass_a * fixed_dt;
    b.velocity -= n * j * mass_b * fixed_dt;

    float correction = error * 0.5f * fixed_dt;
    a.position -= n * correction * mass_a * inv_mass_sum;
    b.position += n * correction * mass_b * inv_mass_sum;

    return error;
}

float ConstraintSolver::solve_hinge(BodyState& state, const Constraint& c, float fixed_dt) const {
    auto& a = state.anchors[c.anchor_a];
    auto& b = state.anchors[c.anchor_b];

    Vec3 diff = a.position - b.position;
    float dist = length(diff);
    if (dist < 1e-8f) return 0.0f;

    Vec3 n = diff * (1.0f / dist);
    float error = dist - c.rest_distance;

    Vec3 rel_vel = a.velocity - b.velocity;
    float rel_vel_n = dot(rel_vel, n);

    float mass_a = 1.0f, mass_b = 1.0f;
    float inv_mass_sum = 1.0f / (mass_a + mass_b);

    float j = -(1.0f + c.damping) * rel_vel_n * inv_mass_sum;
    j += c.stiffness * error * inv_mass_sum;

    a.velocity += n * j * mass_a * fixed_dt;
    b.velocity -= n * j * mass_b * fixed_dt;

    float correction = error * 0.5f * fixed_dt;
    a.position -= n * correction * mass_a * inv_mass_sum;
    b.position += n * correction * mass_b * inv_mass_sum;

    Vec3 tangent = rel_vel - n * rel_vel_n;
    float tangent_len = length(tangent);
    if (tangent_len > 1e-8f) {
        tangent *= 1.0f / tangent_len;
        float damp_j = c.damping * dot(rel_vel, tangent) * inv_mass_sum;
        a.velocity -= tangent * damp_j * mass_a * fixed_dt;
        b.velocity += tangent * damp_j * mass_b * fixed_dt;
    }

    return error;
}

float ConstraintSolver::solve_slider(BodyState& state, const Constraint& c, float fixed_dt) const {
    auto& a = state.anchors[c.anchor_a];
    auto& b = state.anchors[c.anchor_b];

    Vec3 diff = a.position - b.position;
    float dist = length(diff);
    if (dist < 1e-8f) return 0.0f;

    Vec3 n = diff * (1.0f / dist);

    Vec3 rel_vel = a.velocity - b.velocity;
    float rel_vel_n = dot(rel_vel, n);

    if (dist <= c.rest_distance) return 0.0f;

    float error = dist - c.rest_distance;
    float mass_a = 1.0f, mass_b = 1.0f;
    float inv_mass_sum = 1.0f / (mass_a + mass_b);

    float j = -rel_vel_n * inv_mass_sum;
    j += c.stiffness * error * inv_mass_sum;

    if (j > 0.0f) j = 0.0f;

    a.velocity += n * j * mass_a * fixed_dt;
    b.velocity -= n * j * mass_b * fixed_dt;

    return error;
}

void ConstraintSolver::solve_fixed(BodyState& state, const Constraint& c, float fixed_dt) const {
    auto& a = state.anchors[c.anchor_a];
    auto& b = state.anchors[c.anchor_b];

    Vec3 diff = a.position - b.position;
    float dist = length(diff);
    if (dist < 1e-8f) return;

    Vec3 n = diff * (1.0f / dist);
    float error = dist - c.rest_distance;

    Vec3 rel_vel = a.velocity - b.velocity;
    float rel_vel_n = dot(rel_vel, n);

    float mass_a = 1.0f, mass_b = 1.0f;
    float inv_mass_sum = 1.0f / (mass_a + mass_b);

    float j = -(1.0f + c.damping) * rel_vel_n * inv_mass_sum;
    j += c.stiffness * error * inv_mass_sum;

    a.velocity += n * j * mass_a * fixed_dt;
    b.velocity -= n * j * mass_b * fixed_dt;

    Vec3 avg_vel = (a.velocity + b.velocity) * 0.5f;
    a.velocity = avg_vel;
    b.velocity = avg_vel;
}

bool ConstraintSolver::solve(BodyState& state, float fixed_dt, std::string* error) {
    if (constraints_.empty()) return true;

    for (std::uint32_t iter = 0; iter < iterations_; ++iter) {
        for (const auto& c : constraints_) {
            if (c.anchor_a >= state.anchors.size() || c.anchor_b >= state.anchors.size()) {
                if (error) *error = "invalid constraint anchor";
                return false;
            }
            switch (c.type) {
                case ConstraintType::Distance: solve_distance(state, c, fixed_dt); break;
                case ConstraintType::Hinge:    solve_hinge(state, c, fixed_dt); break;
                case ConstraintType::Slider:   solve_slider(state, c, fixed_dt); break;
                case ConstraintType::Fixed:    solve_fixed(state, c, fixed_dt); break;
            }
        }
    }
    return true;
}

} // namespace delirium::m8