#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_state.hpp"
#include <string>
namespace delirium::m8 {
class ConstraintSolver {
public:
    explicit ConstraintSolver(std::uint32_t iterations = 4);
    void add_constraint(Constraint c);
    void clear_constraints();
    bool solve(BodyState& state, float fixed_dt, std::string* error = nullptr);
    std::size_t constraint_count() const { return constraints_.size(); }
    float solve_distance(BodyState& state, const Constraint& c, float fixed_dt) const;
    float solve_hinge(BodyState& state, const Constraint& c, float fixed_dt) const;
    float solve_slider(BodyState& state, const Constraint& c, float fixed_dt) const;
    void solve_fixed(BodyState& state, const Constraint& c, float fixed_dt) const;
private:
    std::uint32_t iterations_;
    std::vector<Constraint> constraints_;
};
} // namespace delirium::m8