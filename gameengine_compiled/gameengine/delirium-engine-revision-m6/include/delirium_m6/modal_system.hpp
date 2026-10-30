#pragma once
#include "delirium_m6/types.hpp"
#include "delirium_m6/body_state.hpp"
#include <vector>
namespace delirium::m6 {
class ModalSystem {
public:
    ModalSystem();
    void add_mode(const ModalMode& mode);
    void clear();
    void compute(BodyState& state, float time) const;
    Vec3 compute_anchor_displacement(const Vec3& rest_position, float time, AnchorId) const;
    std::size_t mode_count() const { return modes_.size(); }
    const ModalMode& mode(std::size_t index) const;
private:
    std::vector<ModalMode> modes_;
    Vec3 compute_mode_displacement(const ModalMode& mode, const Vec3& rest_position, float time) const;
};
} // namespace delirium::m6