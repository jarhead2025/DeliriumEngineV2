#include "delirium_m7/modal_system.hpp"
#include <cmath>

namespace delirium::m7 {

void ModalSystem::add_mode(ModalMode mode) {
    modes_.push_back(std::move(mode));
}

float ModalSystem::evaluate_displacement(std::size_t mode_index, float time) const {
    if (mode_index >= modes_.size()) return 0.0f;
    const auto& m = modes_[mode_index];
    return m.amplitude * std::sin(kTwoPi * m.frequency * time + m.phase);
}

} // namespace delirium::m7