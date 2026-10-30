#include "delirium_m4/modal_system.hpp"

#include <cmath>

namespace delirium::m4 {

ModalSystem::ModalSystem() = default;

void ModalSystem::add_mode(const ModalMode& mode) {
    modes_.push_back(mode);
}

void ModalSystem::clear() {
    modes_.clear();
}

void ModalSystem::compute(BodyState& state, float time) const {
    for (AnchorId i = 0; i < state.anchors.size(); ++i) {
        state.anchors[i].modal_displacement = compute_anchor_displacement(
            state.anchors[i].rest_position, time, i);
    }
}

Vec3 ModalSystem::compute_anchor_displacement(const Vec3& rest_position,
                                               float time,
                                               AnchorId) const {
    Vec3 total{};
    for (const auto& mode : modes_) {
        total += compute_mode_displacement(mode, rest_position, time);
    }
    return total;
}

const ModalMode& ModalSystem::mode(std::size_t index) const {
    if (index >= modes_.size()) {
        static const ModalMode empty;
        return empty;
    }
    return modes_[index];
}

Vec3 ModalSystem::compute_mode_displacement(const ModalMode& mode,
                                             const Vec3& rest_position,
                                             float time) const {
    float scale = mode.amplitude * std::sin(kTwoPi * mode.frequency * time + mode.phase);

    switch (mode.type) {
        case ModalMode::Type::BendingX: {
            // Bending along X: displacement proportional to distance from X axis
            float dist = std::sqrt(rest_position.y * rest_position.y +
                                   rest_position.z * rest_position.z);
            return Vec3{0.0f, dist * scale * 0.1f, 0.0f};
        }
        case ModalMode::Type::BendingY: {
            float dist = std::sqrt(rest_position.x * rest_position.x +
                                   rest_position.z * rest_position.z);
            return Vec3{dist * scale * 0.1f, 0.0f, 0.0f};
        }
        case ModalMode::Type::BendingZ: {
            float dist = std::sqrt(rest_position.x * rest_position.x +
                                   rest_position.y * rest_position.y);
            return Vec3{dist * scale * 0.1f, 0.0f, 0.0f};
        }
        case ModalMode::Type::TorsionX: {
            // Torsion around X axis
            float angle = scale * 0.1f;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float new_y = rest_position.y * cos_a - rest_position.z * sin_a;
            float new_z = rest_position.y * sin_a + rest_position.z * cos_a;
            return Vec3{0.0f, new_y - rest_position.y, new_z - rest_position.z};
        }
        case ModalMode::Type::TorsionY: {
            float angle = scale * 0.1f;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float new_x = rest_position.x * cos_a + rest_position.z * sin_a;
            float new_z = -rest_position.x * sin_a + rest_position.z * cos_a;
            return Vec3{new_x - rest_position.x, 0.0f, new_z - rest_position.z};
        }
        case ModalMode::Type::TorsionZ: {
            float angle = scale * 0.1f;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float new_x = rest_position.x * cos_a - rest_position.y * sin_a;
            float new_y = rest_position.x * sin_a + rest_position.y * cos_a;
            return Vec3{new_x - rest_position.x, new_y - rest_position.y, 0.0f};
        }
        case ModalMode::Type::Breathing: {
            // Uniform expansion/contraction from center
            return rest_position * (scale * 0.05f);
        }
    }

    return Vec3{};
}

} // namespace delirium::m4