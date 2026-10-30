#include "delirium_m5/modal_system.hpp"
#include <cmath>

namespace delirium::m5 {

ModalSystem::ModalSystem() = default;

void ModalSystem::add_mode(const ModalMode& mode) { modes_.push_back(mode); }
void ModalSystem::clear() { modes_.clear(); }

void ModalSystem::compute(BodyState& state, float time) const {
    for (AnchorId i = 0; i < state.anchors.size(); ++i)
        state.anchors[i].modal_displacement = compute_anchor_displacement(state.anchors[i].rest_position, time, i);
}

Vec3 ModalSystem::compute_anchor_displacement(const Vec3& rest_position, float time, AnchorId) const {
    Vec3 total{};
    for (const auto& mode : modes_) total += compute_mode_displacement(mode, rest_position, time);
    return total;
}

const ModalMode& ModalSystem::mode(std::size_t index) const {
    if (index >= modes_.size()) { static const ModalMode empty; return empty; }
    return modes_[index];
}

Vec3 ModalSystem::compute_mode_displacement(const ModalMode& mode, const Vec3& rest_position, float time) const {
    float scale = mode.amplitude * std::sin(kTwoPi * mode.frequency * time + mode.phase);
    switch (mode.type) {
        case ModalMode::Type::BendingX: {
            float d = std::sqrt(rest_position.y * rest_position.y + rest_position.z * rest_position.z);
            return {0.0f, d * scale * 0.1f, 0.0f};
        }
        case ModalMode::Type::BendingY: {
            float d = std::sqrt(rest_position.x * rest_position.x + rest_position.z * rest_position.z);
            return {d * scale * 0.1f, 0.0f, 0.0f};
        }
        case ModalMode::Type::BendingZ: {
            float d = std::sqrt(rest_position.x * rest_position.x + rest_position.y * rest_position.y);
            return {d * scale * 0.1f, 0.0f, 0.0f};
        }
        case ModalMode::Type::TorsionX: {
            float a = scale * 0.1f, c = std::cos(a), s = std::sin(a);
            return {0.0f, rest_position.y * c - rest_position.z * s - rest_position.y,
                     rest_position.y * s + rest_position.z * c - rest_position.z};
        }
        case ModalMode::Type::TorsionY: {
            float a = scale * 0.1f, c = std::cos(a), s = std::sin(a);
            return {rest_position.x * c + rest_position.z * s - rest_position.x, 0.0f,
                    -rest_position.x * s + rest_position.z * c - rest_position.z};
        }
        case ModalMode::Type::TorsionZ: {
            float a = scale * 0.1f, c = std::cos(a), s = std::sin(a);
            return {rest_position.x * c - rest_position.y * s - rest_position.x,
                    rest_position.x * s + rest_position.y * c - rest_position.y, 0.0f};
        }
        case ModalMode::Type::Breathing:
            return rest_position * (scale * 0.05f);
    }
    return Vec3{};
}

} // namespace delirium::m5