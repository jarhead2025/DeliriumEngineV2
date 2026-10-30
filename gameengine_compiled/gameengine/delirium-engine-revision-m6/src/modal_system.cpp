#include "delirium_m6/modal_system.hpp"
#include <cmath>
namespace delirium::m6 {
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
Vec3 ModalSystem::compute_mode_displacement(const ModalMode& mode, const Vec3& rp, float time) const {
    float s = mode.amplitude * std::sin(kTwoPi * mode.frequency * time + mode.phase);
    switch (mode.type) {
        case ModalMode::Type::BendingX: {
            float d = std::sqrt(rp.y*rp.y + rp.z*rp.z);
            return {0.0f, d*s*0.1f, 0.0f};
        }
        case ModalMode::Type::BendingY: {
            float d = std::sqrt(rp.x*rp.x + rp.z*rp.z);
            return {d*s*0.1f, 0.0f, 0.0f};
        }
        case ModalMode::Type::BendingZ: {
            float d = std::sqrt(rp.x*rp.x + rp.y*rp.y);
            return {d*s*0.1f, 0.0f, 0.0f};
        }
        case ModalMode::Type::TorsionX: {
            float a = s*0.1f, c = std::cos(a), si = std::sin(a);
            return {0.0f, rp.y*c - rp.z*si - rp.y, rp.y*si + rp.z*c - rp.z};
        }
        case ModalMode::Type::TorsionY: {
            float a = s*0.1f, c = std::cos(a), si = std::sin(a);
            return {rp.x*c + rp.z*si - rp.x, 0.0f, -rp.x*si + rp.z*c - rp.z};
        }
        case ModalMode::Type::TorsionZ: {
            float a = s*0.1f, c = std::cos(a), si = std::sin(a);
            return {rp.x*c - rp.y*si - rp.x, rp.x*si + rp.y*c - rp.y, 0.0f};
        }
        case ModalMode::Type::Breathing:
            return rp * (s * 0.05f);
    }
    return Vec3{};
}
} // namespace delirium::m6