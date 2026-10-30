#include "delirium_m5/keyframe_animator.hpp"
#include <algorithm>

namespace delirium::m5 {

KeyframeAnimator::KeyframeAnimator() {}

void KeyframeAnimator::set_keyframes(std::uint32_t mode_index, std::vector<Keyframe> frames) {
    if (mode_index >= keyframes_.size())
        keyframes_.resize(mode_index + 1);
    std::sort(frames.begin(), frames.end(), [](const Keyframe& a, const Keyframe& b) {
        return a.time < b.time;
    });
    keyframes_[mode_index] = std::move(frames);
}

float KeyframeAnimator::get_amplitude(std::uint32_t mode_index, float time) const {
    if (mode_index >= keyframes_.size() || keyframes_[mode_index].empty()) return 0.0f;
    const auto& kf = keyframes_[mode_index];

    if (time <= kf.front().time) return kf.front().amplitude;
    if (time >= kf.back().time) return kf.back().amplitude;

    for (std::size_t i = 0; i + 1 < kf.size(); ++i) {
        if (time >= kf[i].time && time <= kf[i+1].time) {
            float t = (time - kf[i].time) / (kf[i+1].time - kf[i].time);
            return kf[i].amplitude + (kf[i+1].amplitude - kf[i].amplitude) * t;
        }
    }
    return kf.back().amplitude;
}

bool KeyframeAnimator::has_keyframes(std::uint32_t mode_index) const {
    return mode_index < keyframes_.size() && !keyframes_[mode_index].empty();
}

void KeyframeAnimator::clear() { keyframes_.clear(); }

} // namespace delirium::m5