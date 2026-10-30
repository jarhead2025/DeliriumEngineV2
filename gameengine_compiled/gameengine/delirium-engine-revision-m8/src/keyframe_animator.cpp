#include "delirium_m8/keyframe_animator.hpp"
#include <algorithm>

namespace delirium::m8 {

void KeyframeAnimator::add_keyframe(std::uint32_t mode_index, float time, float amplitude) {
    Keyframe kf;
    kf.time = time;
    kf.amplitude = amplitude;
    keyframes_.push_back({mode_index, kf});
}

float KeyframeAnimator::get_amplitude(std::uint32_t mode_index, float time) const {
    std::vector<const Keyframe*> list;
    for (const auto& [mi, kf] : keyframes_) {
        if (mi == mode_index) list.push_back(&kf);
    }
    if (list.empty()) return 0.0f;

    if (time <= list.front()->time) return list.front()->amplitude;
    if (time >= list.back()->time) return list.back()->amplitude;

    for (std::size_t i = 0; i + 1 < list.size(); ++i) {
        if (time >= list[i]->time && time <= list[i + 1]->time) {
            float t = (time - list[i]->time) / (list[i + 1]->time - list[i]->time);
            return list[i]->amplitude + t * (list[i + 1]->amplitude - list[i]->amplitude);
        }
    }
    return list.back()->amplitude;
}

const std::vector<Keyframe>& KeyframeAnimator::keyframes(std::uint32_t mode_index) const {
    static const std::vector<Keyframe> empty;
    (void)mode_index;
    return empty;
}

bool KeyframeAnimator::has_keyframes(std::uint32_t mode_index) const {
    for (const auto& [mi, kf] : keyframes_) {
        if (mi == mode_index) return true;
    }
    return false;
}

} // namespace delirium::m8