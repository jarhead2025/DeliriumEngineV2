#include "delirium_m7/animation_timeline.hpp"
#include <algorithm>

namespace delirium::m7 {

void AnimationTimeline::add_clip(AnimationClip clip) {
    clips_.push_back(std::move(clip));
}

const AnimationClip* AnimationTimeline::find_clip(const std::string& name) const {
    for (const auto& c : clips_) {
        if (c.name == name) return &c;
    }
    return nullptr;
}

void AnimationTimeline::play(const std::string& name, float speed) {
    current_clip_ = find_clip(name);
    if (!current_clip_) return;
    current_clip_name_ = name;
    elapsed_ = 0.0f;
    speed_ = speed;
    playing_ = true;
    paused_ = false;
}

void AnimationTimeline::pause() {
    paused_ = true;
}

void AnimationTimeline::stop() {
    playing_ = false;
    paused_ = false;
    elapsed_ = 0.0f;
    current_clip_ = nullptr;
}

void AnimationTimeline::tick(float dt) {
    if (!playing_ || paused_ || !current_clip_) return;
    elapsed_ += dt * speed_;
    if (elapsed_ >= current_clip_->duration) {
        elapsed_ = current_clip_->duration;
        playing_ = false;
    }
}

float AnimationTimeline::get_amplitude(std::uint32_t mode_index) const {
    if (!current_clip_ || current_clip_->mode_index != mode_index) return 0.0f;
    const auto& kfs = current_clip_->keyframes;
    if (kfs.empty()) return 0.0f;

    float t = elapsed_;
    if (t <= kfs.front().time) return kfs.front().amplitude;
    if (t >= kfs.back().time) return kfs.back().amplitude;

    for (std::size_t i = 0; i + 1 < kfs.size(); ++i) {
        float t0 = kfs[i].time;
        float t1 = kfs[i + 1].time;
        if (t >= t0 && t <= t1) {
            float frac = (t - t0) / (t1 - t0);
            return kfs[i].amplitude + frac * (kfs[i + 1].amplitude - kfs[i].amplitude);
        }
    }
    return kfs.back().amplitude;
}

} // namespace delirium::m7