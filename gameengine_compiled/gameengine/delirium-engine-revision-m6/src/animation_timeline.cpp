#include "delirium_m6/animation_timeline.hpp"
#include <algorithm>
namespace delirium::m6 {
AnimationTimeline::AnimationTimeline() {}
void AnimationTimeline::add_clip(AnimationClip clip) { clips_.push_back(std::move(clip)); }
const AnimationClip* AnimationTimeline::find_clip(const std::string& name) const {
    for (const auto& c : clips_)
        if (c.name == name) return &c;
    return nullptr;
}
void AnimationTimeline::play(const std::string& name, float speed) {
    if (find_clip(name)) {
        current_clip_name_ = name;
        current_time_ = 0.0f;
        speed_ = speed;
        playing_ = true;
    }
}
void AnimationTimeline::pause() { playing_ = false; }
void AnimationTimeline::stop() { playing_ = false; current_time_ = 0.0f; current_clip_name_.clear(); }
void AnimationTimeline::tick(float dt) {
    if (!playing_) return;
    current_time_ += dt * speed_;
    const AnimationClip* clip = find_clip(current_clip_name_);
    if (clip && current_time_ >= clip->duration) {
        current_time_ = clip->duration;
        playing_ = false;
    }
}
float AnimationTimeline::get_amplitude(std::uint32_t mode_index) const {
    const AnimationClip* clip = find_clip(current_clip_name_);
    if (!clip || clip->mode_index != mode_index) return 0.0f;
    const auto& kf = clip->keyframes;
    if (kf.empty()) return 0.0f;
    if (current_time_ <= kf.front().time) return kf.front().amplitude;
    if (current_time_ >= kf.back().time) return kf.back().amplitude;
    for (std::size_t i = 0; i + 1 < kf.size(); ++i) {
        if (current_time_ >= kf[i].time && current_time_ <= kf[i+1].time) {
            float t = (current_time_ - kf[i].time) / (kf[i+1].time - kf[i].time);
            return kf[i].amplitude + (kf[i+1].amplitude - kf[i].amplitude) * t;
        }
    }
    return kf.back().amplitude;
}
void AnimationTimeline::clear() { clips_.clear(); stop(); }
} // namespace delirium::m6