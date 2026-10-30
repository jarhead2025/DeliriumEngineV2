#pragma once
#include "delirium_m6/types.hpp"
#include <string>
#include <vector>
namespace delirium::m6 {
// AnimationTimeline — manages multiple animation clips with playback control
class AnimationTimeline {
public:
    AnimationTimeline();
    void add_clip(AnimationClip clip);
    void play(const std::string& clip_name, float speed = 1.0f);
    void pause();
    void stop();
    void tick(float dt);
    float get_amplitude(std::uint32_t mode_index) const;
    bool is_playing() const { return playing_; }
    const std::string& current_clip_name() const { return current_clip_name_; }
    float current_time() const { return current_time_; }
    void clear();
private:
    std::vector<AnimationClip> clips_;
    std::string current_clip_name_;
    float current_time_ = 0.0f;
    float speed_ = 1.0f;
    bool playing_ = false;
    const AnimationClip* find_clip(const std::string& name) const;
};
} // namespace delirium::m6