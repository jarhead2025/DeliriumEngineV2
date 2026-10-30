#pragma once
#include "delirium_m8/types.hpp"
#include <string>
#include <vector>
namespace delirium::m8 {
class AnimationTimeline {
public:
    void add_clip(AnimationClip clip);
    const AnimationClip* find_clip(const std::string& name) const;
    void play(const std::string& name, float speed = 1.0f);
    void pause();
    void stop();
    void tick(float dt);
    bool is_playing() const { return playing_; }
    float get_amplitude(std::uint32_t mode_index) const;
    const std::string& current_clip_name() const { return current_clip_name_; }
    float elapsed() const { return elapsed_; }
private:
    std::vector<AnimationClip> clips_;
    const AnimationClip* current_clip_ = nullptr;
    std::string current_clip_name_;
    float elapsed_ = 0.0f;
    float speed_ = 1.0f;
    bool playing_ = false;
    bool paused_ = false;
};
} // namespace delirium::m8