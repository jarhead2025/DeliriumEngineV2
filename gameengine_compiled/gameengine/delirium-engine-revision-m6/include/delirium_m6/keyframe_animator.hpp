#pragma once
#include "delirium_m6/types.hpp"
#include <vector>
namespace delirium::m6 {
class KeyframeAnimator {
public:
    KeyframeAnimator();
    void set_keyframes(std::uint32_t mode_index, std::vector<Keyframe> frames);
    float get_amplitude(std::uint32_t mode_index, float time) const;
    bool has_keyframes(std::uint32_t mode_index) const;
    void clear();
private:
    std::vector<std::vector<Keyframe>> keyframes_;
};
} // namespace delirium::m6