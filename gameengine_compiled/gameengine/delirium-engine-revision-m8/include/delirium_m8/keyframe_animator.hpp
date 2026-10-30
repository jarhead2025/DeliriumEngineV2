#pragma once
#include "delirium_m8/types.hpp"
#include <vector>
namespace delirium::m8 {
class KeyframeAnimator {
public:
    void add_keyframe(std::uint32_t mode_index, float time, float amplitude);
    float get_amplitude(std::uint32_t mode_index, float time) const;
    const std::vector<Keyframe>& keyframes(std::uint32_t mode_index) const;
    bool has_keyframes(std::uint32_t mode_index) const;
private:
    std::vector<std::pair<std::uint32_t, Keyframe>> keyframes_;
};
} // namespace delirium::m8