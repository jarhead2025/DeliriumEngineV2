#pragma once

#include "delirium_m5/types.hpp"
#include <vector>

namespace delirium::m5 {

// KeyframeAnimator — interpolates modal mode amplitudes over time
class KeyframeAnimator {
public:
    KeyframeAnimator();

    // Set keyframes for a modal mode index
    void set_keyframes(std::uint32_t mode_index, std::vector<Keyframe> frames);

    // Get interpolated amplitude for a mode at the given time
    float get_amplitude(std::uint32_t mode_index, float time) const;

    // Check if a mode has keyframes
    bool has_keyframes(std::uint32_t mode_index) const;

    // Clear all keyframes
    void clear();

private:
    std::vector<std::vector<Keyframe>> keyframes_;
};

} // namespace delirium::m5