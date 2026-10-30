#pragma once
#include <SDL.h>
#include <cstdint>

namespace delirium {

class Input {
public:
    void poll();
    float mouse_dx() const { return mouse_dx_; }
    float mouse_dy() const { return mouse_dy_; }
    bool mouse_button(int button) const { return mouse_buttons_[button]; }
    bool key_down(int key) const { return key_down_[key]; }
    bool key_held(int key) const { return key_states_[key]; }
    int mouse_wheel() const { return mouse_wheel_; }
    void capture_mouse(SDL_Window* window, bool capture);
private:
    float mouse_dx_ = 0.0f, mouse_dy_ = 0.0f;
    int mouse_wheel_ = 0;
    bool mouse_buttons_[4] = {};
    bool key_states_[256] = {};
    bool key_down_[256] = {};
};

} // namespace delirium