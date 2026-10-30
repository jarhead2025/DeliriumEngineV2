#pragma once
#include <SDL.h>
#include <cstdint>

namespace delirium {

class Input {
public:
    void poll();

    // Mouse
    float mouse_x() const { return mouse_x_; }
    float mouse_y() const { return mouse_y_; }
    float mouse_dx() const { return mouse_dx_; }
    float mouse_dy() const { return mouse_dy_; }
    bool mouse_button(int button) const { return mouse_buttons_[button]; }
    bool mouse_down(int button) const { return mouse_down_[button]; } // just pressed this frame

    // Keyboard
    bool key_down(int key) const { return key_down_[key]; } // just pressed this frame
    bool key_held(int key) const { return key_states_[key]; }

    // Capture mouse for camera rotation
    void capture_mouse(SDL_Window* window, bool capture);

private:
    float mouse_x_ = 0.0f, mouse_y_ = 0.0f;
    float mouse_dx_ = 0.0f, mouse_dy_ = 0.0f;
    bool mouse_buttons_[4] = {};  // SDL_BUTTON_* are 1-indexed (1=left, 2=middle, 3=right)
    bool mouse_down_[4] = {};

    bool key_states_[256] = {};
    bool key_down_[256] = {};
};

} // namespace delirium