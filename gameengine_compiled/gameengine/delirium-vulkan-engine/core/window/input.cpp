#include "input.hpp"
#include <SDL.h>

namespace delirium {

void Input::poll() {
    // Reset per-frame "just pressed" state
    for (int i = 0; i < 4; ++i) mouse_down_[i] = false;
    for (int i = 0; i < 256; ++i) key_down_[i] = false;

    // Poll mouse buttons (SDL_BUTTON_* are 1-indexed)
    int mx = 0, my = 0;
    Uint32 mouse_state = SDL_GetMouseState(&mx, &my);
    mouse_x_ = static_cast<float>(mx);
    mouse_y_ = static_cast<float>(my);
    bool new_left  = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT))  != 0;
    bool new_middle = (mouse_state & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0;
    bool new_right = (mouse_state & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;

    mouse_down_[SDL_BUTTON_LEFT]  = new_left  && !mouse_buttons_[SDL_BUTTON_LEFT];
    mouse_down_[SDL_BUTTON_MIDDLE] = new_middle && !mouse_buttons_[SDL_BUTTON_MIDDLE];
    mouse_down_[SDL_BUTTON_RIGHT] = new_right && !mouse_buttons_[SDL_BUTTON_RIGHT];

    mouse_buttons_[SDL_BUTTON_LEFT]  = new_left;
    mouse_buttons_[SDL_BUTTON_MIDDLE] = new_middle;
    mouse_buttons_[SDL_BUTTON_RIGHT] = new_right;

    // Compute mouse delta (from SDL_GetRelativeMouseState)
    int dx = 0, dy = 0;
    SDL_GetRelativeMouseState(&dx, &dy);
    mouse_dx_ = static_cast<float>(dx);
    mouse_dy_ = static_cast<float>(dy);

    // Poll keyboard state
    const Uint8* keys = SDL_GetKeyboardState(nullptr);
    for (int i = 0; i < 256; ++i) {
        bool pressed = keys[i] != 0;
        key_down_[i] = pressed && !key_states_[i];
        key_states_[i] = pressed;
    }
}

void Input::capture_mouse(SDL_Window* window, bool capture) {
    if (window) {
        SDL_SetRelativeMouseMode(capture ? SDL_TRUE : SDL_FALSE);
        if (capture) {
            int w, h;
            SDL_GetWindowSize(window, &w, &h);
            SDL_WarpMouseInWindow(window, w / 2, h / 2);
        }
    }
}

} // namespace delirium