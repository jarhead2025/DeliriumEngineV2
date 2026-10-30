#include "input.hpp"

namespace delirium {

void Input::poll() {
    for (int i = 0; i < 4; ++i) {
        bool new_left  = (SDL_GetMouseState(nullptr, nullptr) & SDL_BUTTON(SDL_BUTTON_LEFT))  != 0;
        bool new_right = (SDL_GetMouseState(nullptr, nullptr) & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
        mouse_buttons_[SDL_BUTTON_LEFT]  = new_left;
        mouse_buttons_[SDL_BUTTON_RIGHT] = new_right;
    }
    int dx = 0, dy = 0;
    SDL_GetRelativeMouseState(&dx, &dy);
    mouse_dx_ = static_cast<float>(dx);
    mouse_dy_ = static_cast<float>(dy);
    mouse_wheel_ = 0;
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