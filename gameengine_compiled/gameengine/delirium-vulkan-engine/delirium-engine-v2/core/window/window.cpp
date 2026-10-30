#include "window.hpp"

namespace delirium {

bool Window::initialize(const char* title, uint32_t width, uint32_t height) {
    width_ = width; height_ = height;
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) return false;
    uint32_t flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE;
    window_ = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                width, height, flags);
    if (!window_) return false;
    return true;
}

void Window::shutdown() {
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

void Window::update_title(const char* title) {
    if (window_) SDL_SetWindowTitle(window_, title);
}

void Window::poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) should_close_ = true;
        if (event.type == SDL_WINDOWEVENT) {
            if (event.window.event == SDL_WINDOWEVENT_CLOSE) should_close_ = true;
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                width_ = static_cast<uint32_t>(event.window.data1);
                height_ = static_cast<uint32_t>(event.window.data2);
            }
        }
    }
}

} // namespace delirium