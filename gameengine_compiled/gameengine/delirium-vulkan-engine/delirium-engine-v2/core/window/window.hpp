#pragma once
#include <SDL.h>
#include <cstdint>

namespace delirium {

class Window {
public:
    bool initialize(const char* title, uint32_t width, uint32_t height);
    void shutdown();
    SDL_Window* handle() const { return window_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    bool should_close() const { return should_close_; }
    void update_title(const char* title);
    void poll_events();
private:
    SDL_Window* window_ = nullptr;
    uint32_t width_ = 1280, height_ = 720;
    bool should_close_ = false;
};

} // namespace delirium