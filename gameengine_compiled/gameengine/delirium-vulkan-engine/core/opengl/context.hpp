#pragma once
#include <SDL.h>

namespace delirium {

class OpenGLContext {
public:
    // Must be called BEFORE SDL_CreateWindow — sets GL version/profile attributes
    static void setup_gl_attributes();

    bool initialize(SDL_Window* window);
    void shutdown();
    void make_current();
    void swap();

    SDL_Window* window() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    SDL_GLContext gl_context_ = nullptr;
};

} // namespace delirium