#pragma once
#include "delirium/render/surfel/surfel_cache.hpp"
#include "delirium/math/mat4.hpp"
#include <SDL2/SDL.h>

namespace delirium {

class Renderer {
public:
    /// Initialize OpenGL + SDL window
    bool init(int width, int height, const char* title);

    /// Render frame: surfel splat → screen
    void render(const SurfelCache& cache, const mat4& view, const mat4& proj);

    /// Process events, return false to quit
    bool poll_events();

    /// Shutdown
    void shutdown();

    int width() const { return w_; }
    int height() const { return h_; }

private:
    SDL_Window* window_ = nullptr;
    SDL_GLContext gl_ctx_ = nullptr;
    int w_, h_;
};

} // namespace delirium