#include "delirium/render/renderer.hpp"
#include <iostream>

namespace delirium {

bool Renderer::init(int width, int height, const char* title) {
    w_ = width;
    h_ = height;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        std::cerr << "[Renderer] SDL init failed: " << SDL_GetError() << "\n";
        return false;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    window_ = SDL_CreateWindow(title,
                               SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                               width, height,
                               SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!window_) {
        std::cerr << "[Renderer] Window creation failed: " << SDL_GetError() << "\n";
        return false;
    }

    gl_ctx_ = SDL_GL_CreateContext(window_);
    if (!gl_ctx_) {
        std::cerr << "[Renderer] GL context failed: " << SDL_GetError() << "\n";
        return false;
    }

    std::cout << "[Renderer] Initialized " << width << "x" << height
              << " | " << glGetString(GL_RENDERER) << "\n";
    return true;
}

void Renderer::render(const SurfelCache& cache, const mat4& view, const mat4& proj) {
    // TODO: surfel splatting shader
    // For now: clear to dark blue
    glClearColor(0.02f, 0.02f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SDL_GL_SwapWindow(window_);
}

bool Renderer::poll_events() {
    SDL_Event evt;
    while (SDL_PollEvent(&evt)) {
        if (evt.type == SDL_QUIT) return false;
        if (evt.type == SDL_KEYDOWN && evt.key.keysym.sym == SDLK_ESCAPE) return false;
    }
    return true;
}

void Renderer::shutdown() {
    if (gl_ctx_) SDL_GL_DeleteContext(gl_ctx_);
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

} // namespace delirium