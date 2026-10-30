#include "opengl_context.h"
#include <iostream>

namespace delirium::render {

void OpenGLContext::setup_gl_attributes() {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
}

OpenGLContext::OpenGLContext() = default;
OpenGLContext::~OpenGLContext() { shutdown(); }

bool OpenGLContext::initialize(const std::string& title, int w, int h) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
        std::cerr << "[GL] SDL_Init: " << SDL_GetError() << "\n";
        return false;
    }
    window_ = SDL_CreateWindow(title.c_str(),
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!window_) { std::cerr << "[GL] Window: " << SDL_GetError() << "\n"; return false; }
    gl_ctx_ = SDL_GL_CreateContext(window_);
    if (!gl_ctx_) { std::cerr << "[GL] Context: " << SDL_GetError() << "\n"; return false; }
    SDL_GL_SetSwapInterval(1);
    if (glewInit() != GLEW_OK) { std::cerr << "[GL] GLEW init failed\n"; return false; }
    width_ = w; height_ = h;
    std::cout << "[GL] " << glGetString(GL_VERSION) << " on " << glGetString(GL_RENDERER) << "\n";
    return true;
}

void OpenGLContext::shutdown() {
    if (gl_ctx_) { SDL_GL_DeleteContext(gl_ctx_); gl_ctx_ = nullptr; }
    if (window_) { SDL_DestroyWindow(window_); window_ = nullptr; }
    SDL_Quit();
}
void OpenGLContext::swap() { if (window_) SDL_GL_SwapWindow(window_); }
void OpenGLContext::make_current() { if (window_ && gl_ctx_) SDL_GL_MakeCurrent(window_, gl_ctx_); }
void OpenGLContext::update_title(const std::string& t) { if (window_) SDL_SetWindowTitle(window_, t.c_str()); }

} // namespace delirium::render