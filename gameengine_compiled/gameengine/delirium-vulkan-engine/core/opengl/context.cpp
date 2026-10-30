#include "context.hpp"
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <iostream>

namespace delirium {

void OpenGLContext::setup_gl_attributes() {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
}

bool OpenGLContext::initialize(SDL_Window* window) {
    window_ = window;

    gl_context_ = SDL_GL_CreateContext(window);
    if (!gl_context_) {
        std::cerr << "[GL] Failed to create context: " << SDL_GetError() << "\n";
        return false;
    }

    if (SDL_GL_SetSwapInterval(1) < 0) {
        std::cerr << "[GL] Warning: vsync not available: " << SDL_GetError() << "\n";
    }

    const char* renderer = (const char*)glGetString(GL_RENDERER);
    const char* version = (const char*)glGetString(GL_VERSION);
    std::cout << "[GL] Renderer: " << renderer << "\n";
    std::cout << "[GL] Version: " << version << "\n";

    return true;
}

void OpenGLContext::shutdown() {
    if (gl_context_) SDL_GL_DeleteContext(gl_context_);
    gl_context_ = nullptr;
}

void OpenGLContext::make_current() {
    if (window_ && gl_context_) SDL_GL_MakeCurrent(window_, gl_context_);
}

void OpenGLContext::swap() {
    SDL_GL_SwapWindow(window_);
}

} // namespace delirium