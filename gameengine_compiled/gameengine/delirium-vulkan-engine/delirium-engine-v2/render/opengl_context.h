#pragma once
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <string>

namespace delirium::render {

/* ── SDL2 window + GL context wrapper ────────────────────────────── */
class OpenGLContext {
public:
    OpenGLContext();
    ~OpenGLContext();

    bool initialize(const std::string& title, int w, int h);
    void shutdown();
    void swap();
    void make_current();
    void update_title(const std::string& title);

    SDL_Window* window() const { return window_; }
    int width() const { return width_; }
    int height() const { return height_; }

    /* Call before SDL_Init so SDL knows we want OpenGL */
    static void setup_gl_attributes();

private:
    SDL_Window* window_ = nullptr;
    SDL_GLContext gl_ctx_ = nullptr;
    int width_ = 0, height_ = 0;
};

} // namespace delirium::render