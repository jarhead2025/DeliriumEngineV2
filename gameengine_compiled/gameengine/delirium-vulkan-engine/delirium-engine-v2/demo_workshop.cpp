#include "ai/generator.h"
#include "core/body_runtime.h"
#include "core/surfel_cache.h"
#include "render/opengl_context.h"
#include "render/shader_program.h"
#include "render/surfel_renderer.h"
#include "render/camera.h"

#include <iostream>
#include <chrono>
#include <string>
#include <array>
#include <SDL2/SDL.h>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace delirium {

class Workshop {
public:
    bool initialize();
    void run();
    void shutdown();

private:
    render::OpenGLContext ctx_;
    render::ShaderProgram sf_shader_;
    render::SurfelRenderer sf_renderer_;
    render::Camera camera_;

    std::unique_ptr<BodyRuntime> body_;
    MaterialRegistry materials_;
    AIGenerator gen_;
    MaterialScienceRegistry mat_science_;
    delirium::MaterialRegistry phys_reg_;

    SurfelCache cache_;
    std::string current_prompt_ = "wolf";
    std::array<const char*, 6> prompts_ = {"wolf", "dragon", "robot arm", "tree", "crystal", "bridge"};
    int prompt_idx_ = 0;

    void regenerate();
};

bool Workshop::initialize() {
    render::OpenGLContext::setup_gl_attributes();
    if (!ctx_.initialize("DeliRium Workshop", 1280, 720)) return false;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.05f, 0.08f, 0.15f, 1.0f);

    std::string vert = std::string(SHADER_DIR) + "/surfel.vert.glsl";
    std::string frag = std::string(SHADER_DIR) + "/surfel.frag.glsl";
    if (!sf_shader_.compile(vert.c_str(), frag.c_str())) return false;
    if (!sf_renderer_.initialize()) return false;

    MaterialProperties m0;
    m0.r = 180; m0.g = 140; m0.b = 100; m0.roughness = 0.6f; m0.youngs_modulus = 1e6f;
    materials_.register_material(0, m0);

    regenerate();

    std::cout << "\n[Workshop] " << current_prompt_ << " loaded\n";
    std::cout << "  Patches: " << body_->topology().patch_count()
              << "  Anchors: " << body_->read_state().anchors.size() << "\n";
    std::cout << "\n[Controls] G=cycle body  ESC=quit\n";
    std::cout << "  Mouse=orbit  Scroll=zoom\n\n";
    return true;
}

void Workshop::regenerate() {
    current_prompt_ = prompts_[prompt_idx_];
    body_ = gen_.generate(current_prompt_, mat_science_, phys_reg_);
    if (!body_) {
        std::cerr << "[FAIL] generation: " << current_prompt_ << "\n";
        return;
    }
    body_->set_materials(materials_);
    auto domain = gen_.classify(current_prompt_);
    std::cout << "[AI] \"" << current_prompt_ << "\" ("
              << (domain == Domain::Organic ? "Organic" : "Inorganic") << ")\n";
}

void Workshop::run() {
    auto last = std::chrono::high_resolution_clock::now();
    bool running = true;
    while (running) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        dt = std::min(dt, 0.05f);

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { running = false; break; }
            if (ev.type == SDL_KEYDOWN) {
                if (ev.key.keysym.sym == SDLK_ESCAPE) { running = false; break; }
                if (ev.key.keysym.sym == SDLK_g) {
                    prompt_idx_ = (prompt_idx_ + 1) % prompts_.size();
                    regenerate();
                }
            }
            camera_.update(dt, ev);
            if (ev.type == SDL_MOUSEMOTION)
                camera_.process_mouse(ev.motion.xrel, ev.motion.yrel);
        }

        if (body_) body_->step(dt);

        ctx_.make_current();
        glViewport(0, 0, ctx_.width(), ctx_.height());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (body_) {
            cache_.surfels_mut().clear();
            body_->rebuild_cache_with_lod(cache_, length(camera_.position()), nullptr);
            sf_renderer_.update_surfels(cache_.surfels());

            if (cache_.count() > 0) {
                float mvp[16];
                camera_.get_mvp(mvp, (float)ctx_.width() / ctx_.height());

                sf_shader_.set_mat4("u_mvp", mvp);
                sf_shader_.set_vec3("u_cam_pos",
                    camera_.position().x, camera_.position().y, camera_.position().z);
                sf_renderer_.draw(sf_shader_);
            }
        }

        ctx_.swap();
        char title[256];
        snprintf(title, sizeof(title), "Workshop | %s | %d surfels | %d patches",
                 current_prompt_.c_str(), (int)cache_.count(),
                 body_ ? (uint32_t)body_->topology().patch_count() : 0);
        ctx_.update_title(title);
    }
}

void Workshop::shutdown() {
    sf_renderer_.shutdown();
    sf_shader_.shutdown();
    ctx_.shutdown();
}

} // namespace delirium

int main() {
    delirium::Workshop w;
    if (!w.initialize()) return 1;
    w.run();
    w.shutdown();
    return 0;
}