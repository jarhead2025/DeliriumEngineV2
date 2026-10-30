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

int main(int argc, char** argv) {
    std::string prompt = "wolf";
    if (argc >= 2) prompt = argv[1];

    std::cout << "\n========================================\n";
    std::cout << "  DeliRium v2 — Visual AI Demo\n";
    std::cout << "========================================\n\n";

    /* ── 1. Generate body ─────────────────────────────────────────── */
    delirium::MaterialScienceRegistry mat_science;
    delirium::MaterialRegistry phys_reg;
    delirium::AIGenerator gen;
    auto body = gen.generate(prompt, mat_science, phys_reg);
    if (!body) { std::cerr << "[FAIL] Generation\n"; return 1; }

    std::cout << "[AI] " << prompt << " ("
              << (gen.classify(prompt) == delirium::Domain::Organic ? "Organic" : "Inorganic")
              << ")\n";
    std::cout << "    Patches: " << body->topology().patch_count()
              << "  Anchors: " << body->read_state().anchors.size() << "\n\n";

    /* ── 2. OpenGL init ───────────────────────────────────────────── */
    delirium::render::OpenGLContext::setup_gl_attributes();
    delirium::render::OpenGLContext ctx;
    if (!ctx.initialize("DeliRium v2 — " + prompt, 1280, 720)) {
        std::cerr << "[FAIL] OpenGL init\n"; return 1;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.08f, 0.10f, 0.15f, 1.0f);

    /* ── 3. Shaders ───────────────────────────────────────────────── */
    delirium::render::ShaderProgram shader;
#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif
    {
        std::string vpath = SHADER_DIR "/surfel.vert.glsl";
        std::string fpath = SHADER_DIR "/surfel.frag.glsl";
        if (!shader.compile(vpath.c_str(), fpath.c_str())) {
            std::cerr << "[FAIL] Shaders (" << vpath << ", " << fpath << ")\n"; return 1;
        }
    }

    /* ── 4. Renderer ──────────────────────────────────────────────── */
    delirium::render::SurfelRenderer sr;
    sr.initialize();

    /* ── 5. Camera ────────────────────────────────────────────────── */
    delirium::render::Camera cam;

    /* ── 6. Physics warmup ────────────────────────────────────────── */
    body->set_external_forces({{0, -9.81f, 0}, {}, 0.01f});
    for (int i = 0; i < 10; i++) body->step(1.0f / 60.0f);

    /* ── 7. Frame loop ────────────────────────────────────────────── */
    bool running = true;
    auto t0 = std::chrono::steady_clock::now();
    int fc = 0;

    std::cout << "[Render] Mouse: orbit | Scroll: zoom | ESC: quit\n\n";

    while (running) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - t0).count();
        t0 = now;
        dt = std::min(dt, 0.05f);

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) running = false;
            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT)
                SDL_SetRelativeMouseMode(SDL_TRUE);
            if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT)
                SDL_SetRelativeMouseMode(SDL_FALSE);
            if (ev.type == SDL_MOUSEMOTION && SDL_GetRelativeMouseMode())
                cam.process_mouse(ev.motion.xrel, ev.motion.yrel, 1.0f);
            cam.update(dt, ev);
        }

        body->step(dt);

        delirium::SurfelCache cache;
        body->rebuild_cache(cache);
        sr.update_surfels(cache.surfels());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        float asp = (float)ctx.width() / (float)ctx.height();
        float mvp[16]; cam.get_mvp(mvp, asp);
        shader.use();
        shader.set_mat4("u_mvp", mvp);
        auto cp = cam.position();
        shader.set_vec3("u_cam_pos", cp.x, cp.y, cp.z);
        sr.draw(shader);
        ctx.swap();

        fc++;
        if (fc % 30 == 0) {
            char t[256];
            snprintf(t, sizeof(t), "DeliRium v2 — %s | %zu surfels",
                     prompt.c_str(), cache.surfels().size());
            ctx.update_title(t);
        }
    }

    sr.shutdown();
    shader.shutdown();
    ctx.shutdown();
    std::cout << "\n[OK] Done\n";
    return 0;
}