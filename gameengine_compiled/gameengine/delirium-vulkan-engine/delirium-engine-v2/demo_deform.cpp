#include "core/body_runtime.h"
#include "core/body_topology.h"
#include "core/body_state.h"
#include "core/modal_system.h"
#include "core/material_registry.h"
#include "core/opengl/context.hpp"
#include "core/opengl/shader.hpp"
#include "core/window/window.hpp"
#include "core/window/input.hpp"
#include "core/render/compute_renderer.hpp"

#include <iostream>
#include <chrono>
#include <cmath>
#include <memory>
#include <SDL2/SDL.h>
#include <GL/glew.h>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace delirium {

struct SphereBuilder {
    std::vector<Vec3> verts;
    std::vector<std::array<AnchorId, 4>> quads;

    void build(int rings = 8, int segments = 16, float radius = 1.0f) {
        verts.clear(); quads.clear();
        for (int r = 0; r <= rings; ++r) {
            float phi = (r * kPi) / rings;
            for (int s = 0; s <= segments; ++s) {
                float theta = (s * kTwoPi) / segments;
                verts.push_back({
                    radius * sinf(phi) * cosf(theta),
                    radius * cosf(phi),
                    radius * sinf(phi) * sinf(theta)
                });
            }
        }
        for (int r = 0; r < rings; ++r)
            for (int s = 0; s < segments; ++s) {
                int a = r*(segments+1)+s;
                int b = r*(segments+1)+s+1;
                int c = (r+1)*(segments+1)+s;
                int d = (r+1)*(segments+1)+s+1;
                quads.push_back({(AnchorId)a,(AnchorId)b,(AnchorId)c,(AnchorId)d});
            }
    }
};

struct Camera {
    float theta = 0.5f, phi = 0.8f, radius = 3.5f;
    Vec3 target{};

    Vec3 position() const {
        return { target.x + radius*sinf(phi)*cosf(theta),
                 target.y + radius*cosf(phi),
                 target.z + radius*sinf(phi)*sinf(theta) };
    }
    void orbit(float dx, float dy) {
        theta -= dx * 0.005f;
        phi = std::max(0.1f, std::min(3.13f, phi - dy * 0.005f));
    }
    void zoom(float y) { radius = std::max(1.5f, radius - y * 0.3f); }
};

} // namespace delirium

int main() {
    using namespace delirium;

    // ── Build sphere body ──────────────────────────────────────────
    SphereBuilder builder;
    builder.build(6, 12, 1.0f);  // 77 anchors, 72 patches

    BodyTopology topology;
    for (size_t i = 0; i < builder.verts.size(); ++i)
        topology.add_anchor();

    for (const auto& quad : builder.quads) {
        PatchCoefficients coef;
        for (int i = 0; i < 8; ++i) coef.scalars[i] = 0.3f;
        topology.add_patch(quad, 0, BasisType::Cupula, coef);
    }
    std::string error;
    topology.finalize(&error);
    if (!error.empty()) { std::cerr << "[ERROR] " << error << "\n"; return 1; }

    auto body = std::make_unique<BodyRuntime>(std::move(topology));
    body->initialize(builder.verts);

    MaterialRegistry mat_reg;
    MaterialProperties mat;
    mat.youngs_modulus = 2.0e6f; mat.poisson_ratio = 0.45f; mat.density = 800.0f;
    mat.r = 180; mat.g = 200; mat.b = 220; mat.roughness = 0.3f;
    mat_reg.register_material(0, mat);
    body->set_materials(std::move(mat_reg));

    ModalSystem modal;
    modal.add_mode({ModalMode::Type::Breathing, 0.15f, 2.0f, 0.0f});
    modal.add_mode({ModalMode::Type::BendingX, 0.10f, 3.5f, 0.0f});
    modal.add_mode({ModalMode::Type::BendingY, 0.08f, 4.0f, 0.5f});
    body->set_modal_system(std::move(modal));
    body->set_external_forces({{}, {}, 0.1f});  // No gravity by default

    std::cout << "[Deform] " << builder.verts.size() << " anchors, "
              << builder.quads.size() << " patches, 3 modal modes\n";

    // ── OpenGL init ────────────────────────────────────────────────
    Window window;
    OpenGLContext ctx;

    OpenGLContext::setup_gl_attributes();
    if (!window.initialize("Delirium v2 — GPU Compute", 1280, 720)) {
        std::cerr << "[ERROR] Window init failed\n"; return 1;
    }
    if (!ctx.initialize(window.handle())) {
        std::cerr << "[ERROR] GL init failed\n"; return 1;
    }
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "[ERROR] GLEW init failed\n"; return 1;
    }
    std::cout << "[GL] " << glGetString(GL_RENDERER) << "\n";
    std::cout << "[GL] " << glGetString(GL_VERSION) << "\n";

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // ── Compute renderer ───────────────────────────────────────────
    ComputeRenderer compute;
    std::string cs_path = std::string(SHADER_DIR) + "/psf_splat.comp.glsl";
    if (!compute.initialize(cs_path.c_str(), window.width(), window.height())) {
        std::cerr << "[ERROR] Compute init failed\n"; return 1;
    }

    Camera camera;
    bool gravity_on = false;
    bool running = true;

    std::cout << "\n[GPU Deform Test]\n";
    std::cout << "  G — toggle gravity\n";
    std::cout << "  Space — impulse kick\n";
    std::cout << "  Drag — orbit | Scroll — zoom\n";
    std::cout << "  ESC — quit\n\n";

    auto last = std::chrono::steady_clock::now();
    float time_ = 0.0f;

    while (running) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        if (dt > 0.05f) dt = 0.05f;

        // Events
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { running = false; break; }
            if (ev.type == SDL_KEYDOWN) {
                if (ev.key.keysym.sym == SDLK_ESCAPE) { running = false; break; }
                if (ev.key.keysym.sym == SDLK_g) {
                    gravity_on = !gravity_on;
                    auto& f = body->external_forces_mut();
                    f.gravity = gravity_on ? Vec3{0,-9.81f,0} : Vec3{};
                    std::cout << "[Gravity] " << (gravity_on?"ON":"OFF") << "\n";
                }
                if (ev.key.keysym.sym == SDLK_SPACE) {
                    auto& state = body->write_state();
                    for (auto& a : state.anchors) a.velocity.y += 2.0f;
                    std::cout << "[Kick] impulse applied\n";
                }
            }
            if (ev.type == SDL_MOUSEWHEEL) camera.zoom(ev.wheel.y);
            if (ev.type == SDL_MOUSEMOTION && (ev.motion.state & SDL_BUTTON_LMASK))
                camera.orbit(ev.motion.xrel, ev.motion.yrel);
        }

        // Physics step
        body->step(1.0f/60.0f);
        time_ += dt;

        // Camera orbits fixed origin — body deforms in place, doesn't drift
        camera.target = Vec3{};

        // ── Render ─────────────────────────────────────────────────
        ctx.make_current();
        glViewport(0, 0, window.width(), window.height());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Vec3 cam_pos = camera.position();

        // View matrix (lookAt) — COLUMN-MAJOR for OpenGL
        float view[16];
        {
            Vec3 forward = normalize_vec3(camera.target - cam_pos);
            Vec3 world_up{0, 1, 0};
            Vec3 right = normalize_vec3(cross(world_up, forward));
            Vec3 up = cross(forward, right);

            // Column-major: each column is a basis vector, translation in column 3
            view[0]=right.x;  view[4]=right.y;  view[8]=right.z;  view[12]=-dot(right,cam_pos);
            view[1]=up.x;     view[5]=up.y;     view[9]=up.z;     view[13]=-dot(up,cam_pos);
            view[2]=-forward.x;view[6]=-forward.y;view[10]=-forward.z;view[14]=dot(forward,cam_pos);
            view[3]=0;        view[7]=0;        view[11]=0;       view[15]=1;
        }

        // Projection matrix (perspective)
        float proj[16] = {};
        {
            float fov = 70.0f * kPi / 180.0f;
            float aspect = (float)window.width() / (float)window.height();
            float tan_half = std::tan(fov / 2.0f);
            float near = 0.1f, far = 100.0f;
            proj[0] = 1.0f / (aspect * tan_half);
            proj[5] = 1.0f / tan_half;
            proj[10] = -(far + near) / (far - near);
            proj[11] = -2.0f * far * near / (far - near);
            proj[14] = -1.0f;
            proj[15] = 0.0f;
        }

        // GPU compute render
        compute.upload(body.get());
        compute.render(cam_pos, view, proj, (uint32_t)window.width(), (uint32_t)window.height());

        ctx.swap();
    }

    compute.shutdown();
    std::cout << "[Deform] Clean exit\n";
    return 0;
}