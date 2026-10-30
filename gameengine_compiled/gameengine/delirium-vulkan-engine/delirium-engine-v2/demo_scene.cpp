#include "demo_scene.hpp"
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <array>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace delirium {

// ── DemoScene implementation ─────────────────────────────────────

bool DemoScene::initialize() {
    OpenGLContext::setup_gl_attributes();
    if (!window_.initialize("DeliRium Engine v2 - GPU Compute", 1280, 720)) {
        std::cerr << "Failed to create window\n";
        return false;
    }
    if (!gl_context_.initialize(window_.handle())) {
        std::cerr << "Failed to create OpenGL context\n";
        return false;
    }
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "[ERROR] GLEW init failed\n";
        return false;
    }
    std::cout << "[GL] " << glGetString(GL_RENDERER) << "\n";
    std::cout << "[GL] " << glGetString(GL_VERSION) << "\n";

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.08f, 0.12f, 0.18f, 1.0f);

    std::string cs_path = std::string(SHADER_DIR) + "/splat.comp.glsl";
    if (!compute_renderer_.initialize(cs_path.c_str(), window_.width(), window_.height())) {
        std::cerr << "Failed to initialize compute renderer\n";
        return false;
    }

    init_demo_body();
    return true;
}

void DemoScene::init_demo_body() {
    MaterialProperties mat0;
    mat0.r = 180; mat0.g = 200; mat0.b = 220; mat0.roughness = 0.3f; mat0.youngs_modulus = 2.0e6f;
    materials_.register_material(0, mat0);

    // ── Sphere: 8 rings × 16 segments → 153 anchors, 128 patches ──
    const int rings = 8, segments = 16;
    const float radius = 1.0f;
    std::vector<AnchorId> anchor_ids((rings+1) * (segments+1));
    std::vector<Vec3> rest_positions;

    for (int r = 0; r <= rings; ++r) {
        float phi = (r * kPi) / rings;
        for (int s = 0; s <= segments; ++s) {
            float theta = (s * kTwoPi) / segments;
            anchor_ids[r*(segments+1)+s] = topology_.add_anchor();
            rest_positions.push_back({
                radius * sinf(phi) * cosf(theta),
                radius * cosf(phi),
                radius * sinf(phi) * sinf(theta)
            });
        }
    }
    for (int r = 0; r < rings; ++r)
        for (int s = 0; s < segments; ++s) {
            int a = r*(segments+1)+s, b = a+1, c = (r+1)*(segments+1)+s, d = c+1;
            PatchCoefficients coef{};
            // Cupula on all patches — uniform outward bulge
            for (int i = 0; i < 8; ++i) coef.scalars[i] = 0.15f;
            topology_.add_patch({(AnchorId)a,(AnchorId)b,(AnchorId)c,(AnchorId)d},
                                0, BasisType::Cupula, coef);
        }

    topology_.finalize();
    std::cout << "[Body] " << rest_positions.size() << " anchors, "
              << topology_.patch_count() << " patches, 3 modal modes\n";
    body_ = std::make_unique<BodyRuntime>(std::move(topology_), 0);
    body_->set_materials(materials_);
    if (!body_->initialize(rest_positions)) { std::cerr << "[FAIL] body init\n"; return; }

    // Modal modes for live deformation
    ModalSystem modal;
    modal.add_mode({ModalMode::Type::Breathing, 0.15f, 2.0f, 0.0f});
    modal.add_mode({ModalMode::Type::BendingX, 0.10f, 3.5f, 0.0f});
    modal.add_mode({ModalMode::Type::BendingY, 0.08f, 4.0f, 0.5f});
    body_->set_modal_system(std::move(modal));
    body_->set_external_forces({{}, {}, 0.1f});  // No gravity by default

    std::cout << "[Body] " << rest_positions.size() << " anchors, "
              << topology_.patch_count() << " patches, 3 modal modes\n";
    std::cout << "[64B] AnchorRecord=" << sizeof(AnchorRecord)
              << " SurfaceRecord=" << sizeof(SurfaceRecord) << "\n";
}



void DemoScene::frame_loop() {
    auto last_time = std::chrono::high_resolution_clock::now();
    bool gravity_on = false;

    std::cout << "\n[GPU Compute Demo]\n";
    std::cout << "  G — toggle gravity\n";
    std::cout << "  Space — impulse kick\n";
    std::cout << "  Drag — orbit | Scroll — zoom\n";
    std::cout << "  ESC — quit\n\n";

    while (!window_.should_close()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        dt = std::min(dt, 0.05f);

        window_.poll_events();
        input_.poll();

        if (input_.key_down(SDLK_ESCAPE)) break;
        if (input_.key_down(SDLK_g)) {
            gravity_on = !gravity_on;
            body_->external_forces_mut().gravity = gravity_on ? Vec3{0,-9.81f,0} : Vec3{};
            std::cout << "[Gravity] " << (gravity_on?"ON":"OFF") << "\n";
        }
        if (input_.key_down(SDLK_SPACE)) {
            auto& state = body_->write_state();
            for (auto& a : state.anchors) a.velocity.y += 2.0f;
            std::cout << "[Kick] impulse\n";
        }
        if (input_.mouse_button(SDL_BUTTON_LEFT)) {
            camera_.orbit(input_.mouse_dx(), input_.mouse_dy());
        }
        if (input_.mouse_wheel() != 0) camera_.zoom(input_.mouse_wheel());

        // Camera orbits fixed origin — body deforms in place, doesn't drift
        camera_.target = Vec3{};

        // Physics
        if (body_) body_->step(1.0f/60.0f);

        gl_context_.make_current();
        glViewport(0, 0, window_.width(), window_.height());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (body_) {
            Vec3 cam_pos = camera_.position();
            // View matrix (lookAt) — COLUMN-MAJOR for OpenGL
            float view[16];
            {
                Vec3 forward = normalize_vec3(camera_.target - cam_pos);
                Vec3 world_up{0, 1, 0};
                Vec3 right = normalize_vec3(cross(world_up, forward));
                Vec3 up = cross(forward, right);
                // Column-major: each column is a basis vector, translation in column 3
                view[0]=right.x;  view[4]=right.y;  view[8]=right.z;  view[12]=-dot(right,cam_pos);
                view[1]=up.x;     view[5]=up.y;     view[9]=up.z;     view[13]=-dot(up,cam_pos);
                view[2]=-forward.x;view[6]=-forward.y;view[10]=-forward.z;view[14]=dot(forward,cam_pos);
                view[3]=0;        view[7]=0;        view[11]=0;       view[15]=1;
            }
            float proj[16] = {};
            {
                float fov = 70.0f * kPi / 180.0f;
                float aspect = (float)window_.width() / (float)window_.height();
                float tan_half = std::tan(fov / 2.0f);
                float near = 0.1f, far = 100.0f;
                proj[0] = 1.0f / (aspect * tan_half);
                proj[5] = 1.0f / tan_half;
                proj[10] = -(far + near) / (far - near);
                proj[11] = -2.0f * far * near / (far - near);
                proj[14] = -1.0f;
                proj[15] = 0.0f;
            }
            compute_renderer_.upload(body_.get());
            compute_renderer_.render(cam_pos, view, proj,
                                     (uint32_t)window_.width(), (uint32_t)window_.height());
        }

        gl_context_.swap();
    }
}

void DemoScene::shutdown() {
    compute_renderer_.shutdown();
    gl_context_.shutdown();
    window_.shutdown();
}

} // namespace delirium