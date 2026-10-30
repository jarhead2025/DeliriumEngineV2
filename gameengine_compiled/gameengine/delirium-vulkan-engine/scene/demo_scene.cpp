#include "demo_scene.hpp"
#include "obj_loader/obj_loader.hpp"
#include "obj_loader/obj_model_builder.hpp"
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace delirium {

bool DemoScene::initialize(const std::string& obj_path, const std::string& tex_path) {
    // MUST be called before SDL_CreateWindow so SDL knows we want OpenGL
    OpenGLContext::setup_gl_attributes();

    if (!window_.initialize("DeliRium Engine - Ray Marching", 1280, 720)) {
        std::cerr << "Failed to create window\n";
        return false;
    }

    if (!gl_context_.initialize(window_.handle())) {
        std::cerr << "Failed to create OpenGL context\n";
        return false;
    }

    // OpenGL version info
    std::cout << "[GL] OpenGL " << glGetString(GL_VERSION) << " loaded\n";

    // Essential GL state
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);

    // ─── Ray marching shader (primary renderer) ──────────────────
    std::string rm_vert = std::string(SHADER_DIR) + "/raymarch.vert.glsl";
    std::string rm_frag = std::string(SHADER_DIR) + "/raymarch.frag.glsl";
    if (!raymarch_shader_.compile(rm_vert.c_str(), rm_frag.c_str())) {
        std::cerr << "Failed to compile ray marching shaders\n";
        return false;
    }

    if (!raymarch_renderer_.initialize()) {
        std::cerr << "Failed to initialize ray marching renderer\n";
        return false;
    }

    // ─── Surfel shader (fallback / comparison) ───────────────────
    std::string sf_vert = std::string(SHADER_DIR) + "/surfel.vert.glsl";
    std::string sf_frag = std::string(SHADER_DIR) + "/surfel.frag.glsl";
    if (!surfel_shader_.compile(sf_vert.c_str(), sf_frag.c_str())) {
        std::cerr << "Failed to compile surfel shaders\n";
        return false;
    }

    if (!surfel_renderer_.initialize()) {
        std::cerr << "Failed to initialize surfel renderer\n";
        return false;
    }

    // Store paths for later use
    obj_path_ = obj_path;
    texture_path_ = tex_path;

    // Choose scene: OBJ model or built-in grid
    if (!obj_path.empty()) {
        init_obj_scene(obj_path, tex_path);
    } else {
        init_grid_scene();
    }

    return true;
}

void DemoScene::init_grid_scene() {
    // Create a deformable body — 8x8 grid of anchors forming a plane
    m8::BodyTopology topology;
    const int grid = 8;
    for (int i = 0; i < grid * grid; ++i) {
        topology.add_anchor();
    }

    // Create patches (quads) from the grid
    for (int y = 0; y < grid - 1; ++y) {
        for (int x = 0; x < grid - 1; ++x) {
            std::array<m8::AnchorId, 4> anchors = {
                static_cast<m8::AnchorId>(y * grid + x),
                static_cast<m8::AnchorId>(y * grid + x + 1),
                static_cast<m8::AnchorId>((y + 1) * grid + x),
                static_cast<m8::AnchorId>((y + 1) * grid + x + 1),
            };
            topology.add_patch(anchors, 0);
        }
    }

    std::string error;
    topology.finalize(&error);

    // Create rest positions — a flat plane
    std::vector<m8::Vec3> positions;
    for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
            positions.push_back({
                (float)x - (float)(grid / 2),
                0.0f,
                (float)y - (float)(grid / 2)
            });
        }
    }

    m8::BodyRuntime runtime(std::move(topology), 0);

    // Create material properties
    m8::MaterialRegistry registry;
    m8::MaterialProperties mat;
    mat.youngs_modulus = 1.0e6f;
    mat.poisson_ratio = 0.3f;
    mat.density = 1000.0f;
    mat.yield_strength = 5.0e5f;
    mat.r = 180; mat.g = 160; mat.b = 140;
    registry.register_material(0, mat);
    runtime.set_materials(std::move(registry));
    runtime.initialize(positions, &error);

    // Apply gravity
    m8::ExternalForce ef;
    ef.gravity = {0.0f, -9.81f, 0.0f};
    runtime.set_external_forces(ef);

    manager_.add_body(std::move(runtime));
    active_bodies_.push_back(manager_.get_body(0));

    // Set up scene budget
    m8::SceneBudget budget;
    budget.max_bodies = 64;
    budget.max_surfels = 100000;
    budget.max_constraints = 2048;
    manager_.set_budget(budget);

    std::cout << "Scene initialized: " << active_bodies_.size() << " body, "
              << positions.size() << " anchors, "
              << (grid - 1) * (grid - 1) << " patches\n";
    std::cout << "[Ctrl] Press R to toggle ray marching / surfel rendering\n";
}

void DemoScene::init_obj_scene(const std::string& obj_path, const std::string& tex_path) {
    std::cout << "[Scene] Loading OBJ model: " << obj_path << "\n";

    // Build the model from OBJ
    ObjBuildResult build_result;
    std::string error;
    auto runtime = build_model_from_obj(obj_path, tex_path, &build_result, &error);

    if (!build_result.success) {
        std::cerr << "[Scene] Failed to build model: " << error << "\n";
        // Fall back to grid scene
        init_grid_scene();
        return;
    }

    // Load texture if provided
    if (!tex_path.empty()) {
        raymarch_renderer_.load_texture(tex_path);
    }

    // Apply gravity
    m8::ExternalForce ef;
    ef.gravity = {0.0f, -9.81f, 0.0f};
    runtime.set_external_forces(ef);

    manager_.add_body(std::move(runtime));
    active_bodies_.push_back(manager_.get_body(0));

    // Set up scene budget
    m8::SceneBudget budget;
    budget.max_bodies = 64;
    budget.max_surfels = 100000;
    budget.max_constraints = 2048;
    manager_.set_budget(budget);

    std::cout << "[Scene] OBJ model loaded: " << build_result.num_anchors << " anchors, "
              << build_result.num_patches << " patches, "
              << build_result.num_quads << " quads, "
              << build_result.num_remaining_tris << " remaining triangles\n";
    std::cout << "[Ctrl] Press R to toggle ray marching / surfel rendering\n";
}

void DemoScene::run() {
    frame_loop();
}

void DemoScene::frame_loop() {
    auto last_time = std::chrono::high_resolution_clock::now();

    while (!window_.should_close()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        dt = std::min(dt, 0.05f);

        window_.poll_events();
        input_.poll();

        // Toggle renderer with 'R' key
        if (input_.key_down(SDLK_r)) {
            use_ray_marching_ = !use_ray_marching_;
            std::cout << "[Mode] " << (use_ray_marching_ ? "Ray Marching" : "Surfel Point Sprites") << "\n";
        }

        camera_.update(dt, input_, (float)window_.width() / (float)window_.height());

        // Step physics
        m8::Frustum frustum = camera_.to_m8_frustum();
        std::string error;
        manager_.step(dt, &frustum, &error);

        // Render
        gl_context_.make_current();
        glViewport(0, 0, window_.width(), window_.height());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (use_ray_marching_) {
            // ─── Ray marching path ───────────────────────────────
            raymarch_renderer_.update_geometry(active_bodies_);
            raymarch_renderer_.draw(raymarch_shader_, camera_,
                                    window_.width(), window_.height());

            surfel_count_ = raymarch_renderer_.patch_count() * 16; // 4x4 samples per patch
        } else {
            // ─── Surfel point sprite path ────────────────────────
            surfel_renderer_.update_surfels(active_bodies_, camera_.position());
            surfel_count_ = surfel_renderer_.surfel_count();

            auto mvp = camera_.mvp_matrix();
            surfel_shader_.set_mat4("u_mvp", mvp.m.data());
            surfel_shader_.set_vec3("u_cam_pos", camera_.position().x,
                                    camera_.position().y, camera_.position().z);
            surfel_renderer_.draw(surfel_shader_, window_.width(), window_.height());
        }

        // First-frame diagnostic
        static bool first_frame = true;
        if (first_frame) {
            first_frame = false;
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                std::cerr << "[GL] Error after first frame: 0x" << std::hex << err << "\n";
            }
            std::cout << "[Render] First frame: " << surfel_count_ << " surfels, "
                      << "cam=(" << camera_.position().x << ", "
                      << camera_.position().y << ", "
                      << camera_.position().z << ")\n";
        }

        gl_context_.swap();

        // Update title
        char title[256];
        snprintf(title, sizeof(title), "DeliRium Engine - %d surfels [%s]",
                 surfel_count_, use_ray_marching_ ? "RayMarch" : "Surfel");
        window_.update_title(title);
    }
}

void DemoScene::shutdown() {
    raymarch_renderer_.shutdown();
    raymarch_shader_.shutdown();
    surfel_renderer_.shutdown();
    surfel_shader_.shutdown();
    gl_context_.shutdown();
    window_.shutdown();
}

} // namespace delirium