#pragma once
#include "window/window.hpp"
#include "window/input.hpp"
#include "camera/camera.hpp"
#include "opengl/context.hpp"
#include "opengl/shader.hpp"
#include "surfel_renderer.hpp"
#include "raymarch_renderer.hpp"
#include "delirium_m8/multi_body_manager.hpp"
#include <string>
#include <vector>

namespace delirium {

class DemoScene {
public:
    /// Initialize the scene.
    /// If obj_path is non-empty, loads that OBJ file as the model.
    /// Otherwise falls back to the built-in 8x8 grid demo.
    bool initialize(const std::string& obj_path = "", const std::string& texture_path = "");
    void run();
    void shutdown();

private:
    void init_grid_scene();       // Built-in 8x8 deformable grid
    void init_obj_scene(const std::string& obj_path, const std::string& texture_path);  // OBJ model
    void frame_loop();

    Window window_;
    Input input_;
    Camera camera_;
    OpenGLContext gl_context_;

    // Two renderers: surfel (point sprites) and ray marching (implicit surface)
    ShaderProgram surfel_shader_;
    SurfelRenderer surfel_renderer_;

    ShaderProgram raymarch_shader_;
    RayMarchRenderer raymarch_renderer_;

    bool use_ray_marching_ = true;  // toggle with 'R' key

    m8::MultiBodyManager manager_;
    std::vector<m8::BodyRuntime*> active_bodies_;

    uint32_t surfel_count_ = 0;

    std::string obj_path_;
    std::string texture_path_;
};

} // namespace delirium