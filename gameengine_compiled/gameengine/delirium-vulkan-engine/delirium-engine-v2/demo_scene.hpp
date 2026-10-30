#pragma once
#include "core/opengl/context.hpp"
#include "core/window/window.hpp"
#include "core/window/input.hpp"
#include "core/render/compute_renderer.hpp"
#include "core/body_runtime.h"
#include "core/material_registry.h"
#include <memory>
#include <string>
#include <cmath>

namespace delirium {

struct OrbitCamera {
    float theta = 0.5f, phi = 0.8f, radius = 4.0f;
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

class DemoScene {
public:
    bool initialize();
    void run() { frame_loop(); }
    void shutdown();
private:
    void init_demo_body();
    void frame_loop();

    Window window_;
    Input input_;
    OpenGLContext gl_context_;
    ComputeRenderer compute_renderer_;
    OrbitCamera camera_;
    BodyTopology topology_;
    std::unique_ptr<BodyRuntime> body_;
    MaterialRegistry materials_;
};

} // namespace delirium