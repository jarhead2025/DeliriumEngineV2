#include "delirium/core/engine.hpp"
#include "delirium/core/scene.hpp"
#include "delirium/render/renderer.hpp"
#include "delirium/physics/physics_world.hpp"
#include <iostream>
#include <chrono>

namespace delirium {

bool Engine::init(int width, int height) {
    scene_ = new Scene();
    renderer_ = new Renderer();
    physics_ = new PhysicsWorld();

    if (!renderer_->init(width, height, "DeliRium Engine v2")) {
        std::cerr << "[Engine] Failed to initialize renderer\n";
        return false;
    }

    // Build demo scene
    scene_->build_demo();

    std::cout << "[Engine] Initialized\n";
    return true;
}

void Engine::run() {
    auto last_time = std::chrono::steady_clock::now();

    while (renderer_->poll_events()) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        // Cap dt to avoid spiral of death
        dt = std::min(dt, 0.033f);

        update_physics(dt);
        update_render();
    }
}

void Engine::update_physics(float dt) {
    if (physics_) {
        physics_->step(scene_, dt);
    }
}

void Engine::update_render() {
    if (scene_ && renderer_) {
        auto& cache = scene_->get_surfel_cache();
        auto camera = scene_->get_camera();
        renderer_->render(cache, camera.view, camera.projection);
    }
}

void Engine::shutdown() {
    delete physics_;
    delete renderer_;
    delete scene_;
    std::cout << "[Engine] Shutdown complete\n";
}

} // namespace delirium