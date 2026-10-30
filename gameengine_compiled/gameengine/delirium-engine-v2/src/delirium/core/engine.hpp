#pragma once

namespace delirium {

// Forward declarations
class PhysicsWorld;
class Scene;
class Renderer;

/// Engine — owns the frame loop: physics → dirty → render
class Engine {
public:
    bool init(int width = 1280, int height = 720);
    void run();
    void shutdown();

private:
    void update_physics(float dt);
    void update_render();

    Scene* scene_ = nullptr;
    Renderer* renderer_ = nullptr;
    PhysicsWorld* physics_ = nullptr;
};

} // namespace delirium