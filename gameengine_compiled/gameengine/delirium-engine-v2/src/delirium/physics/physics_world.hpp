#pragma once
#include "delirium/core/scene.hpp"

namespace delirium {

class PhysicsWorld {
public:
    /// Step physics for one frame
    void step(Scene* scene, float dt);
};

} // namespace delirium