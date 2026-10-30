#pragma once
#include "../core/types.h"
#include <SDL2/SDL.h>
#include <array>

namespace delirium::render {

class Camera {
public:
    Camera();

    void update(float dt, const SDL_Event& event);
    void process_mouse(int dx, int dy, float sensitivity = 1.0f);

    Vec3 position() const { return pos_; }

    void get_view(float out[16]) const;
    void get_projection(float out[16], float aspect) const;
    void get_mvp(float out[16], float aspect) const;

private:
    Vec3 pos_    = {0, 2, 5};
    Vec3 target_ = {0, 0, 0};
    float yaw_   = -90.0f;
    float pitch_ = -15.0f;
};

} // namespace delirium::render