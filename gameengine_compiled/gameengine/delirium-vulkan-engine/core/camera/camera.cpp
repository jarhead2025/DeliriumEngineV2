#include "camera.hpp"
#include <SDL.h>
#include <cmath>

namespace delirium {

Camera::Camera(float fov_y) : fov_y_(fov_y) {
    update_projection();
}

void Camera::update(float dt, const Input& input, float aspect) {
    aspect_ = aspect;
    update_projection();

    // Mouse rotation
    float sensitivity = 0.003f;
    yaw_ -= input.mouse_dx() * sensitivity;
    pitch_ += input.mouse_dy() * sensitivity;
    pitch_ = std::max(-1.5f, std::min(1.5f, pitch_));

    // WASD orbit
    float speed = 2.0f;
    if (input.key_held(SDLK_a)) yaw_ += speed * dt;
    if (input.key_held(SDLK_d)) yaw_ -= speed * dt;
    if (input.key_held(SDLK_w)) distance_ = std::max(1.0f, distance_ - 5.0f * dt);
    if (input.key_held(SDLK_s)) distance_ = std::min(50.0f, distance_ + 5.0f * dt);

    // Right-click pan
    if (input.mouse_button(SDL_BUTTON_RIGHT)) {
        Vec3 right = {std::cos(yaw_), 0, std::sin(yaw_)};
        Vec3 up = {0, 1, 0};
        target_ += right * (-input.mouse_dx() * 0.01f) + up * (input.mouse_dy() * 0.01f);
    }

    // Update position from orbit
    position_ = target_ + Vec3{
        std::cos(pitch_) * std::cos(yaw_) * distance_,
        std::sin(pitch_) * distance_,
        std::sin(pitch_) * std::sin(yaw_) * distance_
    };

    view_ = Mat4::look_at(position_, target_, {0, 1, 0});
}

void Camera::update_projection() {
    proj_ = Mat4::perspective(fov_y_ * 3.14159265f / 180.0f, aspect_, near_, far_);
}

m8::Frustum Camera::to_m8_frustum() const {
    Vec3 forward = (target_ - position_).normalized();
    Vec3 right = cross({0, 1, 0}, forward).normalized();
    Vec3 up = cross(forward, right).normalized();

    m8::Frustum f;
    f.origin = {position_.x, position_.y, position_.z};
    f.forward = {forward.x, forward.y, forward.z};
    f.right = {right.x, right.y, right.z};
    f.up = {up.x, up.y, up.z};
    f.near_plane = near_;
    f.far_plane = far_;
    f.fov_y = fov_y_ * 3.14159265f / 180.0f;
    f.aspect = aspect_;
    return f;
}

} // namespace delirium