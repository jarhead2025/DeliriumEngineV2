#pragma once
#include "math/mat4.hpp"
#include "delirium_m8/types.hpp"
#include "window/input.hpp"

namespace delirium {

class Camera {
public:
    explicit Camera(float fov_y = 60.0f);

    void update(float dt, const Input& input, float aspect);
    Mat4 view_matrix() const { return view_; }
    Mat4 projection_matrix() const { return proj_; }
    Mat4 mvp_matrix() const { return proj_ * view_; }
    Mat4 view_inv() const { return view_.inverse(); }
    Mat4 proj_inv() const { return proj_.inverse(); }
    Vec3 position() const { return position_; }

    // Extract frustum for physics culling (matches M8 Frustum type)
    m8::Frustum to_m8_frustum() const;

    void set_aspect(float a) { aspect_ = a; update_projection(); }

private:
    void update_projection();

    float yaw_ = 0.0f;
    float pitch_ = -0.2f;
    float distance_ = 10.0f;
    Vec3 target_{};
    Vec3 position_;

    float fov_y_ = 60.0f;
    float aspect_ = 16.0f / 9.0f;
    float near_ = 0.1f;
    float far_ = 1000.0f;

    Mat4 view_;
    Mat4 proj_;
};

} // namespace delirium