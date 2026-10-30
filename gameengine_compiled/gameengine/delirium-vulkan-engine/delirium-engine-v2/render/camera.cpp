#include "camera.h"
#include <cmath>
#include <cstring>

namespace delirium::render {

Camera::Camera() {
    SDL_SetRelativeMouseMode(SDL_TRUE);
}

void Camera::process_mouse(int dx, int dy, float sensitivity) {
    yaw_   += dx * sensitivity * 0.2f;
    pitch_ -= dy * sensitivity * 0.2f;
    pitch_ = std::max(-89.99f, std::min(89.99f, pitch_));

    float rad_y = yaw_ * 3.14159265f / 180.0f;
    float rad_p = pitch_ * 3.14159265f / 180.0f;

    Vec3 dir;
    dir.x = std::cos(rad_y) * std::cos(rad_p);
    dir.y = std::sin(rad_p);
    dir.z = std::sin(rad_y) * std::cos(rad_p);

    Vec3 d = pos_ - target_;
    float dist = length(d);
    pos_ = target_ + dir * dist;
}

void Camera::update(float dt, const SDL_Event& event) {
    if (event.type == SDL_MOUSEWHEEL) {
        Vec3 d = pos_ - target_;
        float dist = length(d);
        dist -= event.wheel.y * 0.5f;
        dist = std::max(0.5f, std::min(50.0f, dist));
        if (length(d) > 0.001f) {
            pos_ = target_ + normalize_vec3(d) * dist;
        }
    }
}

void Camera::get_view(float out[16]) const {
    std::memset(out, 0, 16 * sizeof(float));
    Vec3 fz = target_ - pos_;
    float fl = length(fz);
    if (fl < 0.001f) return;
    fz = fz / fl;

    Vec3 up = {0, 1, 0};
    Vec3 sx = cross(up, fz);
    float sl = length(sx);
    if (sl > 0.001f) sx = sx / sl;
    Vec3 sy = cross(fz, sx);

    // COLUMN-MAJOR: each column is a basis vector
    out[0]  = sx.x; out[1]  = sx.y; out[2]  = sx.z; out[3]  = -dot(sx, pos_);
    out[4]  = sy.x; out[5]  = sy.y; out[6]  = sy.z; out[7]  = -dot(sy, pos_);
    out[8]  = fz.x; out[9]  = fz.y; out[10] = fz.z; out[11] = -dot(fz, pos_);
    out[12] = 0;    out[13] = 0;    out[14] = 0;    out[15] = 1;
}

void Camera::get_projection(float out[16], float aspect) const {
    std::memset(out, 0, 16 * sizeof(float));
    float f = 1.0f / std::tan(45.0f * 3.14159265f / 360.0f);
    float near = 0.1f, far = 100.0f;
    out[0]  = f / aspect;
    out[5]  = f;
    out[10] = -(far + near) / (far - near);
    out[11] = -2.0f * far * near / (far - near);
    out[14] = -1.0f;
}

void Camera::get_mvp(float out[16], float aspect) const {
    float view[16], proj[16];
    get_view(view);
    get_projection(proj, aspect);

    // Column-major multiply: out = proj * view
    std::memset(out, 0, 16 * sizeof(float));
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 4; ++k)
                out[c*4+r] += proj[k*4+r] * view[c*4+k];
}

} // namespace delirium::render