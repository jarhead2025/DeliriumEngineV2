#include "delirium/render/Camera.h"
#include <cmath>

namespace DeliRium {

// Row-major MVP = Perspective * LookAt, depth mapped to [0,1].
// MUST mirror look_at()/perspective() in tools/psf_validate.py.
void Camera::BuildMVP(float out[16], float aspect) const {
    Vec3 f = { target.x-position.x, target.y-position.y, target.z-position.z };
    float fl = std::sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
    f = { f.x/fl, f.y/fl, f.z/fl };
    Vec3 r = { f.y*up.z - f.z*up.y, f.z*up.x - f.x*up.z, f.x*up.y - f.y*up.x };
    float rl = std::sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
    r = { r.x/rl, r.y/rl, r.z/rl };
    Vec3 u = { r.y*f.z - r.z*f.y, r.z*f.x - r.x*f.z, r.x*f.y - r.y*f.x };

    float V[16] = { r.x, r.y, r.z, -(r.x*position.x + r.y*position.y + r.z*position.z),
                    u.x, u.y, u.z, -(u.x*position.x + u.y*position.y + u.z*position.z),
                   -f.x,-f.y,-f.z,  (f.x*position.x + f.y*position.y + f.z*position.z),
                    0, 0, 0, 1 };
    const float t = 1.0f / std::tan(fov_deg * 3.14159265358979f / 360.0f);
    const float zz = far_p / (near_p - far_p);
    const float zw = near_p * far_p / (near_p - far_p);
    float P[16] = { t/aspect, 0, 0, 0,
                    0, t, 0, 0,
                    0, 0, zz, zw,
                    0, 0, -1, 0 };
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += P[i*4+k] * V[k*4+j];
            out[i*4+j] = s;
        }
}

void Camera::PixelRay(float px, float py, int width, int height,
                      Vec3& origin, Vec3& dir) const {
    Vec3 f = { target.x-position.x, target.y-position.y, target.z-position.z };
    float fl = std::sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
    f = { f.x/fl, f.y/fl, f.z/fl };
    Vec3 r = { f.y*up.z - f.z*up.y, f.z*up.x - f.x*up.z, f.x*up.y - f.y*up.x };
    float rl = std::sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
    r = { r.x/rl, r.y/rl, r.z/rl };
    const Vec3 u = { r.y*f.z - r.z*f.y, r.z*f.x - r.x*f.z, r.x*f.y - r.y*f.x };
    const float t  = std::tan(fov_deg * 3.14159265358979f / 360.0f);
    const float nx = ( (px / (width  - 1)) * 2.0f - 1.0f) * t
                   * (float(width) / float(height));
    const float ny = (1.0f - (py / (height - 1)) * 2.0f) * t;
    origin = position;
    dir = { f.x + nx*r.x + ny*u.x, f.y + nx*r.y + ny*u.y,
            f.z + nx*r.z + ny*u.z };
    const float dl = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir = { dir.x/dl, dir.y/dl, dir.z/dl };
}

} // namespace DeliRium
