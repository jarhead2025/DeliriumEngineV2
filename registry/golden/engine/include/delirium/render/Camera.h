#pragma once
#include "../core/Types.h"

namespace DeliRium {

struct Camera {
    Vec3  position{0,0,5}, target{0,0,0}, up{0,1,0};
    float fov_deg = 45, near_p = 0.1f, far_p = 100;
    // Row-major MVP exactly as kernels/psf_splat.cl consumes (float16).
    void BuildMVP(float out[16], float aspect) const;
    // world-space ray through a pixel (px,py) of a width x height frame
    void PixelRay(float px, float py, int width, int height,
                  Vec3& origin, Vec3& dir) const;
};

} // namespace DeliRium
