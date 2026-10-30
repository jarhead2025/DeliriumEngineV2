#pragma once
#include <cstdint>
#include <vector>
#include "../core/Types.h"

namespace DeliRium {

class PSFBody;
struct Camera;

enum class LightType { Directional, Point, Headlight };

struct LightSource {
    LightType type = LightType::Directional;
    Vec3 dir_or_pos{0.4f, 0.7f, 0.5f}; // dir: light travels this way; point: world pos
    Vec3 color{1.f, 1.f, 1.f};
    float intensity = 1.f;
    float attenuation = 0.15f; // point lights
};

struct RayMarchLights {
    LightSource lights[4]{};
    int n_lights = 2;
    float ambient = 0.08f;
    float shininess = 32.f;
    float emissive_strength = 0.35f; // torus inherent glow
    float time_sec = 0.f;           // animates glow color
    bool shadows = true;
    int uv_samples = 6;   // equation probes per patch axis
    float hit_radius = 0.05f;
    int pixel_stride = 1; // 2 = faster preview
};

// True view-ray / lateral equation hit (not splat). Fills RGBA8888.
bool LateralRayMarch(const PSFBody& body, const Camera& cam, int W, int H,
                     const RayMarchLights& lights, std::vector<uint32_t>& out_rgba);

} // namespace DeliRium
