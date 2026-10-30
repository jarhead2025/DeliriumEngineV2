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

// Packed surfel for the layered lateral march (kernels/surfel_march.cl).
// BYTE-MATCHES the OpenCL `Surfel` struct (40 bytes). The GPU grid +
// per-layer draw/fill/compose consume this.
struct Surfel {
    float px,py,pz, nx,ny,nz, r,g,b, radius;
};
static_assert(sizeof(Surfel) == 40, "Surfel must match CL layout");

// Bake the LIVE deformed PSF surface into a surfel list for the layered
// lateral march. Reuses the exact deform math (bilinear anchors +
// deform_h(coef[8]) + analytic normal) so the bake tracks modal/dent/AMR
// deformation. uvn = samples per patch axis (density). out is cleared.
// `smooth_normals` (if non-null) is the per-anchor smooth shading normal
// (PSFBody::ComputeSmoothNormals); it is bilinearly interpolated per sample
// so shading matches the splat path (correct outward orientation).
bool BakeSurfels(const PSFBody& body, const Camera& cam, int uvn,
                 const std::vector<Vec3>* smooth_normals,
                 std::vector<Surfel>& out);

} // namespace DeliRium
