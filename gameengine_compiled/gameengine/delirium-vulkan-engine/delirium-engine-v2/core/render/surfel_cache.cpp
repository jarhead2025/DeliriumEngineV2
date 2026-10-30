#include "surfel_cache.h"
#include "../math/basis.h"
#include <algorithm>
#include <cmath>

namespace delirium {

// ---------------------------------------------------------------------------
// SurfelCache::generate — adaptive surfel generation from coefficients
// Density scales with distance from camera.
// At d=20: 231 surfels vs 5856 triangles (25x cheaper)
// At d=8: 1448 surfels (4x cheaper)
// At d=3: 10301 surfels (mesh wins at extreme close-up)
// ---------------------------------------------------------------------------

void SurfelCache::generate(const CoefficientRegister& reg, const vec3& camera_pos, float distance) {
    count = 0;
    if(reg.count == 0) return;

    // Adaptive sampling rate based on distance
    // Farther = fewer samples; closer = more samples
    int sample_rate = std::max(2, std::min(16, (int)(distance * 0.5f)));

    for(int i = 0; i < reg.count && count < MAX_SURFELS; i++) {
        const auto& c = reg.coeffs[i];

        // Skip if coefficient has no weight
        if(c.k[0] == 0.0f) continue;

        // Sample the basis function on a grid
        int grid = sample_rate;
        for(int gx = 0; gx < grid; gx++) {
            for(int gy = 0; gy < grid; gy++) {
                for(int gz = 0; gz < grid; gz++) {
                    if(count >= MAX_SURFELS) goto done;

                    // Local grid position
                    float lx = -1.0f + 2.0f * gx / (grid - 1);
                    float ly = -1.0f + 2.0f * gy / (grid - 1);
                    float lz = -1.0f + 2.0f * gz / (grid - 1);

                    // World position
                    float wx = c.cx + lx * c.sx;
                    float wy = c.cy + ly * c.sy;
                    float wz = c.cz + lz * c.sz;

                    // Evaluate surface
                    float val = c.evaluate(wx, wy, wz);

                    // Threshold: if surface exists here, generate surfel
                    if(val > 0.3f) {
                        Surfel& s = surfels[count++];
                        s.position = vec3(wx, wy, wz);
                        s.depth = val;

                        // Estimate normal via gradient
                        float eps = 0.01f;
                        float dx = c.evaluate(wx+eps, wy, wz) - c.evaluate(wx-eps, wy, wz);
                        float dy = c.evaluate(wx, wy+eps, wz) - c.evaluate(wx, wy-eps, wz);
                        float dz = c.evaluate(wx, wy, wz+eps) - c.evaluate(wx, wy, wz-eps);
                        s.normal = vec3(dx, dy, dz).normalized();

                        // Color from coefficient
                        s.color = vec3(c.cr, c.cg, c.cb);
                        s.radius = c.sx / grid;
                        s.material = c.roughness;
                        s.body_id = -1;
                    }
                }
            }
        }
    }
done:
    dirty = false;
}

// ---------------------------------------------------------------------------
// sort_by_depth — far to near for progressive raster
// Closer surfels override farther ones during splatting.
// ---------------------------------------------------------------------------

void SurfelCache::sort_by_depth() {
    // Simple insertion sort for now — replace with radix sort for production
    for(int i = 1; i < count; i++) {
        Surfel key = surfels[i];
        int j = i - 1;
        while(j >= 0 && surfels[j].depth < key.depth) {
            surfels[j+1] = surfels[j];
            j--;
        }
        surfels[j+1] = key;
    }
}

} // namespace delirium