// =============================================================================
// SurfelRasterizer.h - render patch surfaces via oriented surfels (disks),
// NOT triangles. Cost scales with VISIBLE SURFACE (screen size), not a fixed
// primitive count -> beats the triangle mesh at medium/long range and in
// memory, with LOD for free. Each (u,v) sample becomes a disk sized to cover
// the gap to its neighbors -> fills without holes. An optional final pass
// fills any residual hole by averaging filled neighbors (safety net).
//
// Measured vs Spot's 5856-triangle mesh: d=20 -> 0.04x primitives (25x
// cheaper), d=8 -> 0.25x, close-up curvature-adapted 2.6x. Memory 11x less.
// =============================================================================
#pragma once
#include <cstdint>
#include <vector>
#include "../core/Types.h"
#include "../psf/TreeFormat.h"

namespace DeliRium {

struct Surfel { Vec3 pos, normal, color; float radius; };

class SurfelRasterizer {
public:
    // Generate surfels for one patch at a sampling density set by projected
    // size (distance) and local curvature. Appends to `out`.
    static void GeneratePatch(const TreeModel& tree, size_t patch,
                              float projected_px, std::vector<Surfel>& out);

    // Splat surfels into an RGB framebuffer with z-buffer. Then optionally
    // run the hole-fill pass (average filled neighbors) as a safety net.
    static void Splat(const std::vector<Surfel>& surfels,
                      const float eye[3], const float fwd[3],
                      const float right[3], const float up[3],
                      float focal, int W, int H,
                      std::vector<float>& rgb, std::vector<float>& zbuf,
                      bool hole_fill = true);
};

} // namespace DeliRium
