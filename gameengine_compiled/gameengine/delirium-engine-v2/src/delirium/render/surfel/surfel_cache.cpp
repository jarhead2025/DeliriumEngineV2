#include "delirium/render/surfel/surfel_cache.hpp"
#include <cmath>

namespace delirium {

void SurfelCache::generate(const std::vector<Patch>& patches,
                           const std::vector<AnchorRecord>& anchors,
                           float density) {
    surfels.clear();

    // Adaptive grid: more samples in high-curvature regions
    int grid = static_cast<int>(8.0f * density);

    for (const auto& patch : patches) {
        for (int i = 0; i < grid; ++i) {
            for (int j = 0; j < grid; ++j) {
                float u = (i + 0.5f) / grid;
                float v = (j + 0.5f) / grid;

                auto sample = patch.evaluate(u, v, anchors);

                // Compute surfel radius from grid spacing
                Patch::AABB bounds = patch.compute_bounds(anchors);
                vec3 extent = {bounds.max.x - bounds.min.x,
                               bounds.max.y - bounds.min.y,
                               bounds.max.z - bounds.min.z};
                float avg_extent = (extent.x + extent.y + extent.z) / 3.0f;
                float radius = avg_extent / (grid * 1.5f); // 1.5x overlap for coverage

                surfels.push_back(Surfel{
                    .center = sample.position,
                    .normal = sample.normal,
                    .color = sample.color,
                    .radius = radius,
                    .patch_id = patch.patch_id,
                    .material_id = patch.material_id
                });
            }
        }
    }

    generation_id++;
}

void SurfelCache::clear() {
    surfels.clear();
    generation_id++;
}

} // namespace delirium