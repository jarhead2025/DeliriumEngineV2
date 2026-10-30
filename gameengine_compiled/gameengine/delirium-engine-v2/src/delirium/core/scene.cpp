#include "delirium/core/scene.hpp"
#include <cmath>

namespace delirium {

Scene::Scene() {
    camera.position = {0, 0, 8};
    camera.target = {0, 0, 0};
    camera.up = {0, 1, 0};
    camera.fov = 50.0f * M_PI / 180.0f;
    camera.aspect = 1280.0f / 720.0f;
    camera.update_matrices();
}

void Scene::build_demo() {
    // Build an 8×8 grid of patches forming a plane
    const int grid = 8;
    const float spacing = 0.5f;

    // Create anchors in a grid
    for (int j = 0; j <= grid; ++j) {
        for (int i = 0; i <= grid; ++i) {
            AnchorRecord anchor;
            float x = (i - grid * 0.5f) * spacing;
            float z = (j - grid * 0.5f) * spacing;
            anchor.position = {x, 0, z};
            anchor.normal = {0, 1, 0};
            anchors.push_back(std::move(anchor));
        }
    }

    // Create patches (each patch = 4 anchors)
    int patch_id = 0;
    for (int j = 0; j < grid; ++j) {
        for (int i = 0; i < grid; ++i) {
            Patch patch;
            int idx = j * (grid + 1) + i;
            patch.anchor_indices = {idx, idx + 1, idx + grid + 1, idx + grid + 2};
            patch.patch_id = patch_id++;

            // Give each patch a subtle color variation
            patch.surface.color_K[0] = 0.3f + 0.1f * std::sin(i * 0.5f);
            patch.surface.color_K[1] = 0.5f + 0.1f * std::cos(j * 0.5f);
            patch.surface.color_K[2] = 0.7f;

            // Small bulge in center patches
            if (i > 2 && i < 5 && j > 2 && j < 5) {
                patch.surface.shape_K[4] = 0.1f; // bulge basis
            }

            patches.push_back(std::move(patch));
        }
    }

    // Default material
    MaterialRecord mat;
    mat.albedo = {0.8f, 0.8f, 0.8f};
    mat.roughness = 0.5f;
    materials.push_back(std::move(mat));

    update_surfel_cache();
}

void Scene::update_surfel_cache() {
    surfel_cache.generate(patches, anchors, 1.0f);
    generation_id_++;
}

} // namespace delirium