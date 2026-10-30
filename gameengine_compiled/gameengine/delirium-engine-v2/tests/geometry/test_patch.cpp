#include "delirium/geometry/patch/patch.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace delirium;

int main() {
    std::cout << "Geometry patch tests:\n";

    // Build 4 anchors in a unit square
    std::vector<AnchorRecord> anchors(4);
    anchors[0].position = {-0.5f, 0, -0.5f}; // bottom-left
    anchors[1].position = {0.5f, 0, -0.5f};  // bottom-right
    anchors[2].position = {-0.5f, 0, 0.5f};  // top-left
    anchors[3].position = {0.5f, 0, 0.5f};   // top-right
    for (auto& a : anchors) a.normal = {0, 1, 0};

    // Create patch
    Patch patch;
    patch.anchor_indices = {0, 1, 2, 3};

    // Test 1: bilinear at center → should be (0, 0, 0)
    vec3 center = patch.evaluate_bilinear(0.5f, 0.5f, anchors);
    assert(std::abs(center.x) < 1e-5f);
    assert(std::abs(center.y) < 1e-5f);
    assert(std::abs(center.z) < 1e-5f);
    std::cout << "  Bilinear center: (0,0,0) ✓\n";

    // Test 2: bilinear at corner → should match anchor
    vec3 corner = patch.evaluate_bilinear(0, 0, anchors);
    assert(std::abs(corner.x - (-0.5f)) < 1e-5f);
    std::cout << "  Bilinear corner: matches anchor ✓\n";

    // Test 3: normal should be (0, 1, 0) on flat patch
    vec3 normal = patch.evaluate_normal(0.5f, 0.5f, anchors);
    assert(std::abs(normal.y - 1.0f) < 1e-3f);
    std::cout << "  Normal: (0,1,0) ✓\n";

    // Test 4: full evaluation with displacement
    patch.surface.shape_K[4] = 0.2f; // bulge
    auto sample = patch.evaluate(0.5f, 0.5f, anchors);
    assert(sample.position.y > 0); // bulge pushes up along normal
    std::cout << "  Shape displacement: pushes along normal ✓\n";

    // Test 5: AABB bounds contain all anchors
    auto bounds = patch.compute_bounds(anchors);
    for (const auto& a : anchors) {
        vec3 p = a.current_position();
        assert(p.x >= bounds.min.x && p.x <= bounds.max.x);
        assert(p.y >= bounds.min.y && p.y <= bounds.max.y);
        assert(p.z >= bounds.min.z && p.z <= bounds.max.z);
    }
    std::cout << "  AABB bounds: contain all anchors ✓\n";

    std::cout << "All geometry tests passed ✓\n";
    return 0;
}