#include "delirium/basis/basis_functions.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace delirium;

void test_shape_bases() {
    // Bilinear bases must sum to 1.0 everywhere (partition of unity)
    for (float u = 0; u <= 1.0f; u += 0.1f) {
        for (float v = 0; v <= 1.0f; v += 0.1f) {
            float sum = shape_base_00(u, v) + shape_base_01(u, v)
                      + shape_base_10(u, v) + shape_base_11(u, v);
            assert(std::abs(sum - 1.0f) < 1e-5f);
        }
    }
    std::cout << "  Shape bases: partition of unity ✓\n";
}

void test_detail_bases() {
    // Detail bases should average to ~0 over [0,1]² (zero-mean)
    for (auto* fn : DETAIL_BASES) {
        float sum = 0;
        int samples = 100;
        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < samples; ++j) {
                float u = (i + 0.5f) / samples;
                float v = (j + 0.5f) / samples;
                sum += fn(u, v);
            }
        }
        float mean = sum / (samples * samples);
        assert(std::abs(mean) < 0.05f); // near zero mean
    }
    std::cout << "  Detail bases: zero-mean ✓\n";
}

void test_color_dc() {
    // Color base 0 (DC) must be constant 1.0
    for (float u = 0; u <= 1.0f; u += 0.25f) {
        for (float v = 0; v <= 1.0f; v += 0.25f) {
            assert(std::abs(color_base_0(u, v) - 1.0f) < 1e-5f);
        }
    }
    std::cout << "  Color DC: constant 1.0 ✓\n";
}

void test_evaluation() {
    // Evaluate shape with uniform coefficients → should produce smooth output
    float K[8] = {0, 0, 0, 0, 0.1f, 0, 0, 0}; // only bulge
    std::span<const float, 8> sK(K);
    float val = evaluate_shape(0.5f, 0.5f, sK);
    // bulge at center = sin(π)·sin(π) = 0... use (0.25, 0.25)
    val = evaluate_shape(0.25f, 0.25f, sK);
    float expected = 0.1f * std::sin(M_PI * 0.25f) * std::sin(M_PI * 0.25f);
    assert(std::abs(val - expected) < 1e-5f);
    std::cout << "  Shape evaluation: correct ✓\n";
}

int main() {
    std::cout << "Math basis tests:\n";
    test_shape_bases();
    test_detail_bases();
    test_color_dc();
    test_evaluation();
    std::cout << "All math tests passed ✓\n";
    return 0;
}