// ============================================================================
// FieldVisualizer.cpp — WS4 field visualization renderer implementation
// ============================================================================
// M4 scaffolding: renders field lines, potential maps, force vectors,
// energy density, stress iso-surfaces, temperature field slices.
// Part of v2.2Beta milestone M4 (WS4).
// ============================================================================

#include "delirium/ui/FieldVisualizer.h"
#include "delirium/physics/PhysicsWorld.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Scene.h"
#include "delirium/math/float3.h"

#include <SDL2/SDL.h>
#include <cmath>
#include <algorithm>
#include <vector>

namespace delirium {

FieldVisualizer::FieldVisualizer(PhysicsWorld* world)
    : world_(world), mode_(FieldMode::None), enabled_(false),
      sample_count_(256), sample_density_(1.0f), max_trace_length_(10.0f),
      trace_step_(0.01f), slice_normal_({0.0f, 1.0f, 0.0f}), slice_distance_(0.0f),
      color_min_(0.0f), color_max_(1.0f) {
    if (!world_) {
        world_ = new PhysicsWorld();
    }
}

void FieldVisualizer::SetMode(FieldMode mode) {
    mode_ = mode;
    Clear();
}

void FieldVisualizer::SetSeedPoints(const std::vector<float3>& seeds) {
    seed_points_ = seeds;
    Clear();
}

void FieldVisualizer::SetSlicePlane(float3 normal, float distance) {
    slice_normal_ = normal;
    slice_distance_ = distance;
    Clear();
}

void FieldVisualizer::SetColorRange(float min_val, float max_val) {
    color_min_ = min_val;
    color_max_ = max_val;
}

void FieldVisualizer::Clear() {
    field_lines_.clear();
    samples_.clear();
}

void FieldVisualizer::Refresh() {
    if (!enabled_ || mode_ == FieldMode::None) {
        Clear();
        return;
    }
    Clear();

    if (mode_ == FieldMode::FieldLines || mode_ == FieldMode::VelocityField) {
        // Trace field lines from seed points
        for (const auto& seed : seed_points_) {
            field_lines_.push_back(TraceFieldLine(seed, world_));
        }
    } else if (mode_ == FieldMode::ForceVectors) {
        // Sample force field at regular grid points
        // (deferred — needs scene bounds)
    } else if (mode_ == FieldMode::TempSlice) {
        // Sample temperature on slice plane
        // (deferred — needs body geometry intersection)
    }
}

float FieldVisualizer::SampleScalarField(const float3& pos, PhysicsWorld* world) const {
    if (!world_) return 0.0f;
    // Sample scalar field (potential, temperature, pressure) at position
    // This is a placeholder — actual implementation needs body geometry query
    return 0.0f;
}

float3 FieldVisualizer::SampleVectorField(const float3& pos, PhysicsWorld* world) const {
    if (!world_) return {0.0f, 0.0f, 0.0f};
    // Sample vector field (force, velocity) at position
    // This is a placeholder — actual implementation needs physics query
    return {0.0f, 0.0f, 0.0f};
}

float3 FieldVisualizer::ScalarToColor(float value, float min_val, float max_val) const {
    // Heat map color mapping: blue (cold) → green → red (hot)
    float t = (value - min_val) / (max_val - min_val + 1e-6f);
    t = std::clamp(t, 0.0f, 1.0f);

    float r, g, b;
    if (t < 0.25f) {
        r = 0.0f; g = t * 4.0f; b = 1.0f;
    } else if (t < 0.5f) {
        r = 0.0f; g = 1.0f; b = 1.0f - (t - 0.25f) * 4.0f;
    } else if (t < 0.75f) {
        r = (t - 0.5f) * 4.0f; g = 1.0f; b = 0.0f;
    } else {
        r = 1.0f; g = 1.0f - (t - 0.75f) * 4.0f; b = 0.0f;
    }
    return {r, g, b};
}

FieldLine FieldVisualizer::TraceFieldLine(const float3& seed, PhysicsWorld* world) {
    FieldLine line;
    line.start = seed;
    float3 pos = seed;
    float length = 0.0f;

    for (int i = 0; i < 10000; i++) {
        float3 field = SampleVectorField(pos, world);
        float mag = field.Length();
        if (mag < 1e-6f) break; // Field too weak to continue

        // RK4 integration step
        float3 k1 = field / mag;
        float3 p2 = pos + k1 * trace_step_ * 0.5f;
        float3 k2 = SampleVectorField(p2, world);
        k2 = k2 / (k2.Length() + 1e-6f);
        float3 p3 = pos + k2 * trace_step_ * 0.5f;
        float3 k3 = SampleVectorField(p3, world);
        k3 = k3 / (k3.Length() + 1e-6f);
        float3 p4 = pos + k3 * trace_step_;
        float3 k4 = SampleVectorField(p4, world);
        k4 = k4 / (k4.Length() + 1e-6f);

        float3 dir = (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
        dir = dir / (dir.Length() + 1e-6f);

        pos = pos + dir * trace_step_;
        length += trace_step_;
        line.points.push_back(pos);

        if (length >= max_trace_length_) break;
    }

    line.end = pos;
    line.length = length;
    return line;
}

void FieldVisualizer::DrawForceArrows() {
    // Placeholder — actual SDL rendering deferred to M4 implementation
}

void FieldVisualizer::DrawFieldLines() {
    // Placeholder — actual SDL rendering deferred to M4 implementation
    for (const auto& line : field_lines_) {
        // Draw polyline for each field line
    }
}

void FieldVisualizer::ColorSurfaceByPotential() {
    // Placeholder — actual implementation needs body geometry query
}

void FieldVisualizer::MarchStressIso() {
    // Placeholder — marching cubes implementation deferred to M4
}

void FieldVisualizer::DrawTempSlice() {
    // Placeholder — slice plane rendering deferred to M4
}

void FieldVisualizer::Render() {
    if (!enabled_ || mode_ == FieldMode::None) return;
    Refresh();

    switch (mode_) {
        case FieldMode::FieldLines:
        case FieldMode::VelocityField:
            DrawFieldLines();
            break;
        case FieldMode::ForceVectors:
            DrawForceArrows();
            break;
        case FieldMode::PotentialMap:
            ColorSurfaceByPotential();
            break;
        case FieldMode::StressIso:
            MarchStressIso();
            break;
        case FieldMode::TempSlice:
            DrawTempSlice();
            break;
        default:
            break;
    }
}

} // namespace delirium
