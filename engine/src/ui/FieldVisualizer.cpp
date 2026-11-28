// ============================================================================
// FieldVisualizer.cpp — WS4 field visualization renderer implementation
// ============================================================================
// M4 scaffolding: renders field lines, potential maps, force vectors,
// energy density, stress iso-surfaces, temperature field slices.
// Part of v2.2Beta milestone M4 (WS4).
// NOTE: uses real engine Vec3 type + DeliRium namespace. Samples from:
//   ElectrostaticField::FieldAt(p), MagnetostaticField::FieldAt(p)
//   AeroField::DragForce/LiftForce, HydroField::BuoyancyForce/DragForce
//   ThermalState.temperature, ChargeState.total_charge/surface_potential
// ============================================================================

#include "delirium/ui/FieldVisualizer.h"
#include "delirium/physics/PhysicsWorld.h"
#include "delirium/physics/FieldPhysics.h"
#include "delirium/physics/ThermalField.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Types.h"

#include <cmath>
#include <algorithm>
#include <vector>

namespace DeliRium {

FieldVisualizer::FieldVisualizer(PhysicsWorld* world)
    : world_(world), mode_(FieldMode::None), enabled_(false),
      sample_count_(256), max_trace_length_(10.0f), trace_step_(0.01f),
      slice_normal_{0.0f, 1.0f, 0.0f}, slice_distance_(0.0f),
      color_min_(0.0f), color_max_(1.0f) {
    if (!world_) {
        world_ = new PhysicsWorld();
    }
}

void FieldVisualizer::SetMode(FieldMode mode) {
    mode_ = mode;
    Clear();
}

void FieldVisualizer::SetSeedPoints(const std::vector<Vec3>& seeds) {
    seed_points_ = seeds;
    Clear();
}

void FieldVisualizer::SetSlicePlane(Vec3 normal, float distance) {
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
    }
}

Vec3 FieldVisualizer::SampleVectorField(const Vec3& pos, PhysicsWorld* world) const {
    if (!world_) return {0.0f, 0.0f, 0.0f};
    // Sample vector field (force) at position from active field modules
    Vec3 total = {0.0f, 0.0f, 0.0f};
    // Electrostatic: requires charge coupling + external E-field/sources
    if (world_->ElectrostaticEnabled()) {
        total = total + world_->electrostatic.FieldAt(pos);
    }
    // Magnetostatic
    if (world_->MagnetostaticEnabled()) {
        total = total + world_->magnetostatic.FieldAt(pos);
    }
    return total;
}

float FieldVisualizer::SampleScalarField(const Vec3& pos, PhysicsWorld* world) const {
    if (!world_) return 0.0f;
    // Sample scalar field (potential, temperature) at position
    float val = 0.0f;
    // Electrostatic potential: V = k q / r (sum over point charges)
    if (world_->ElectrostaticEnabled()) {
        // Placeholder: would need to sum potentials from sources
        val = 0.0f;
    }
    return val;
}

Vec3 FieldVisualizer::ScalarToColor(float value, float min_val, float max_val) const {
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

FieldLine FieldVisualizer::TraceFieldLine(const Vec3& seed, PhysicsWorld* world) {
    FieldLine line;
    line.start = seed;
    Vec3 pos = seed;
    float length = 0.0f;

    for (int i = 0; i < 10000; i++) {
        Vec3 field = SampleVectorField(pos, world);
        float mag = std::sqrt(field.x*field.x + field.y*field.y + field.z*field.z);
        if (mag < 1e-6f) break; // Field too weak to continue

        // RK4 integration step
        Vec3 k1 = field * (1.0f / mag);
        Vec3 p2 = pos + k1 * (trace_step_ * 0.5f);
        Vec3 k2 = SampleVectorField(p2, world);
        float m2 = std::sqrt(k2.x*k2.x + k2.y*k2.y + k2.z*k2.z);
        k2 = k2 * (1.0f / (m2 + 1e-6f));
        Vec3 p3 = pos + k2 * (trace_step_ * 0.5f);
        Vec3 k3 = SampleVectorField(p3, world);
        float m3 = std::sqrt(k3.x*k3.x + k3.y*k3.y + k3.z*k3.z);
        k3 = k3 * (1.0f / (m3 + 1e-6f));
        Vec3 p4 = pos + k3 * trace_step_;
        Vec3 k4 = SampleVectorField(p4, world);
        float m4 = std::sqrt(k4.x*k4.x + k4.y*k4.y + k4.z*k4.z);
        k4 = k4 * (1.0f / (m4 + 1e-6f));

        Vec3 dir = (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (1.0f / 6.0f);
        float md = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
        dir = dir * (1.0f / (md + 1e-6f));

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

} // namespace DeliRium
