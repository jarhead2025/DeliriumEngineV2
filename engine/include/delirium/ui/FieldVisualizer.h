// ============================================================================
// FieldVisualizer.h — WS4 field visualization panel header
// ============================================================================
// M4 scaffolding: renders field lines, potential maps, force vectors,
// energy density, stress iso-surfaces, temperature field slices.
// Part of v2.2Beta milestone M4 (WS4).
// NOTE: uses real engine Vec3 type + DeliRium namespace. Samples from:
//   ElectrostaticField::FieldAt(p), MagnetostaticField::FieldAt(p)
//   AeroField::DragForce/LiftForce, HydroField::BuoyancyForce/DragForce
//   ThermalState.temperature, ChargeState.total_charge/surface_potential
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace DeliRium {

// Forward declarations
class PhysicsWorld;
class PSFBody;
class GameObject;

// ============================================================================
// FieldMode — enumeration of supported field visualization modes
// ============================================================================
enum class FieldMode {
    None,            // No field visualization
    FieldLines,      // Trace field lines via RK4
    PotentialMap,    // Color surface by scalar potential
    ForceVectors,    // Draw force arrows at sample points
    EnergyDensity,   // Color surface by energy density
    StressIso,       // Marching cubes on stress grid
    TempSlice,       // Slice plane color-mapped by temperature
    ChargeDensity,   // Color surface by charge density
    VelocityField,   // Trace velocity field lines
    PressureField,   // Color surface by pressure
};

// ============================================================================
// FieldSample — a single point sampled from a field
// ============================================================================
struct FieldSample {
    Vec3 position;    // World-space position
    Vec3 value;       // Vector field value (force, velocity, etc.)
    float scalar;     // Scalar field value (potential, pressure, etc.)
    float intensity;  // Normalized intensity for color mapping

    FieldSample() : scalar(0.0f), intensity(0.0f) {}
    FieldSample(const Vec3& p, const Vec3& v, float s)
        : position(p), value(v), scalar(s), intensity(s) {}
};

// ============================================================================
// FieldLine — a traced field line (sequence of points)
// ============================================================================
struct FieldLine {
    std::vector<Vec3> points;   // Points along the line
    Vec3 start;                 // Seed point
    Vec3 end;                   // Termination point
    float length;               // Total arc length
    float max_intensity;        // Max intensity along line

    FieldLine() : length(0.0f), max_intensity(0.0f) {}
};

// ============================================================================
// FieldVisualizer — WS4 field visualization renderer
// ============================================================================
// Renders physical fields in real time:
// - Field lines via RK4 integration (dP/ds = F(P)/|F(P)|)
// - Potential maps (color surface by scalar potential)
// - Force vectors (arrows at sample points)
// - Energy density (color surface by energy density)
// - Stress iso-surfaces (marching cubes on stress grid)
// - Temperature field slices (slice plane color-mapped by T)
//
// Usage:
//   FieldVisualizer viz(world);
//   viz.SetMode(FieldMode::FieldLines);
//   viz.SetSeedPoints(seeds);
//   viz.Render();
// ============================================================================
class FieldVisualizer {
public:
    explicit FieldVisualizer(PhysicsWorld* world);
    ~FieldVisualizer() = default;

    // Set the visualization mode
    void SetMode(FieldMode mode);

    // Get current mode
    FieldMode Mode() const { return mode_; }

    // Enable/disable visualization
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }

    // Set seed points for field line tracing
    void SetSeedPoints(const std::vector<Vec3>& seeds);

    // Set sampling parameters
    void SetSampleCount(int count) { sample_count_ = count; }
    void SetMaxTraceLength(float len) { max_trace_length_ = len; }
    void SetTraceStepSize(float step) { trace_step_ = step; }

    // Set slice plane for temperature/stress slices
    void SetSlicePlane(Vec3 normal, float distance);

    // Set color mapping range for scalar fields
    void SetColorRange(float min_val, float max_val);

    // Render the field visualization (SDL rendering calls)
    void Render();

    // Refresh field samples from current PhysicsWorld state
    void Refresh();

    // Clear cached field samples and lines
    void Clear();

    // Access traced field lines / samples (for testing/validation)
    const std::vector<FieldLine>& FieldLines() const { return field_lines_; }
    const std::vector<FieldSample>& Samples() const { return samples_; }

private:
    // Trace a single field line via RK4 integration
    FieldLine TraceFieldLine(const Vec3& seed, PhysicsWorld* world);

    // Sample scalar field at a point (potential, temperature, pressure)
    float SampleScalarField(const Vec3& pos, PhysicsWorld* world) const;

    // Sample vector field at a point (force, velocity)
    Vec3 SampleVectorField(const Vec3& pos, PhysicsWorld* world) const;

    // Color map scalar value to RGB (heat map)
    Vec3 ScalarToColor(float value, float min_val, float max_val) const;

    // Draw force arrows at sample points
    void DrawForceArrows();

    // Draw field lines
    void DrawFieldLines();

    // Color surface by scalar potential
    void ColorSurfaceByPotential();

    // Marching cubes on stress grid
    void MarchStressIso();

    // Slice plane color-mapped by temperature
    void DrawTempSlice();

    PhysicsWorld* world_;          // Physics world (read state)
    FieldMode mode_;               // Current visualization mode
    bool enabled_;                 // Whether visualization is active
    std::vector<Vec3> seed_points_; // Seed points for field line tracing
    std::vector<FieldLine> field_lines_; // Traced field lines
    std::vector<FieldSample> samples_;   // Sampled field values
    int sample_count_;             // Number of samples per surface
    float max_trace_length_;       // Max arc length for field line tracing
    float trace_step_;             // Step size for RK4 integration
    Vec3 slice_normal_;            // Slice plane normal
    float slice_distance_;         // Slice plane distance from origin
    float color_min_;              // Color mapping minimum
    float color_max_;              // Color mapping maximum
};

} // namespace DeliRium
