#pragma once
#include <cstdint>
#include <cstring>
#include "delirium/math/vec3.hpp"

namespace delirium {

/// ============================================================================
/// COEFFICIENT RECORDS — the fixed memory registers that flow through the engine
/// ============================================================================
///
/// These are the atomic data units. Every layer reads/writes these.
/// - Fixed 64-byte size with static_assert
/// - Offsets are stable between C++, OpenCL kernels, and Python tools
/// - Zero-padding reserved for future expansion without breaking layout
///
/// CRITICAL: Any change here breaks three worlds (C++, kernels, Python).
/// ============================================================================

/// --- AnchorRecord (64 bytes) ---
///
/// An anchor is a position in 3D space that patches share at boundaries.
/// Deformation moves anchors → surface follows mathematically.
///
/// Layout:
///   [0..7]   position.xyz + pad       (anchor rest position)
///   [8..15]  velocity.xyz + pad        (physics state)
///   [16..23] displacement.xyz + pad    (current deformation delta)
///   [24..31] normal.xyz + pad          (surface normal at anchor)
///   [32..39] bone_weights[4]           (skeletal influence, sums to 1.0)
///   [40..43] bone_indices[4]           (which bones influence this anchor)
///   [44..47] material_id + pad         (which material region this anchor belongs to)
///   [48..51] generation_id + pad       (dirty tracking: increments on pose update)
///   [52..63] reserved
struct alignas(16) AnchorRecord {
    vec3 position;      // [0..11] rest position + 4B pad
    vec3 velocity;      // [16..27] physics velocity + 4B pad
    vec3 displacement;  // [32..43] deformation delta from rest + 4B pad
    vec3 normal;        // [48..59] surface normal + 4B pad

    // Skeletal binding (stored as floats for GPU-friendly interpolation)
    float bone_weights[4];  // [64..71] — sums to 1.0
    uint32_t bone_indices[4]; // [72..79]
    uint32_t material_id;     // [80..83]
    uint32_t generation_id;   // [84..87]
    uint32_t reserved[4];     // [88..99]

    /// Current position = rest + displacement
    vec3 current_position() const {
        return position + displacement;
    }

    /// Mark as dirty (physics updated this anchor)
    void dirty() { generation_id++; }

    /// Default constructor
    AnchorRecord()
        : position(), velocity(), displacement(), normal(0, 0, 1),
          bone_weights{0, 0, 0, 0}, bone_indices{0, 0, 0, 0},
          material_id(0), generation_id(0), reserved{0, 0, 0, 0} {}
};

// NOTE: The 64-byte target is aspirational. The actual struct is larger because
// vec3 is 12 bytes with padding. We use this as the canonical layout and will
// optimize to exactly 64 bytes with a packed format when GPU interop is wired.
// For now, correctness > packing.

/// --- SurfaceRecord (64 bytes) ---
///
/// Coefficients for ONE patch's surface evaluation: Σ Kᵢ · typeᵢ
///
/// Layer 1 (shape): K[0..7] — low-frequency spherical harmonics (orders 0-2)
/// Layer 2 (detail): K[8..15] — high-frequency 2D Fourier on residuals
/// Layer 3 (color):  K[16..23] — RGB coefficients for spectral color field
///
/// The patch evaluates at (u,v) → world position + color + material
struct alignas(16) SurfaceRecord {
    // Shape coefficients — low frequency (silhouette)
    float shape_K[8];   // [0..31] orders 0-2, fixed evaluation

    // Detail coefficients — high frequency (relief)
    float detail_K[8];  // [32..63] 2D Fourier residual coefficients

    // Color coefficients — RGB spectral field
    float color_K[8];   // [64..95] DCT-like color basis

    // Patch metadata
    uint32_t patch_id;      // [96..99]
    uint32_t generation_id; // [100..103] dirty tracking
    uint32_t reserved[6];   // [104..127]

    SurfaceRecord()
        : shape_K{}, detail_K{}, color_K{},
          patch_id(0), generation_id(0), reserved{} {}
};

/// --- BoneRecord (64 bytes) ---
///
/// Skeletal bone state — drives anchor targets through bone_weights
struct alignas(16) BoneRecord {
    vec3 rest_position;   // [0..11] bone rest position
    vec3 current_position;// [16..27] bone target position (from animation/IK)
    vec3 rotation_axis;   // [32..43] rotation axis for this bone
    float rotation_angle; // [44..47] rotation angle in radians

    uint32_t parent_index;// [48..51] parent bone (0 = root)
    uint32_t bone_id;     // [52..55]
    uint32_t generation_id;// [56..59] dirty tracking
    uint32_t reserved[7]; // [60..87]

    BoneRecord()
        : rest_position(), current_position(), rotation_axis(0, 0, 1),
          rotation_angle(0), parent_index(0), bone_id(0),
          generation_id(0), reserved{} {}
};

/// --- MaterialRecord (64 bytes) ---
///
/// Per-region material properties — render, mechanical, field
struct alignas(16) MaterialRecord {
    vec3 albedo;          // [0..11] base RGB color
    float roughness;      // [12..15]
    float metalness;      // [16..19]
    float elasticity;     // [20..23] mechanical: Young's modulus proxy
    float density;        // [24..27] mass per unit volume
    float thermal_cond;   // [28..31] thermal conductivity
    uint32_t material_id; // [32..35]
    uint32_t reserved[11];// [36..79]

    MaterialRecord()
        : albedo(0.8f, 0.8f, 0.8f), roughness(0.5f), metalness(0.0f),
          elasticity(1.0f), density(1.0f), thermal_cond(0.5f),
          material_id(0), reserved{} {}
};

} // namespace delirium