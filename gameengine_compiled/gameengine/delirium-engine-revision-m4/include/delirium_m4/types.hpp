#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace delirium::m4 {

// C++17-compatible constants
inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kTwoPi = 2.0f * kPi;

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------
using AnchorId = std::uint32_t;
using PatchId = std::uint32_t;
using MaterialId = std::uint32_t;
using Generation = std::uint64_t;
using TopologyGeneration = std::uint64_t;

constexpr AnchorId kInvalidAnchor = UINT32_MAX;
constexpr PatchId kInvalidPatch = UINT32_MAX;
constexpr MaterialId kInvalidMaterial = UINT32_MAX;

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    Vec3& operator/=(float s) { float inv = 1.0f / s; x *= inv; y *= inv; z *= inv; return *this; }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
inline Vec3 operator*(Vec3 a, float s) { return a *= s; }
inline Vec3 operator*(float s, Vec3 a) { return a *= s; }
inline Vec3 operator/(Vec3 a, float s) { return a /= s; }
inline Vec3 operator-(Vec3 a) { return a *= -1.0f; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize_vec3(const Vec3& v) {
    float l = length(v);
    return l > 1e-8f ? v * (1.0f / l) : Vec3{1.0f, 0.0f, 0.0f};
}
inline bool nearly_equal(const Vec3& a, const Vec3& b, float eps = 1e-5f) {
    return length(a - b) <= eps;
}

// ---------------------------------------------------------------------------
// Patch coefficients — 8 scalar + optional vector residuals
// ---------------------------------------------------------------------------
struct PatchCoefficients {
    std::array<float, 8> scalars{};
    std::array<Vec3, 4> residuals{};

    bool has_residuals() const {
        for (const auto& r : residuals) {
            if (length(r) > 1e-6f) return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Basis function type
// ---------------------------------------------------------------------------
enum class BasisType : std::uint8_t {
    Bilinear = 0,
    Plano = 1,
    Cupula = 2,
    Silla = 3,
    ValleU = 4,
    ValleV = 5,
    Eggbox = 6,
    Twist = 7,
};

// ---------------------------------------------------------------------------
// Welded seam descriptor
// ---------------------------------------------------------------------------
struct WeldedSeam {
    PatchId patch_a = kInvalidPatch;
    PatchId patch_b = kInvalidPatch;

    float a_u0, a_v0, a_u1, a_v1;
    float b_u0, b_v0, b_u1, b_v1;

    float tolerance = 1e-4f;

    bool is_valid() const {
        return patch_a != kInvalidPatch && patch_b != kInvalidPatch;
    }
};

// ---------------------------------------------------------------------------
// MaterialProperties — M4: per-region mechanical behavior
// ---------------------------------------------------------------------------
//
// Maps a MaterialId to physical properties used by the elastic solver and
// damage accumulator. Also defines rendering appearance (color, roughness,
// metalness) so physics and rendering stay consistent.
// ---------------------------------------------------------------------------
struct MaterialProperties {
    // Mechanical properties
    float youngs_modulus = 1.0e6f;   // Pa (default: soft rubber)
    float poisson_ratio = 0.45f;     // dimensionless
    float density = 1000.0f;         // kg/m^3
    float yield_strength = 5.0e5f;   // Pa — stress threshold for plastic deformation

    // Rendering properties
    std::uint8_t r = 200;
    std::uint8_t g = 200;
    std::uint8_t b = 200;
    float roughness = 0.5f;
    float metalness = 0.0f;

    // Derived: shear modulus from Young's modulus and Poisson ratio
    float shear_modulus() const {
        return youngs_modulus / (2.0f * (1.0f + poisson_ratio));
    }

    // Derived: bulk modulus
    float bulk_modulus() const {
        return youngs_modulus / (3.0f * (1.0f - 2.0f * poisson_ratio));
    }
};

// ---------------------------------------------------------------------------
// ModalMode — M4: global elastic displacement mode
// ---------------------------------------------------------------------------
//
// Each modal mode represents a low-frequency deformation pattern (bending,
// torsion, breathing) that displaces all anchors coherently. Modes compose
// before the local material solve.
// ---------------------------------------------------------------------------
struct ModalMode {
    enum class Type : std::uint8_t {
        BendingX = 0,   // Bending along X axis
        BendingY = 1,   // Bending along Y axis
        BendingZ = 2,   // Bending along Z axis
        TorsionX = 3,   // Torsion around X axis
        TorsionY = 4,   // Torsion around Y axis
        TorsionZ = 5,   // Torsion around Z axis
        Breathing = 6,  // Uniform expansion/contraction
    };

    Type type = Type::Breathing;
    float amplitude = 0.0f;
    float frequency = 1.0f;
    float phase = 0.0f;

    // Per-anchor displacement weight (computed at registration time)
    // Stored as a displacement direction; magnitude is scaled by amplitude
    std::array<Vec3, 1> axis{};  // primary axis for the mode
};

// ---------------------------------------------------------------------------
// Surfel (render cache element) — M4: material-driven colors
// ---------------------------------------------------------------------------
struct Surfel {
    Vec3 position{};
    Vec3 normal{};
    float radius = 0.0f;

    std::uint8_t r = 200;
    std::uint8_t g = 200;
    std::uint8_t b = 200;
    std::uint8_t a = 255;

    float roughness = 0.5f;
    float metalness = 0.0f;

    MaterialId material = kInvalidMaterial;
    PatchId source_patch = kInvalidPatch;
};

// ---------------------------------------------------------------------------
// Spatial cell — multi-surfel container
// ---------------------------------------------------------------------------
struct SpatialCell {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
    std::uint32_t material_mask = 0;

    bool is_empty() const { return count == 0; }
};

} // namespace delirium::m4