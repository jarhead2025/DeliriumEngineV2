#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace delirium::m3 {

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
//
// The 8 scalar coefficients encode the bilinear base (4 corner heights) plus
// 4 higher-order modes. Vector residuals add sub-anchor geometric features
// (creases, dents, bumps) that the scalar basis cannot represent.
// ---------------------------------------------------------------------------
struct PatchCoefficients {
    // Scalar coefficients: [c00, c10, c01, c11, m1, m2, m3, m4]
    // c00-c11: corner heights at (0,0), (1,0), (0,1), (1,1)
    // m1-m4: higher-order mode amplitudes
    std::array<float, 8> scalars{};

    // Vector residuals: per-corner displacement vectors for sub-anchor detail
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
    Bilinear = 0,     // Simple bilinear interpolation (M2 baseline)
    Plano = 1,        // Flat plane with higher-order correction
    Cupula = 2,       // Dome/cap shape
    Silla = 3,        // Saddle shape
    ValleU = 4,       // Valley along u direction
    ValleV = 5,       // Valley along v direction
    Eggbox = 6,       // Egg-crate (double saddle)
    Twist = 7,        // Twisted surface
};

// ---------------------------------------------------------------------------
// Welded seam descriptor
// ---------------------------------------------------------------------------
//
// A welded seam connects two patches along a shared edge that does NOT share
// anchors (non-shared boundary). The seam enforces positional continuity
// within a configurable tolerance.
// ---------------------------------------------------------------------------
struct WeldedSeam {
    PatchId patch_a = kInvalidPatch;
    PatchId patch_b = kInvalidPatch;

    // Edge endpoints in parametric space: (u,v) on each patch
    float a_u0, a_v0, a_u1, a_v1;
    float b_u0, b_v0, b_u1, b_v1;

    // Maximum allowed positional discontinuity after stitching
    float tolerance = 1e-4f;

    bool is_valid() const {
        return patch_a != kInvalidPatch && patch_b != kInvalidPatch;
    }
};

// ---------------------------------------------------------------------------
// Surfel (render cache element)
// ---------------------------------------------------------------------------
struct Surfel {
    Vec3 position{};
    Vec3 normal{};
    float radius = 0.0f;

    std::uint8_t r = 200;
    std::uint8_t g = 200;
    std::uint8_t b = 200;
    std::uint8_t a = 255;

    MaterialId material = kInvalidMaterial;
    PatchId source_patch = kInvalidPatch;
};

// ---------------------------------------------------------------------------
// Spatial cell — multi-surfel container
// ---------------------------------------------------------------------------
//
// Replaces the legacy single-surfel-per-cell grid. Each cell can hold
// multiple surfels from different patches/materials. Material identity
// is preserved per-surfel.
// ---------------------------------------------------------------------------
struct SpatialCell {
    // Compact surfel indices (into the global surfel array)
    std::uint32_t first = 0;
    std::uint32_t count = 0;

    // Material mask: bit i set if surfels from material i are present
    std::uint32_t material_mask = 0;

    bool is_empty() const { return count == 0; }
};

} // namespace delirium::m3