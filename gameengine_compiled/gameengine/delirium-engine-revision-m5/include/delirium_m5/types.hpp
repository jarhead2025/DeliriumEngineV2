#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace delirium::m5 {

inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kTwoPi = 2.0f * kPi;

using AnchorId = std::uint32_t;
using PatchId = std::uint32_t;
using MaterialId = std::uint32_t;
using Generation = std::uint64_t;
using TopologyGeneration = std::uint64_t;
using BodyId = std::uint32_t;

constexpr AnchorId kInvalidAnchor = UINT32_MAX;
constexpr PatchId kInvalidPatch = UINT32_MAX;
constexpr MaterialId kInvalidMaterial = UINT32_MAX;
constexpr BodyId kInvalidBody = UINT32_MAX;

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
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

struct PatchCoefficients {
    std::array<float, 8> scalars{};
    std::array<Vec3, 4> residuals{};
    bool has_residuals() const {
        for (const auto& r : residuals)
            if (length(r) > 1e-6f) return true;
        return false;
    }
};

enum class BasisType : std::uint8_t {
    Bilinear = 0, Plano = 1, Cupula = 2, Silla = 3,
    ValleU = 4, ValleV = 5, Eggbox = 6, Twist = 7,
};

struct WeldedSeam {
    PatchId patch_a = kInvalidPatch, patch_b = kInvalidPatch;
    float a_u0, a_v0, a_u1, a_v1;
    float b_u0, b_v0, b_u1, b_v1;
    float tolerance = 1e-4f;
    bool is_valid() const { return patch_a != kInvalidPatch && patch_b != kInvalidPatch; }
};

struct MaterialProperties {
    float youngs_modulus = 1.0e6f;
    float poisson_ratio = 0.45f;
    float density = 1000.0f;
    float yield_strength = 5.0e5f;
    float thermal_conductivity = 0.5f;
    float thermal_expansion = 1e-5f;
    std::uint8_t r = 200, g = 200, b = 200;
    float roughness = 0.5f, metalness = 0.0f;
    float shear_modulus() const { return youngs_modulus / (2.0f * (1.0f + poisson_ratio)); }
    float bulk_modulus() const { return youngs_modulus / (3.0f * (1.0f - 2.0f * poisson_ratio)); }
};

struct ModalMode {
    enum class Type : std::uint8_t {
        BendingX = 0, BendingY = 1, BendingZ = 2,
        TorsionX = 3, TorsionY = 4, TorsionZ = 5, Breathing = 6,
    };
    Type type = Type::Breathing;
    float amplitude = 0.0f, frequency = 1.0f, phase = 0.0f;
};

struct Surfel {
    Vec3 position{}, normal{};
    float radius = 0.0f;
    std::uint8_t r = 200, g = 200, b = 200, a = 255;
    float roughness = 0.5f, metalness = 0.0f;
    MaterialId material = kInvalidMaterial;
    PatchId source_patch = kInvalidPatch;
    BodyId source_body = kInvalidBody;
};

struct SpatialCell {
    std::uint32_t first = 0, count = 0, material_mask = 0;
    bool is_empty() const { return count == 0; }
};

// M5: contact pair between two surfels on different bodies
struct ContactPair {
    BodyId body_a = kInvalidBody, body_b = kInvalidBody;
    std::uint32_t surfel_a = 0, surfel_b = 0;
    Vec3 contact_point{};
    Vec3 normal{};
    float penetration = 0.0f;
};

// M5: animation keyframe for modal amplitude
struct Keyframe {
    float time = 0.0f;
    float amplitude = 0.0f;
};

} // namespace delirium::m5