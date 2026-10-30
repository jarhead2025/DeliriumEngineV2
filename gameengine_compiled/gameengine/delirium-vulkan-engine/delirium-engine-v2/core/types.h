#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace delirium {

inline constexpr float kPi    = 3.14159265358979323846f;
inline constexpr float kTwoPi = 2.0f * kPi;

/* ── ID types ─────────────────────────────────────────────────────── */
using AnchorId         = std::uint32_t;
using PatchId          = std::uint32_t;
using MaterialId       = std::uint32_t;
using Generation       = std::uint64_t;
using TopologyGeneration = std::uint64_t;
using BodyId           = std::uint32_t;

constexpr AnchorId         kInvalidAnchor = UINT32_MAX;
constexpr PatchId          kInvalidPatch  = UINT32_MAX;
constexpr MaterialId       kInvalidMaterial = UINT32_MAX;
constexpr BodyId           kInvalidBody   = UINT32_MAX;

/* ── Vec3 (M8-style: operators + free functions) ─────────────────── */
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s)       { x *= s;   y *= s;   z *= s;   return *this; }
    Vec3& operator/=(float s)       { float inv = 1.0f / s; x *= inv; y *= inv; z *= inv; return *this; }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
inline Vec3 operator*(Vec3 a, float s)       { return a *= s; }
inline Vec3 operator*(float s, Vec3 a)       { return a *= s; }
inline Vec3 operator/(Vec3 a, float s)       { return a /= s; }
inline Vec3 operator-(Vec3 a)                { return a *= -1.0f; }

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize_vec3(const Vec3& v) {
    float l = length(v);
    return l > 1e-8f ? v * (1.0f / l) : Vec3{1.0f, 0.0f, 0.0f};
}
inline bool nearly_equal(const Vec3& a, const Vec3& b, float eps = 1e-5f) {
    return length(a - b) <= eps;
}

/* ── Patch coefficients (8 scalar + 4 Vec3 residuals) ────────────── */
struct PatchCoefficients {
    std::array<float, 8>  scalars{};
    std::array<Vec3, 4>   residuals{};
    bool has_residuals() const {
        for (const auto& r : residuals)
            if (length(r) > 1e-6f) return true;
        return false;
    }
};

/* ── Basis type enum ─────────────────────────────────────────────── */
enum class BasisType : std::uint8_t {
    Bilinear = 0, Plano = 1, Cupula = 2, Silla = 3,
    ValleU = 4, ValleV = 5, Eggbox = 6, Twist = 7,
};

/* ── Material ────────────────────────────────────────────────────── */
struct MaterialProperties {
    float   youngs_modulus    = 1.0e6f;
    float   poisson_ratio     = 0.45f;
    float   density           = 1000.0f;
    float   yield_strength    = 5.0e5f;
    float   thermal_conductivity = 0.5f;
    float   thermal_expansion  = 1e-5f;
    std::uint8_t r = 200, g = 200, b = 200;
    float   roughness = 0.5f, metalness = 0.0f;
    float shear_modulus() const { return youngs_modulus / (2.0f * (1.0f + poisson_ratio)); }
    float bulk_modulus()  const { return youngs_modulus / (3.0f * (1.0f - 2.0f * poisson_ratio)); }
};

/* ── Modal mode ──────────────────────────────────────────────────── */
struct ModalMode {
    enum class Type : std::uint8_t {
        BendingX = 0, BendingY = 1, BendingZ = 2,
        TorsionX = 3, TorsionY = 4, TorsionZ = 5, Breathing = 6,
    };
    Type  type      = Type::Breathing;
    float amplitude = 0.0f, frequency = 1.0f, phase = 0.0f;
};

/* ── Surfel (render primitive) ───────────────────────────────────── */
struct Surfel {
    Vec3     position{}, normal{};
    float    radius = 0.0f;
    std::uint8_t r = 200, g = 200, b = 200, a = 255;
    float    roughness = 0.5f, metalness = 0.0f;
    MaterialId material     = kInvalidMaterial;
    PatchId  source_patch   = kInvalidPatch;
    BodyId   source_body    = kInvalidBody;
};

/* ── 64-byte bus-transfer records (frozen layout) ──────────────────
 *
 * These are the canonical on-disk / GPU-transfer records.  Each is
 * exactly 64 bytes — one cache line — so a contiguous blob is optimal
 * for DMA and coherent-shareable transfers.  The CPU runtime types
 * (AnchorState, PatchState) are separate; the I/O layer converts
 * between the two representations.
 *
 * Layout mirrors the original engine's Types.h and the GPU kernel
 * (kernels/psf_splat.cl).  #pragma pack(1) forces tight packing.
 * ───────────────────────────────────────────────────────────────── */
#pragma pack(push, 1)

struct AnchorRecord {
    std::uint32_t bone_index;       // 4  bone this anchor is bound to
    std::uint16_t connectivity[4];  // 8  neighbour anchor IDs
    std::uint8_t  curve_params[16]; // 16 stitch coefficients
    std::uint16_t pos[3];           // 6  quantised position (bbox-normalised)
    std::uint8_t  normal[3];        // 3  quantised normal
    int16_t       delta[3];         // 6  displacement delta
    std::uint8_t  color[3];         // 3  base colour
    std::uint8_t  roughness;        // 1
    std::uint8_t  metallic;         // 1
    float         anim_weight[4];   // 16 skeletal blend weights
}; // 64 bytes total

struct SurfaceRecord {
    std::uint16_t anchor_idx[4];    // 8  corner anchors
    std::uint16_t surface_def[2];   // 4  {basis_type, flags}
    float         coef[8];          // 32 corner-vanishing deformation basis
    std::uint8_t  color_pattern[18];// 18 spectral colour coefficients
    std::uint8_t  pad[2];           // 2  alignment
}; // 64 bytes total

#pragma pack(pop)

static_assert(sizeof(AnchorRecord)  == 64, "AnchorRecord must be 64 bytes");
static_assert(sizeof(SurfaceRecord) == 64, "SurfaceRecord must be 64 bytes");

/* ── Scene / interaction types ───────────────────────────────────── */
struct Frustum {
    Vec3  origin{}, forward{}, right{}, up{};
    float near_plane = 0.1f, far_plane = 100.0f;
    float fov_y = 1.0f, aspect = 1.33f;
};

struct PickResult {
    BodyId body_id  = kInvalidBody;
    PatchId patch_id = kInvalidPatch;
    float u = 0.0f, v = 0.0f;
    float distance = 1e30f;
    Vec3 hit_point{}, normal{};
};

struct ExternalForce {
    Vec3 gravity{};
    Vec3 wind{};
    float drag = 0.0f;
};

} // namespace delirium