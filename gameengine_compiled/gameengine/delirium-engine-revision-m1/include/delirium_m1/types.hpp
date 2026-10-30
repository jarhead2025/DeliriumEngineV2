#pragma once

#include <cmath>
#include <cstdint>

namespace delirium::m1 {

using AnchorId = std::uint32_t;
using PatchId = std::uint32_t;
using BoneId = std::uint32_t;
using MaterialId = std::uint32_t;
using Generation = std::uint64_t;

constexpr AnchorId kInvalidAnchor = UINT32_MAX;
constexpr PatchId kInvalidPatch = UINT32_MAX;
constexpr BoneId kInvalidBone = UINT32_MAX;
constexpr MaterialId kInvalidMaterial = UINT32_MAX;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3& operator+=(const Vec3& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float scalar) { return {a.x * scalar, a.y * scalar, a.z * scalar}; }
inline Vec3 operator*(float scalar, Vec3 a) { return a * scalar; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float length(const Vec3& value) { return std::sqrt(dot(value, value)); }
inline bool nearly_equal(const Vec3& a, const Vec3& b, float epsilon = 1e-5f) {
    return length(a - b) <= epsilon;
}

struct Transform {
    // M1 intentionally supports translation only. Rotation/scale and full
    // skinning transforms belong to M2 once the target-pose/spring contract
    // has been validated end-to-end.
    Vec3 translation{};
};

struct PatchCacheRange {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
};

} // namespace delirium::m1
