#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace delirium::m2 {

// C++17-compatible pi constant (std::numbers is C++20)
inline constexpr float kPi = 3.14159265358979323846f;

using AnchorId = std::uint32_t;
using PatchId = std::uint32_t;
using BoneId = std::uint32_t;
using JointId = std::uint32_t;
using MaterialId = std::uint32_t;
using Generation = std::uint64_t;
using TopologyGeneration = std::uint64_t;

constexpr AnchorId kInvalidAnchor = UINT32_MAX;
constexpr PatchId kInvalidPatch = UINT32_MAX;
constexpr BoneId kInvalidBone = UINT32_MAX;
constexpr JointId kInvalidJoint = UINT32_MAX;
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
inline Vec3 normalize(const Vec3& v) {
    float l = length(v);
    return l > 1e-8f ? v * (1.0f / l) : Vec3{1.0f, 0.0f, 0.0f};
}
inline bool nearly_equal(const Vec3& a, const Vec3& b, float eps = 1e-5f) {
    return length(a - b) <= eps;
}

// ---------------------------------------------------------------------------
// Quaternion  (w, x, y, z) — unit quaternions only
// ---------------------------------------------------------------------------
struct Quat {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    static Quat identity() { return {1.0f, 0.0f, 0.0f, 0.0f}; }

    static Quat from_angle_axis(float angle, const Vec3& axis);
    static Quat from_rotation_matrix(const float m[3][3]);

    Quat& normalize();
    Quat normalized() const { Quat q = *this; return q.normalize(); }

    Quat conjugate() const { return {w, -x, -y, -z}; }
    Quat operator*(const Quat& other) const;
};

inline Vec3 quat_rotate(const Quat& q, const Vec3& v) {
    Quat qv{0.0f, v.x, v.y, v.z};
    Quat result = q * qv * q.conjugate();
    return {result.x, result.y, result.z};
}

// ---------------------------------------------------------------------------
// Transform — translation + orientation
// ---------------------------------------------------------------------------
struct Transform {
    Vec3 translation{};
    Quat rotation{};

    bool is_identity() const {
        return nearly_equal(translation, {}) &&
               (std::abs(rotation.w - 1.0f) < 1e-5f &&
                std::abs(rotation.x) < 1e-5f &&
                std::abs(rotation.y) < 1e-5f &&
                std::abs(rotation.z) < 1e-5f);
    }

    // Apply this transform to a point (translation + rotation)
    Vec3 apply_to_point(const Vec3& p) const {
        return quat_rotate(rotation, p) + translation;
    }

    // Apply this transform to a direction (rotation only, no translation)
    Vec3 apply_to_direction(const Vec3& d) const {
        return quat_rotate(rotation, d);
    }
};

inline Transform operator*(const Transform& a, const Transform& b) {
    return {
        a.translation + quat_rotate(a.rotation, b.translation),
        a.rotation * b.rotation
    };
}

// ---------------------------------------------------------------------------
// Joint limits
// ---------------------------------------------------------------------------
struct JointLimits {
    // Angular limits in radians (per axis in local space)
    float min_angle_x = -kPi;
    float max_angle_x = kPi;
    float min_angle_y = -kPi;
    float max_angle_y = kPi;
    float min_angle_z = -kPi;
    float max_angle_z = kPi;

    // Translation limits
    Vec3 min_translation = {-1e6f, -1e6f, -1e6f};
    Vec3 max_translation = {1e6f, 1e6f, 1e6f};

    bool is_unbounded() const {
        return min_translation.x < -1e5f; // sentinel
    }
};

// ---------------------------------------------------------------------------
// Spring / damper parameters
// ---------------------------------------------------------------------------
struct SpringParams {
    float stiffness = 120.0f;
    float damping = 18.0f;
    float mass = 1.0f;
};

struct AngularSpringParams {
    float stiffness = 200.0f;
    float damping = 25.0f;
    float inertia = 1.0f;
};

// ---------------------------------------------------------------------------
// Patch cache range
// ---------------------------------------------------------------------------
struct PatchCacheRange {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
};

// ---------------------------------------------------------------------------
// Optional pose target (for IK / gameplay controllers)
// ---------------------------------------------------------------------------
struct PoseTarget {
    Transform transform{};
    float weight = 1.0f;       // blending weight [0, 1]
    bool active = false;
};

} // namespace delirium::m2