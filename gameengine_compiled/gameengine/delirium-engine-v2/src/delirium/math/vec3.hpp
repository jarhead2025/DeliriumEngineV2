#pragma once
#include <cmath>
#include <array>

namespace delirium {

/// Fixed 3-component vector — the atomic unit of all surface evaluation.
/// Every anchor position, normal, coefficient delta, and light bounce is a vec3.
struct alignas(16) vec3 {
    float x, y, z;

    constexpr vec3() : x(0), y(0), z(0) {}
    constexpr vec3(float v) : x(v), y(v), z(v) {}
    constexpr vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    // Arithmetic — all element-wise, no allocations
    constexpr vec3 operator+() const { return *this; }
    constexpr vec3 operator-() const { return vec3(-x, -y, -z); }

    constexpr vec3 operator+(const vec3& o) const { return vec3(x + o.x, y + o.y, z + o.z); }
    constexpr vec3 operator-(const vec3& o) const { return vec3(x - o.x, y - o.y, z - o.z); }
    constexpr vec3 operator*(float s) const { return vec3(x * s, y * s, z * s); }
    constexpr vec3 operator/(float s) const { float r = 1.0f / s; return vec3(x * r, y * r, z * r); }

    constexpr vec3& operator+=(const vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    constexpr vec3& operator-=(const vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    constexpr vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    // Dot / cross — used by lighting bounce and normal derivation
    constexpr float dot(const vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    constexpr vec3 cross(const vec3& o) const {
        return vec3(y * o.z - z * o.y,
                    z * o.x - x * o.z,
                    x * o.y - y * o.x);
    }

    constexpr float length_sq() const { return x * x + y * y + z * z; }
    constexpr float length() const { return std::sqrt(length_sq()); }

    vec3 normalized() const {
        float l = length();
        return l > 0.0f ? *this / l : vec3(0, 0, 0);
    }

    constexpr bool operator==(const vec3& o) const { return x == o.x && y == o.y && z == o.z; }
    constexpr bool operator!=(const vec3& o) const { return !(*this == o); }

    // Array access for GPU-interoperable memory layout
    constexpr float& operator[](size_t i) {
        return reinterpret_cast<float*>(&this->x)[i];
    }
    constexpr const float& operator[](size_t i) const {
        return reinterpret_cast<const float*>(&this->x)[i];
    }
};

// Scalar multiplication commutativity
constexpr vec3 operator*(float s, const vec3& v) { return v * s; }

} // namespace delirium