#pragma once
#include <cmath>
#include <cstring>

namespace delirium {

/// 3D vector — the fundamental spatial type.
/// All positions, directions, normals, and colors flow through this.
struct vec3 {
    float x, y, z;

    vec3() : x(0), y(0), z(0) {}
    vec3(float v) : x(v), y(v), z(v) {}
    vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    vec3 operator+(const vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    vec3 operator-(const vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    vec3 operator/(float s) const { float i = 1.0f/s; return {x*i, y*i, z*i}; }
    vec3& operator+=(const vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    vec3& operator-=(const vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
    vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
    bool operator==(const vec3& o) const { return x==o.x && y==o.y && z==o.z; }
    vec3 operator-() const { return {-x, -y, -z}; }

    float dot(const vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    vec3 cross(const vec3& o) const { return {y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x}; }
    float length_sq() const { return x*x + y*y + z*z; }
    float length() const { return sqrtf(length_sq()); }
    vec3 normalized() const { float l = length(); return l > 1e-8f ? *this/l : vec3(0); }
};

inline vec3 operator*(float s, const vec3& v) { return v*s; }

} // namespace delirium