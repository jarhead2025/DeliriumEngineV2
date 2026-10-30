#include "delirium_m2/types.hpp"

#include <cmath>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// Quat implementation
// ---------------------------------------------------------------------------

Quat Quat::from_angle_axis(float angle, const Vec3& axis) {
    Vec3 a;
    float l = length(axis);
    if (l > 1e-8f) {
        float inv = 1.0f / l;
        a = axis * inv;
    } else {
        a = Vec3{1.0f, 0.0f, 0.0f};
    }
    float half = angle * 0.5f;
    float s = std::sin(half);
    return {std::cos(half), a.x * s, a.y * s, a.z * s};
}

Quat Quat::from_rotation_matrix(const float m[3][3]) {
    // Standard trace method
    float trace = m[0][0] + m[1][1] + m[2][2];
    Quat q;

    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m[2][1] - m[1][2]) / s;
        q.y = (m[0][2] - m[2][0]) / s;
        q.z = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        float s = std::sqrt(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f;
        q.w = (m[2][1] - m[1][2]) / s;
        q.x = 0.25f * s;
        q.y = (m[0][1] + m[1][0]) / s;
        q.z = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        float s = std::sqrt(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f;
        q.w = (m[0][2] - m[2][0]) / s;
        q.x = (m[0][1] + m[1][0]) / s;
        q.y = 0.25f * s;
        q.z = (m[1][2] + m[2][1]) / s;
    } else {
        float s = std::sqrt(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f;
        q.w = (m[1][0] - m[0][1]) / s;
        q.x = (m[0][2] + m[2][0]) / s;
        q.y = (m[1][2] + m[2][1]) / s;
        q.z = 0.25f * s;
    }
    return q.normalize();
}

Quat& Quat::normalize() {
    float l = std::sqrt(w*w + x*x + y*y + z*z);
    if (l > 1e-8f) {
        float inv = 1.0f / l;
        w *= inv; x *= inv; y *= inv; z *= inv;
    }
    return *this;
}

Quat Quat::operator*(const Quat& other) const {
    return {
        w * other.w - x * other.x - y * other.y - z * other.z,
        w * other.x + x * other.w + y * other.z - z * other.y,
        w * other.y - x * other.z + y * other.w + z * other.x,
        w * other.z + x * other.y - y * other.x + z * other.w
    };
}

} // namespace delirium::m2