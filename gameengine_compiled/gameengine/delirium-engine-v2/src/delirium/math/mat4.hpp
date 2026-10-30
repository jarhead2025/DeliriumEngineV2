#pragma once
#include "vec3.hpp"
#include <array>
#include <cstring>

namespace delirium {

/// 4x4 column-major matrix — transformations, camera, projection.
/// Column-major for OpenGL interoperability.
struct alignas(16) mat4 {
    /// 16 floats, column-major layout (shared with shaders and OpenCL)
    alignas(16) std::array<float, 16> data;

    constexpr mat4() : data{} { data.fill(0); }

    static constexpr mat4 identity() {
        mat4 m;
        m.data.fill(0);
        m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1.0f;
        return m;
    }

    /// Column access (column-major)
    constexpr float& operator()(size_t col, size_t row) {
        return data[col * 4 + row];
    }
    constexpr const float& operator()(size_t col, size_t row) const {
        return data[col * 4 + row];
    }

    /// Transform a vec3 (affine: m * [x,y,z,1])
    vec3 transform_point(const vec3& p) const {
        float w = (0)[3] * p.x + (1)[3] * p.y + (2)[3] * p.z + (3)[3];
        return vec3(
            ((0)[0] * p.x + (1)[0] * p.y + (2)[0] * p.z + (3)[0]) / w,
            ((0)[1] * p.x + (1)[1] * p.y + (2)[1] * p.z + (3)[1]) / w,
            ((0)[2] * p.x + (1)[2] * p.y + (2)[2] * p.z + (3)[2]) / w
        );
    }

    /// Transform a vec3 (direction: m * [x,y,z,0])
    vec3 transform_dir(const vec3& d) const {
        return vec3(
            (0)[0] * d.x + (1)[0] * d.y + (2)[0] * d.z,
            (0)[1] * d.x + (1)[1] * d.y + (2)[1] * d.z,
            (0)[2] * d.x + (1)[2] * d.y + (2)[2] * d.z
        );
    }

    /// Matrix multiplication: this * other
    mat4 operator*(const mat4& o) const {
        mat4 result;
        for (size_t c = 0; c < 4; ++c)
            for (size_t r = 0; r < 4; ++r) {
                float sum = 0;
                for (size_t k = 0; k < 4; ++k)
                    sum += data[k * 4 + r] * o.data[c * 4 + k];
                result.data[c * 4 + r] = sum;
            }
        return result;
    }

    /// Translation matrix
    static mat4 translate(const vec3& t) {
        mat4 m = identity();
        m.data[12] = t.x;
        m.data[13] = t.y;
        m.data[14] = t.z;
        return m;
    }

    /// Scale matrix
    static mat4 scale(const vec3& s) {
        mat4 m;
        m.data.fill(0);
        m.data[0] = s.x;
        m.data[5] = s.y;
        m.data[10] = s.z;
        m.data[15] = 1.0f;
        return m;
    }

    /// Perspective projection matrix
    static mat4 perspective(float fov_y, float aspect, float near, float far) {
        mat4 m;
        m.data.fill(0);
        float f = 1.0f / std::tan(fov_y * 0.5f);
        m.data[0] = f / aspect;
        m.data[5] = f;
        m.data[10] = (far + near) / (near - far);
        m.data[11] = -1.0f;
        m.data[14] = (2.0f * far * near) / (near - far);
        return m;
    }

    /// Look-at camera matrix
    static mat4 look_at(const vec3& eye, const vec3& target, const vec3& up) {
        vec3 fwd = (target - eye).normalized();
        vec3 right = fwd.cross(up).normalized();
        vec3 actual_up = right.cross(fwd);
        mat4 m = identity();
        m.data[0] = right.x;  m.data[1] = right.y;  m.data[2] = right.z;
        m.data[4] = actual_up.x;  m.data[5] = actual_up.y;  m.data[6] = actual_up.z;
        m.data[8] = -fwd.x;  m.data[9] = -fwd.y;  m.data[10] = -fwd.z;
        m.data[12] = -right.dot(eye);
        m.data[13] = -actual_up.dot(eye);
        m.data[14] = fwd.dot(eye);
        return m;
    }
};

} // namespace delirium