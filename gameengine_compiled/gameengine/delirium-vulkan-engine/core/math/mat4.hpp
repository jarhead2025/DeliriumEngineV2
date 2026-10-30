#pragma once
#include <array>
#include <cmath>

namespace delirium {

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const {
        float l = length();
        return l > 1e-8f ? *this * (1.0f / l) : Vec3{0, 0, 1};
    }
};

// Free functions BEFORE Mat4 so look_at() can call them
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float length(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Column-major Mat4: element(row, col) is stored at m[col*4 + row]
// This matches OpenGL's native format — glUniformMatrix4fv(..., GL_FALSE, ...) works directly.
struct Mat4 {
    std::array<float, 16> m{};

    static Mat4 identity() {
        Mat4 result;
        result.m[0] = result.m[5] = result.m[10] = result.m[15] = 1.0f;
        return result;
    }

    static Mat4 perspective(float fov_y, float aspect, float near, float far) {
        Mat4 result;
        float tan_half = std::tan(fov_y / 2.0f);
        // Column-major OpenGL perspective matrix
        // element(row, col) stored at m[col*4 + row]
        result.m[0] = 1.0f / (aspect * tan_half);    // element(0,0)
        result.m[5] = 1.0f / tan_half;                // element(1,1)
        result.m[10] = (far + near) / (near - far);   // element(2,2)
        result.m[11] = -1.0f;                          // element(3,2) — w-row, z-column
        result.m[14] = (2.0f * far * near) / (near - far); // element(2,3) — z-row, w-column
        return result;
    }

    static Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
        Vec3 z = (eye - target).normalized();
        Vec3 x = cross(up, z).normalized();
        Vec3 y = cross(z, x).normalized();

        Mat4 result;
        // Column-major view matrix
        // Column 0 = x axis
        result.m[0] = x.x; result.m[1] = x.y; result.m[2] = x.z; result.m[3] = 0.0f;
        // Column 1 = y axis
        result.m[4] = y.x; result.m[5] = y.y; result.m[6] = y.z; result.m[7] = 0.0f;
        // Column 2 = z axis
        result.m[8] = z.x; result.m[9] = z.y; result.m[10] = z.z; result.m[11] = 0.0f;
        // Column 3 = translation
        result.m[12] = -dot(x, eye);
        result.m[13] = -dot(y, eye);
        result.m[14] = -dot(z, eye);
        result.m[15] = 1.0f;
        return result;
    }

    static Mat4 translate(Vec3 v) {
        Mat4 result = identity();
        result.m[12] = v.x;
        result.m[13] = v.y;
        result.m[14] = v.z;
        return result;
    }

    static Mat4 rotate_y(float angle) {
        Mat4 result = identity();
        float c = std::cos(angle), s = std::sin(angle);
        // Column-major rotation around Y axis
        result.m[0] = c;  result.m[2] = s;  // column 0: (0,0)=c, (2,0)=s
        result.m[8] = -s; result.m[10] = c; // column 2: (0,2)=-s, (2,2)=c
        return result;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 result;
        // Column-major multiply: element(i,j) stored at m[j*4 + i]
        // C[i][j] = sum_k( A[i][k] * B[k][j] )
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                result.m[j * 4 + i] = 0;
                for (int k = 0; k < 4; ++k)
                    result.m[j * 4 + i] += m[k * 4 + i] * o.m[j * 4 + k];
            }
        return result;
    }

    // Invert this matrix (Gauss-Jordan elimination). Returns identity if singular.
    Mat4 inverse() const {
        Mat4 result = identity();
        // Augmented matrix [M | I] stored as 4x8 column-major
        float a[4][8];
        for (int i = 0; i < 4; ++i) {
            a[i][0] = m[i];     a[i][1] = m[4 + i];   a[i][2] = m[8 + i];   a[i][3] = m[12 + i];
            a[i][4] = (i == 0); a[i][5] = (i == 1);    a[i][6] = (i == 2);    a[i][7] = (i == 3);
        }
        // Forward elimination with partial pivoting
        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 4; ++row)
                if (std::fabs(a[row][col]) > std::fabs(a[pivot][col])) pivot = row;
            std::swap(a[col], a[pivot]);
            if (std::fabs(a[col][col]) < 1e-12f) return identity(); // singular
            float inv = 1.0f / a[col][col];
            for (int j = 0; j < 8; ++j) a[col][j] *= inv;
            for (int row = 0; row < 4; ++row) {
                if (row == col) continue;
                float f = a[row][col];
                for (int j = 0; j < 8; ++j) a[row][j] -= f * a[col][j];
            }
        }
        // Extract inverse from right half
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.m[j * 4 + i] = a[i][4 + j];
        return result;
    }
};

} // namespace delirium