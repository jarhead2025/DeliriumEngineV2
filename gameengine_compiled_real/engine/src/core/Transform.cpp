// FILEPATH: engine/src/core/Transform.cpp
#include "delirium/core/Transform.h"
#include <cmath>

namespace DeliRium {

Vec3 Transform::Rotate(const Vec3& local) const {
    // q * v * q^-1 with q = (x,y,z,w). Standard, branch-free.
    const float x = rotation.x, y = rotation.y, z = rotation.z, w = rotation.w;
    const float tx = 2.f * (y * local.z - z * local.y);
    const float ty = 2.f * (z * local.x - x * local.z);
    const float tz = 2.f * (x * local.y - y * local.x);
    return {
        local.x + w * tx + (y * tz - z * ty),
        local.y + w * ty + (z * tx - x * tz),
        local.z + w * tz + (x * ty - y * tx)
    };
}

Vec3 Transform::InverseVector(const Vec3& v) const {
    // Inverse rotation = conjugate quaternion.
    const float x = -rotation.x, y = -rotation.y, z = -rotation.z, w = rotation.w;
    const float tx = 2.f * (y * v.z - z * v.y);
    const float ty = 2.f * (z * v.x - x * v.z);
    const float tz = 2.f * (x * v.y - y * v.x);
    return {
        v.x + w * tx + (y * tz - z * ty),
        v.y + w * ty + (z * tx - x * tz),
        v.z + w * tz + (x * ty - y * tx)
    };
}

Vec3 Transform::InversePoint(const Vec3& world) const {
    Vec3 local = InverseVector(world - position);
    const float sx = std::fabs(scale.x) > 1e-9f ? scale.x : 1e-9f;
    const float sy = std::fabs(scale.y) > 1e-9f ? scale.y : 1e-9f;
    const float sz = std::fabs(scale.z) > 1e-9f ? scale.z : 1e-9f;
    return { local.x / sx, local.y / sy, local.z / sz };
}

Mat4 Transform::ToMatrix() const {
    // R from the quaternion (column-major accumulation, then transposed into
    // row-major), scaled per axis, translated into the last column.
    const float x = rotation.x, y = rotation.y, z = rotation.z, w = rotation.w;
    Mat4 r;
    r.m[0] = 1.f - 2.f * (y*y + z*z);
    r.m[1] = 2.f * (x*y - z*w);
    r.m[2] = 2.f * (x*z + y*w);
    r.m[3] = 0.f;
    r.m[4] = 2.f * (x*y + z*w);
    r.m[5] = 1.f - 2.f * (x*x + z*z);
    r.m[6] = 2.f * (y*z - x*w);
    r.m[7] = 0.f;
    r.m[8] = 2.f * (x*z - y*w);
    r.m[9] = 2.f * (y*z + x*w);
    r.m[10] = 1.f - 2.f * (x*x + y*y);
    r.m[11] = 0.f;
    r.m[12] = 0.f; r.m[13] = 0.f; r.m[14] = 0.f; r.m[15] = 1.f;
    // Apply scale to the rotation columns.
    r.m[0] *= scale.x; r.m[1] *= scale.y; r.m[2] *= scale.z;
    r.m[4] *= scale.x; r.m[5] *= scale.y; r.m[6] *= scale.z;
    r.m[8] *= scale.x; r.m[9] *= scale.y; r.m[10] *= scale.z;
    // Translation into the LAST COLUMN (m[3], m[7], m[11]).
    //
    // WHY THE COLUMN AND NOT THE ROW: the matrix is row-major and is
    // consumed as M * (p,1) -- Mat4_TransformPoint (CPU) and the kernels'
    // xform_point (dot(m0, (p,1))) BOTH read the translation from the last
    // COLUMN. Writing it into the last ROW put the offset where a point
    // transform never looks, so the translation was silently DROPPED (and
    // the w-row was corrupted). It also made ToMatrix disagree with
    // InversePoint, which assumes M = T*R*S.
    r.m[3] = position.x; r.m[7] = position.y; r.m[11] = position.z;
    return r;
}

bool Transform::IsIdentity() const {
    return position.x == 0.f && position.y == 0.f && position.z == 0.f
        && rotation.x == 0.f && rotation.y == 0.f && rotation.z == 0.f && rotation.w == 1.f
        && scale.x == 1.f && scale.y == 1.f && scale.z == 1.f;
}

Mat4 Mat4_Multiply(const Mat4& a, const Mat4& b) {
    Mat4 out;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += a.m[i*4+k] * b.m[k*4+j];
            out.m[i*4+j] = s;
        }
    return out;
}

Vec3 Mat4_TransformPoint(const Mat4& m, const Vec3& p) {
    return {
        m.m[0]*p.x + m.m[1]*p.y + m.m[2]*p.z + m.m[3],
        m.m[4]*p.x + m.m[5]*p.y + m.m[6]*p.z + m.m[7],
        m.m[8]*p.x + m.m[9]*p.y + m.m[10]*p.z + m.m[11]
    };
}

Vec3 Mat4_TransformVector(const Mat4& m, const Vec3& v) {
    return {
        m.m[0]*v.x + m.m[1]*v.y + m.m[2]*v.z,
        m.m[4]*v.x + m.m[5]*v.y + m.m[6]*v.z,
        m.m[8]*v.x + m.m[9]*v.y + m.m[10]*v.z
    };
}

} // namespace DeliRium
