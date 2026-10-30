#pragma once
#include "vec3.h"
#include <cstring>

namespace delirium {

/// 4x4 column-major matrix for transforms, projection, view.
struct mat4 {
    float m[4][4];

    mat4() { memset(m, 0, sizeof(m)); }

    static mat4 identity() {
        mat4 r;
        r.m[0][0]=r.m[1][1]=r.m[2][2]=r.m[3][3]=1;
        return r;
    }

    static mat4 translation(const vec3& t) {
        mat4 r = identity();
        r.m[0][3]=t.x; r.m[1][3]=t.y; r.m[2][3]=t.z;
        return r;
    }

    static mat4 scale(float s) {
        mat4 r; r.m[0][0]=r.m[1][1]=r.m[2][2]=s; r.m[3][3]=1; return r;
    }
    static mat4 scale(const vec3& s) {
        mat4 r; r.m[0][0]=s.x; r.m[1][1]=s.y; r.m[2][2]=s.z; r.m[3][3]=1; return r;
    }

    static mat4 rotationX(float a) {
        mat4 r = identity(); float c=cosf(a), s=sinf(a);
        r.m[1][1]=c; r.m[1][2]=-s; r.m[2][1]=s; r.m[2][2]=c; return r;
    }
    static mat4 rotationY(float a) {
        mat4 r = identity(); float c=cosf(a), s=sinf(a);
        r.m[0][0]=c; r.m[0][2]=s; r.m[2][0]=-s; r.m[2][2]=c; return r;
    }
    static mat4 rotationZ(float a) {
        mat4 r = identity(); float c=cosf(a), s=sinf(a);
        r.m[0][0]=c; r.m[0][1]=-s; r.m[1][0]=s; r.m[1][1]=c; return r;
    }

    mat4 operator*(const mat4& o) const {
        mat4 r;
        for(int i=0;i<4;i++) for(int j=0;j<4;j++)
            for(int k=0;k<4;k++) r.m[i][j] += m[i][k]*o.m[k][j];
        return r;
    }

    vec3 transformPoint(const vec3& p) const {
        return {m[0][0]*p.x+m[0][1]*p.y+m[0][2]*p.z+m[0][3],
                m[1][0]*p.x+m[1][1]*p.y+m[1][2]*p.z+m[1][3],
                m[2][0]*p.x+m[2][1]*p.y+m[2][2]*p.z+m[2][3]};
    }
    vec3 transformDir(const vec3& d) const {
        return {m[0][0]*d.x+m[0][1]*d.y+m[0][2]*d.z,
                m[1][0]*d.x+m[1][1]*d.y+m[1][2]*d.z,
                m[2][0]*d.x+m[2][1]*d.y+m[2][2]*d.z};
    }

    static mat4 perspective(float fov, float aspect, float near, float far) {
        mat4 r; float f = 1.0f/tanf(fov*0.5f);
        r.m[0][0]=f/aspect; r.m[1][1]=f;
        r.m[2][2]=(far+near)/(near-far);
        r.m[2][3]=(2.0f*far*near)/(near-far);
        r.m[3][2]=-1.0f; r.m[3][3]=0.0f;
        return r;
    }

    static mat4 lookAt(const vec3& eye, const vec3& target, const vec3& up) {
        vec3 z = (eye-target).normalized();
        vec3 x = up.cross(z).normalized();
        vec3 y = z.cross(x);
        mat4 r = identity();
        r.m[0][0]=x.x; r.m[0][1]=x.y; r.m[0][2]=x.z; r.m[0][3]=-x.dot(eye);
        r.m[1][0]=y.x; r.m[1][1]=y.y; r.m[1][2]=y.z; r.m[1][3]=-y.dot(eye);
        r.m[2][0]=z.x; r.m[2][1]=z.y; r.m[2][2]=z.z; r.m[2][3]=-z.dot(eye);
        return r;
    }

    static mat4 ortho(float left, float right, float bottom, float top, float near, float far) {
        mat4 r;
        r.m[0][0]=2.0f/(right-left);
        r.m[1][1]=2.0f/(top-bottom);
        r.m[2][2]=-2.0f/(far-near);
        r.m[0][3]=-(right+left)/(right-left);
        r.m[1][3]=-(top+bottom)/(top-bottom);
        r.m[2][3]=-(far+near)/(far-near);
        r.m[3][3]=1.0f;
        return r;
    }

    /// Invert via Gauss-Jordan elimination. Returns identity if singular.
    mat4 inverse() const {
        mat4 result = identity();
        float a[4][8];
        for (int i = 0; i < 4; ++i) {
            a[i][0] = m[i][0]; a[i][1] = m[i][1]; a[i][2] = m[i][2]; a[i][3] = m[i][3];
            a[i][4] = (i==0); a[i][5] = (i==1); a[i][6] = (i==2); a[i][7] = (i==3);
        }
        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 4; ++row)
                if (std::fabs(a[row][col]) > std::fabs(a[pivot][col])) pivot = row;
            std::swap(a[col], a[pivot]);
            if (std::fabs(a[col][col]) < 1e-12f) return identity();
            float inv = 1.0f / a[col][col];
            for (int j = 0; j < 8; ++j) a[col][j] *= inv;
            for (int row = 0; row < 4; ++row) {
                if (row == col) continue;
                float f = a[row][col];
                for (int j = 0; j < 8; ++j) a[row][j] -= f * a[col][j];
            }
        }
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.m[i][j] = a[i][4 + j];
        return result;
    }
};

} // namespace delirium