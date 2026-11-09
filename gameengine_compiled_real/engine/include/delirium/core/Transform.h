// =============================================================================
// Transform.h - Rigid world transform for a GameObject.
//
// WHY A SEPARATE TRANSFORM (and not PSFBody::Translate):
//   The 64-byte AnchorRecord/SurfaceRecord layouts are FROZEN and shared
//   byte-for-byte with the GPU kernels, the .psf file format and the Python
//   encoder. Anchors are QUANTIZED over the body's bbox, so translating a
//   body by mutating its anchors re-quantizes every position and can lose
//   sub-quantum precision -- permanently, for a placement that is pure
//   metadata. PSFBody::Translate exists for the one-shot case of placing a
//   FILE-LOADED body and is correct there.
//
//   A workspace needs to MOVE objects around continuously (drag a gizmo,
//   place a generated object, undo). So the placement lives HERE, outside
//   the records, and the renderer applies it as a MODEL matrix on the GPU.
//
//   Consequence: the 64-byte records stay byte-identical to the file on
//   disk; moving an object is free (no re-upload of the anchor blob).
//
// The matrix is row-major 4x4, matching Camera::BuildMVP's convention
// (out[i*4+j] = row i, column j) and the kernels' float16 MVP upload.
// =============================================================================
#pragma once
#include "../core/Types.h"

namespace DeliRium {

// Row-major 4x4. m[r*4+c].
struct Mat4 {
    float m[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

// Rigid transform: rotation (quaternion) + translation + uniform scale.
struct Transform {
    Vec3  position{0,0,0};
    Quat  rotation{0,0,0,1};   // x,y,z,w
    Vec3  scale{1,1,1};

    // World -> local (inverse of ToMatrix, for uniform+rigid rotation).
    Vec3  InversePoint(const Vec3& world) const;
    Vec3  InverseVector(const Vec3& v) const;
    Mat4  ToMatrix() const;              // local -> world
    // Local -> world direction (rotation only, no translation/scale).
    Vec3  Rotate(const Vec3& local) const;
    bool  IsIdentity() const;
};

Mat4 Mat4_Multiply(const Mat4& a, const Mat4& b);   // a * b
Vec3 Mat4_TransformPoint(const Mat4& m, const Vec3& p);
Vec3 Mat4_TransformVector(const Mat4& m, const Vec3& v);

} // namespace DeliRium
