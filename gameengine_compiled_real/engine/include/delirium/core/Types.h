// =============================================================================
// Types.h - Packed binary records shared by file format, CPU and GPU.
// The byte layout is FROZEN and validated byte-for-byte against
// tools/psf_encode.py and kernels/psf_splat.cl. Do not reorder fields.
// =============================================================================
#pragma once
#include <cstdint>
#include <cmath>

namespace DeliRium {

struct Vec3  { float x, y, z; };
struct Quat  { float x, y, z, w; };

// Minimal vector algebra. The engine predates these (most math is inline
// component arithmetic in each .cpp), but core/Transform.h needs subtraction
// and the workspace needs the usual set. Kept header-only so no ABI or link
// changes ripple through the frozen 64-byte records.
//
// NOTE: `dot` / `cross` / `length` are deliberately NOT defined here --
// several translation units already declare their own file-local versions
// (PhysicsWorld.cpp, AnchorPhysics.cpp, SurfelRasterizer.cpp) and adding
// namespace-scope ones would make those calls ambiguous.
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator-(const Vec3& a)                 { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(const Vec3& a, float s)        { return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator*(float s, const Vec3& a)        { return {a.x*s, a.y*s, a.z*s}; }
inline Vec3 operator/(const Vec3& a, float s)        { return {a.x/s, a.y/s, a.z/s}; }
inline Vec3& operator+=(Vec3& a, const Vec3& b)      { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
inline Vec3& operator-=(Vec3& a, const Vec3& b)      { a.x-=b.x; a.y-=b.y; a.z-=b.z; return a; }
inline Vec3& operator*=(Vec3& a, float s)            { a.x*=s; a.y*=s; a.z*=s; return a; }

#pragma pack(push, 1)

// 64 bytes. One shared corner between neighboring patches.
// curve_params doubles as the SEAM (stitch) coefficient store:
// per outgoing connection, int8 {a, b} of s(t) = t(1-t)(a + b*t).
struct AnchorRecord {
    uint32_t bone_index;        //  0
    uint16_t connectivity[4];   //  4  neighbor anchor indices (0xFFFF = none)
    uint8_t  curve_types[4];    // 12
    int8_t   curve_params[16];  // 16  <- seam coefficients live here
    uint16_t pos[3];            // 32  quantized over object bbox
    int16_t  delta[3];          // 38  animation displacement
    uint8_t  normal[3];         // 44  displacement direction field
    uint8_t  tangents[4];       // 47
    uint8_t  color[3];          // 51
    uint8_t  roughness;         // 54
    uint8_t  metallic;          // 55
    uint8_t  anim_weight;       // 56
    uint8_t  curvature[3];      // 57
    uint8_t  reserved[4];       // 60
};
static_assert(sizeof(AnchorRecord) == 64, "frozen layout");

// 64 bytes. One quad patch: 4 anchors + corner-vanishing deformation
// basis (8 coefficients) + pattern-sum color (3 ch x [2B idx + 4B K]).
struct SurfaceRecord {
    uint16_t anchor_idx[4];     //  0
    uint32_t surface_def;       //  8  equation/family id
    float    coef[8];           // 12  corner-vanishing basis weights
    uint8_t  color_pattern[18]; // 44  per ch: idx nibbles x4 + int8 K x4
    uint8_t  pad[2];            // 62
};
static_assert(sizeof(SurfaceRecord) == 64, "frozen layout");

// 64 bytes. Skeletal node for NPCs / articulated objects.
struct BoneRecord {
    uint32_t parent;            //  0
    float    bind_pos[3];       //  4
    float    bind_rot[4];       // 16
    float    length;            // 32
    uint8_t  reserved[28];      // 36
};
static_assert(sizeof(BoneRecord) == 64, "frozen layout");

#pragma pack(pop)

} // namespace DeliRium
