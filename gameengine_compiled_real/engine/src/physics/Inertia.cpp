// =============================================================================
// Inertia.cpp - mass distribution, centre of mass, and the inertia tensor of a
// skinned body.
//
// The header states the invariants; this file implements them. The load-bearing
// idea is the ONE at the top of the header: the mass belongs to the BONES, not
// to the surface. A surface's area depends on the LOD; a bone's mass is a
// property of the mechanism. So the distribution spreads each bone's authored
// mass EQUALLY over that bone's anchors -- and the total mass is therefore
// independent of how finely the body was tessellated.
//
// That is what "complete in representation, coarse for speed" requires: the
// coarse version has the same mass, the same centre of mass, and (to within
// the sampling error of where the mass sits) the same tensor.
// =============================================================================
#include "delirium/physics/Inertia.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace DeliRium {

namespace {

// A small, local 3-vector with the operations the tensor maths needs. The
// engine's Vec3 has no dot/cross at namespace scope (several .cpp files
// declare their own), so these stay file-local.
struct V3 {
    double x = 0, y = 0, z = 0;
    V3() = default;
    V3(double X, double Y, double Z) : x(X), y(Y), z(Z) {}
};
inline V3 operator+(const V3& a, const V3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline V3 operator-(const V3& a, const V3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline V3 operator*(const V3& a, double s)    { return {a.x*s, a.y*s, a.z*s}; }
inline double dot(const V3& a, const V3& b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline V3 cross(const V3& a, const V3& b) {
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
inline double norm(const V3& a) { return std::sqrt(dot(a, a)); }

// The symmetric 3x3 tensor, stored as the six independent components.
struct Tensor {
    double xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
};

// I += m * (r x (r x v)) for each point, about the ORIGIN. The full tensor
// about the centre of mass is recovered by the parallel-axis theorem, so the
// accumulation is done once, about the origin, and then shifted.
inline void Accumulate(Tensor& I, const V3& r, double m) {
    // r x (r x v) = r (r.v) - v (r.r)
    const double rr = dot(r, r);
    const double rx = r.x, ry = r.y, rz = r.z;
    // Column x of r x (r x e_x) etc. For a point at r with mass m, the
    // contribution to the tensor is m * (|r|^2 * I - r (x) r).
    I.xx += m * (rr - rx*rx);
    I.yy += m * (rr - ry*ry);
    I.zz += m * (rr - rz*rz);
    I.xy += m * (-rx*ry);
    I.xz += m * (-rx*rz);
    I.yz += m * (-ry*rz);
}

// Shift the tensor from about the origin to about the centre of mass.
inline Tensor ShiftToCom(Tensor I, const V3& com, double M) {
    const double d2 = dot(com, com);
    I.xx -= M * (d2 - com.x*com.x);
    I.yy -= M * (d2 - com.y*com.y);
    I.zz -= M * (d2 - com.z*com.z);
    I.xy += M * (com.x*com.y);
    I.xz += M * (com.x*com.z);
    I.yz += M * (com.y*com.z);
    return I;
}

} // namespace

// -----------------------------------------------------------------------------
// The directional inertia about an arbitrary unit axis: u . I . u.
//
// This is the "how hard is it to spin about THIS axis" question, answered
// directly from the tensor rather than by rotating the body and re-solving.
// -----------------------------------------------------------------------------
float MassProperties::InertiaAbout(const Vec3& unit_axis) const {
    const double ux = unit_axis.x, uy = unit_axis.y, uz = unit_axis.z;
    // u . I . u, with I stored as six components.
    const double v = Ixx*ux*ux + Iyy*uy*uy + Izz*uz*uz
                   + 2.0*(Ixy*ux*uy + Ixz*ux*uz + Iyz*uy*uz);
    return float(v);
}

// -----------------------------------------------------------------------------
// The principal moments (eigenvalues of the symmetric tensor) and their axes.
//
// These are the numbers that characterise a body's SHAPE: the largest moment
// is the axis the body is longest along, the smallest the axis it is thinnest
// along. A rod has one tiny moment (spin about its own length) and two equal
// large ones (spin across it) -- a sphere has three equal ones.
//
// The solver is the classical cyclic Jacobi method: it is short, has no
// eigenvalue ordering assumptions, and converges quadratically for a
// symmetric matrix -- which this is, by construction.
// -----------------------------------------------------------------------------
void MassProperties::PrincipalMoments(Vec3 moments[3], Vec3 axes[3]) const {
    // The symmetric matrix, in the full 3x3 form Jacobi wants.
    double a[3][3] = {
        {Ixx, Ixy, Ixz},
        {Ixy, Iyy, Iyz},
        {Ixz, Iyz, Izz}
    };
    // The rotation that diagonalises it: v' = V^T v.
    double v[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    for (int sweep = 0; sweep < 32; ++sweep) {
        // The largest off-diagonal magnitude decides when to stop.
        double off = 0;
        for (int p = 0; p < 3; ++p)
            for (int q = p+1; q < 3; ++q)
                off = std::max(off, std::fabs(a[p][q]));
        if (off < 1e-12) break;

        for (int p = 0; p < 3; ++p) {
            for (int q = p+1; q < 3; ++q) {
                if (std::fabs(a[p][q]) < 1e-18) continue;
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta*theta + 1.0));
                const double c = 1.0 / std::sqrt(t*t + 1.0);
                const double s = t * c;
                // Rotate rows/cols p,q of `a` and columns p,q of `v`.
                const double app = a[p][p], aqq = a[q][q], apq = a[p][q];
                a[p][p] = c*c*app - 2.0*s*c*apq + s*s*aqq;
                a[q][q] = s*s*app + 2.0*s*c*apq + c*c*aqq;
                a[p][q] = a[q][p] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = a[r][p], arq = a[r][q];
                    a[r][p] = a[p][r] = c*arp - s*arq;
                    a[r][q] = a[q][r] = s*arp + c*arq;
                }
                for (int r = 0; r < 3; ++r) {
                    const double vrp = v[r][p], vrq = v[r][q];
                    v[r][p] = c*vrp - s*vrq;
                    v[r][q] = s*vrp + c*vrq;
                }
            }
        }
    }

    // The diagonal holds the moments; the columns of `v` hold their axes.
    for (int i = 0; i < 3; ++i) {
        moments[i] = Vec3{float(a[i][i]), 0.f, 0.f};
        axes[i] = Vec3{float(v[0][i]), float(v[1][i]), float(v[2][i])};
        // Normalise (the rotation is orthogonal, but a sign/precision fix-up
        // costs nothing and keeps the axes unit-length for the caller).
        const double l = norm(V3(v[0][i], v[1][i], v[2][i]));
        if (l > 1e-12) {
            axes[i].x /= float(l);
            axes[i].y /= float(l);
            axes[i].z /= float(l);
        }
    }
    // Sort by DESCENDING moment, carrying each axis with its moment, so the
    // caller always gets "longest axis first".
    for (int i = 0; i < 3; ++i)
        for (int j = i+1; j < 3; ++j)
            if (moments[j].x > moments[i].x) {
                std::swap(moments[i], moments[j]);
                std::swap(axes[i], axes[j]);
            }
}

std::string MassProperties::Describe() const {
    char buf[512];
    Vec3 m[3], ax[3];
    PrincipalMoments(m, ax);
    std::snprintf(buf, sizeof(buf),
        "mass=%.4f com=(%.3f,%.3f,%.3f) I=(%.5f,%.5f,%.5f,%.5f,%.5f,%.5f)\n"
        "  principal: %.5f about (%.3f,%.3f,%.3f) | %.5f | %.5f",
        total_mass, centre_of_mass.x, centre_of_mass.y, centre_of_mass.z,
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz,
        m[0].x, ax[0].x, ax[0].y, ax[0].z, m[1].x, m[2].x);
    return std::string(buf);
}

// -----------------------------------------------------------------------------
// ComputeMassProperties - the entry point.
//
// The mass policy, in order:
//   1. If `bone_mass` is non-empty, each bone's authored mass is spread
//      EQUALLY over that bone's anchors. An anchor bound to a bone with no
//      entry contributes nothing -- the CALLER decides the density policy,
//      and a body with a skeleton but no authored masses has no mass.
//   2. If `bone_mass` is empty, the mass is read from the BONE RECORDS'
//      reserved bytes (the baker stored the authored mass there), and
//      spread the same way. This is the path a caller takes when it has
//      a reconstruction model but no separate mass table.
//   3. If neither yields a mass, every BOUND anchor gets
//      `fallback_density` -- a uniform-density body, which lets a caller
//      ask "the inertia of this shape" without authoring a skeleton.
//
// WHY the mass is spread equally rather than by patch area: the mass
// belongs to the BONE, and the bone does not care how many anchors carry
// it. Equal spreading makes the total mass exactly the authored sum,
// independent of the LOD -- which is the property the coarseness budget
// depends on.
//
// The bones argument is REQUIRED for the mass to mean anything: the
// skeleton is not in the PSFBody, it is in the reconstruction model.
bool ComputeMassProperties(const PSFBody& body,
                           const std::vector<float>& bone_mass,
                           float fallback_density,
                           MassProperties& out, std::string& why) {
    // No skeleton given: the caller supplied an explicit mass table (or
    // wants the uniform-density fallback).
    return ComputeMassProperties(body, std::vector<BoneRecord>(), bone_mass,
                                  fallback_density, out, why);
}

bool ComputeMassProperties(const PSFBody& body,
                           const std::vector<BoneRecord>& bones,
                           const std::vector<float>& bone_mass,
                           float fallback_density,
                           MassProperties& out, std::string& why) {
    const AnchorPool& pool = body.AnchorsConst();
    const size_t n = pool.Count();
    if (n == 0) { why = "the body has no anchors"; return false; }

    // Only the anchors a SURFACE actually references carry mass.
    //
    // WHY: the baker writes an anchor for EVERY grid cell, and the cells
    // outside a bounded substance's support get a well-formed PLACEHOLDER
    // record parked at the bbox centre ("purely a well-formed unused
    // record"). Those placeholders are not skin -- they are the absence of
    // skin -- and counting them put the body's mass at a single point.
    //
    // Measured on a rod (sx=0.5, sy=0.02) before this fix: 990 of 1024
    // anchors were placeholders at the bbox centre, so 97% of the mass sat
    // at one point and the inertia tensor was ~4x too small.
    std::vector<char> referenced(n, 0);
    for (const SurfaceRecord& s : body.SurfaceBlob())
        for (int k = 0; k < 4; ++k)
            if (s.anchor_idx[k] < n) referenced[s.anchor_idx[k]] = 1;

    std::vector<double> per_anchor(n, 0.0);

    std::vector<float> masses = bone_mass;
if (masses.empty() && !bones.empty()) {
    masses.resize(bones.size(), 0.f);
    for (size_t i = 0; i < bones.size(); ++i)
        // The baker stores the authored mass as a FLOAT in reserved[0..3]
        // (it is unbounded, so a normalised byte would clamp it).
        std::memcpy(&masses[i], bones[i].reserved, sizeof(float));
}
const bool use_bone_mass = !masses.empty();
std::vector<size_t> bone_anchor_count;
if (use_bone_mass) {
    // The bone count is the largest index seen plus one, so a sparse
    // vector still addresses correctly.
    size_t max_bone = 0;
    bool any = false;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t b = pool.Blob()->at(i).bone_index;
        if (b == 0xFFFFFFFFu) continue;
        max_bone = std::max(max_bone, size_t(b));
        any = true;
    }
    if (!any) {
        why = "the body has bones but no anchor is bound to one";
        return false;
    }
    bone_anchor_count.assign(max_bone + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t b = pool.Blob()->at(i).bone_index;
        if (b == 0xFFFFFFFFu) continue;
        if (!referenced[i]) continue;         // a placeholder carries no mass
        ++bone_anchor_count[b];
    }
}

double total = 0.0;
for (size_t i = 0; i < n; ++i) {
    const uint32_t bi = pool.Blob()->at(i).bone_index;
    double m = 0.0;
    if (use_bone_mass) {
        if (bi == 0xFFFFFFFFu) continue;         // unbound: no mass
        if (bi >= masses.size()) continue;        // no authored mass
        if (bone_anchor_count[bi] == 0) continue;
        if (!referenced[i]) continue;             // a placeholder carries no mass
        m = double(masses[bi]) / double(bone_anchor_count[bi]);
    } else {
        if (bi == 0xFFFFFFFFu) continue;         // fallback needs a bone
        if (!referenced[i]) continue;             // a placeholder carries no mass
        m = double(fallback_density);
    }
    per_anchor[i] = m;
    total += m;
}

if (total <= 0.0) {
    why = use_bone_mass
        ? "the body has no authored mass (every bone's mass is zero, or "
          "no bone carries an anchor)"
        : "the fallback density is zero, or no anchor is bound to a bone";
    return false;
}

    // The centre of mass.
    V3 com{0,0,0};
    std::vector<V3> pos(n);
    for (size_t i = 0; i < n; ++i) {
        if (per_anchor[i] <= 0.0) { pos[i] = V3(); continue; }
        const Vec3 p = pool.Position(uint16_t(i));
        pos[i] = V3(p.x, p.y, p.z);
        com = com + pos[i] * per_anchor[i];
    }
    com = com * (1.0 / total);

    // The tensor about the centre of mass: accumulate about the origin, then
    // shift by the parallel-axis theorem. Accumulating about the COM directly
    // would need the COM first -- which is why this is two passes.
    Tensor I;
    for (size_t i = 0; i < n; ++i) {
        if (per_anchor[i] <= 0.0) continue;
        Accumulate(I, pos[i], per_anchor[i]);
    }
    I = ShiftToCom(I, com, total);

    out.total_mass = float(total);
    out.centre_of_mass = Vec3{float(com.x), float(com.y), float(com.z)};
    out.Ixx = float(I.xx); out.Iyy = float(I.yy); out.Izz = float(I.zz);
    out.Ixy = float(I.xy); out.Ixz = float(I.xz); out.Iyz = float(I.yz);
    return true;
}

} // namespace DeliRium
