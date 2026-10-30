// =============================================================================
// Patch.cpp - CPU evaluation of one PSF v2 quad patch.
//
// THIS IS THE CPU MIRROR OF kernels/psf_splat.cl. Every method below must
// produce bit-near-identical results to the OpenCL kernel for the same
// (u, v) and records. The two are cross-validated by tests/ (validate_eval
// compares the analytic normal against a finite-difference normal, and the
// splat path is the reference the CPU must match).
//
// Surface(u,v) = Bilinear(anchors) + h(u,v) * NormalField(u,v)
//   h = deform_h(coef, u, v) + stitch_h(stitch, u, v)
// The corner-vanishing basis makes h == 0 at all four corners, so the
// surface ALWAYS interpolates its anchors regardless of coefficients.
//
// Corner ordering (matches SurfaceRecord::anchor_idx and the kernel):
//   0 = (u=0,v=0)   1 = (u=1,v=0)   2 = (u=1,v=1)   3 = (u=0,v=1)
// =============================================================================
#include "delirium/psf/Patch.h"
#include "delirium/psf/AnchorPool.h"
#include <cmath>

namespace DeliRium {

namespace {

// ---- dequant helpers (mirror kernel anchor_pos / anchor_nrm) ------------
inline Vec3 dequant_pos(const AnchorRecord& a, const Vec3& mn, const Vec3& sp) {
    const float k = 1.0f / 65535.0f;
    return { mn.x + sp.x * a.pos[0] * k,
             mn.y + sp.y * a.pos[1] * k,
             mn.z + sp.z * a.pos[2] * k };
}
inline Vec3 dequant_dir(const AnchorRecord& a) {
    const float k = 2.0f / 255.0f;
    Vec3 v{ a.normal[0] * k - 1.0f, a.normal[1] * k - 1.0f, a.normal[2] * k - 1.0f };
    float l = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (l < 1e-12f) return {0.f, 0.f, 1.f};
    return { v.x/l, v.y/l, v.z/l };
}
inline Vec3 lerp3(const Vec3& a, const Vec3& b, float t) {
    return { a.x + (b.x - a.x)*t, a.y + (b.y - a.y)*t, a.z + (b.z - a.z)*t };
}
inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
inline Vec3 norm3(const Vec3& v) {
    float l = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (l < 1e-12f) return {0.f, 0.f, 1.f};
    return { v.x/l, v.y/l, v.z/l };
}

// corner-vanishing deformation basis (mirror of kernel deform_h)
inline float deform_h(const float c[8], float u, float v) {
    float bu = u * (1.0f - u), bv = v * (1.0f - v);
    float au = 2.0f*u - 1.0f,  av = 2.0f*v - 1.0f;
    float w  = 1.0f - v,       x  = 1.0f - u;
    return c[0]*bu*w + c[1]*bu*v + c[2]*bv*x + c[3]*bv*u
         + c[4]*bu*au*w + c[5]*bu*au*v + c[6]*bv*av*x + c[7]*bv*av*u;
}
// analytic partials of deform_h (mirror of kernel deform_dh)
inline void deform_dh(const float c[8], float u, float v,
                      float& dhu, float& dhv) {
    float bu = u * (1.0f - u), bv = v * (1.0f - v);
    float au = 2.0f*u - 1.0f,  av = 2.0f*v - 1.0f;
    float w  = 1.0f - v,       x  = 1.0f - u;
    float dbu = 1.0f - 2.0f*u, dbv = 1.0f - 2.0f*v;
    float dcu = dbu*au + 2.0f*bu;   // d/du [u(1-u)(2u-1)]
    float dcv = dbv*av + 2.0f*bv;   // d/dv [v(1-v)(2v-1)]
    dhu = c[0]*dbu*w + c[1]*dbu*v - c[2]*bv + c[3]*bv
        + c[4]*dcu*w + c[5]*dcu*v - c[6]*bv*av + c[7]*bv*av;
    dhv = -c[0]*bu + c[1]*bu + c[2]*dbv*x + c[3]*dbv*u
        - c[4]*bu*au + c[5]*bu*au + c[6]*dcv*x + c[7]*dcv*u;
}
// seam height from the 8 int8 coefficients (mirror of kernel stitch_h)
inline float stitch_h(const int8_t* st, float u, float v, float s_scale) {
    if (!st) return 0.0f;
    float tu = u * (1.0f - u), tv = v * (1.0f - v);
    return (1.0f - v) * tu * (st[0]*s_scale + st[1]*s_scale*u)
         + v * tu * (st[2]*s_scale + st[3]*s_scale*u)
         + (1.0f - u) * tv * (st[4]*s_scale + st[5]*s_scale*v)
         + u * tv * (st[6]*s_scale + st[7]*s_scale*v);
}
// analytic partials of stitch_h
inline void stitch_dh(const int8_t* st, float u, float v, float s_scale,
                      float& dhu, float& dhv) {
    if (!st) { dhu = 0.0f; dhv = 0.0f; return; }
    float tu = u * (1.0f - u), tv = v * (1.0f - v);
    float dbu = 1.0f - 2.0f*u, dbv = 1.0f - 2.0f*v;
    dhu = (1.0f - v) * (dbu*(st[0]*s_scale + st[1]*s_scale*u)
                        + tu*st[1]*s_scale)
        + v * (dbu*(st[2]*s_scale + st[3]*s_scale*u)
               + tu*st[3]*s_scale)
        + (-tv*(st[4]*s_scale + st[5]*s_scale*v)
           + (1.0f - u)*st[5]*s_scale*dbv)
        + (tv*(st[6]*s_scale + st[7]*s_scale*v)
           + u*st[7]*s_scale*dbv);
    dhv = (-tu*(st[0]*s_scale + st[1]*s_scale*u)
           + (1.0f - u)*st[1]*s_scale*dbu)
        + (tu*(st[2]*s_scale + st[3]*s_scale*u)
           + u*st[3]*s_scale*dbu)
        + (1.0f - u) * (dbv*(st[4]*s_scale + st[5]*s_scale*v)
                        + tv*st[5]*s_scale)
        + u * (dbv*(st[6]*s_scale + st[7]*s_scale*v)
               + tv*st[7]*s_scale);
}
// the 16 color patterns (mirror of kernel pattern_eval)
inline float pattern_eval(int idx, float u, float v) {
    switch (idx & 15) {
        case 0:  return 1.0f;
        case 1:  return u;
        case 2:  return v;
        case 3:  return u*v;
        case 4:  return u*u;
        case 5:  return v*v;
        case 6:  return u*u*v;
        case 7:  return u*v*v;
        case 8:  return std::cos(3.14159265358979323846f * u);
        case 9:  return std::cos(3.14159265358979323846f * v);
        case 10: return std::cos(3.14159265358979323846f * 2.0f*u);
        case 11: return std::cos(3.14159265358979323846f * 2.0f*v);
        case 12: return std::cos(3.14159265358979323846f * u)
                     * std::cos(3.14159265358979323846f * v);
        case 13: return std::cos(3.14159265358979323846f * 2.0f*u)
                     * std::cos(3.14159265358979323846f * 2.0f*v);
        case 14: return std::cos(3.14159265358979323846f * 3.0f*u);
        default: return std::cos(3.14159265358979323846f * 3.0f*v);
    }
}
// one color channel (mirror of kernel channel_color)
inline float channel_color(const uint8_t* c, float u, float v) {
    uint8_t i0 = c[0] >> 4, i1 = c[0] & 15u, i2 = c[1] >> 4, i3 = c[1] & 15u;
    const int8_t* K = reinterpret_cast<const int8_t*>(c + 2);
    float r = (K[0] * (1.0f/63.0f)) * pattern_eval(i0, u, v);
    r += (K[1] * (1.0f/63.0f)) * pattern_eval(i1, u, v);
    r += (K[2] * (1.0f/63.0f)) * pattern_eval(i2, u, v);
    r += (K[3] * (1.0f/63.0f)) * pattern_eval(i3, u, v);
    if (r < 0.0f) r = 0.0f; else if (r > 1.0f) r = 1.0f;
    return r;
}

} // namespace

Patch::Patch(SurfaceRecord* r, AnchorPool* a, int8_t* s, float ss, uint32_t id)
    : rec(r), pool(a), stitch(s), s_scale(ss), id(id), dirty(true) {}

void Patch::EvaluateDerivs(float u, float v, Vec3& P,
                           Vec3& dPdu, Vec3& dPdv) const {
    if (!pool || !pool->Blob()) {
        P = {0.f, 0.f, 0.f}; dPdu = {1.f, 0.f, 0.f}; dPdv = {0.f, 1.f, 0.f};
        return;
    }
    const AnchorRecord* A = pool->Blob()->data();
    const Vec3 mn = pool->BBoxMin(), sp = pool->BBoxSpan();
    // corner positions + displacement directions
    Vec3 P0 = dequant_pos(A[rec->anchor_idx[0]], mn, sp);
    Vec3 P1 = dequant_pos(A[rec->anchor_idx[1]], mn, sp);
    Vec3 P2 = dequant_pos(A[rec->anchor_idx[2]], mn, sp);
    Vec3 P3 = dequant_pos(A[rec->anchor_idx[3]], mn, sp);
    Vec3 N0 = dequant_dir(A[rec->anchor_idx[0]]);
    Vec3 N1 = dequant_dir(A[rec->anchor_idx[1]]);
    Vec3 N2 = dequant_dir(A[rec->anchor_idx[2]]);
    Vec3 N3 = dequant_dir(A[rec->anchor_idx[3]]);
    // base bilinear position
    Vec3 pl = lerp3(P0, P3, u);
    Vec3 pr = lerp3(P1, P2, u);
    Vec3 Pb = lerp3(pl, pr, v);
    // displacement direction field (bilinear, then normalize)
    Vec3 nld = lerp3(N0, N3, u);
    Vec3 nrd = lerp3(N1, N2, u);
    Vec3 Nd = norm3(lerp3(nld, nrd, v));
    // height (deform + seam) for POSITION -- the surface is displaced by both
    float h = deform_h(rec->coef, u, v) + stitch_h(stitch, u, v, s_scale);
    P = { Pb.x + h*Nd.x, Pb.y + h*Nd.y, Pb.z + h*Nd.z };
    // base bilinear partials. The POSITION above is convention A:
    //   Pb = (1-u)(1-v)P0 + u(1-v)P3 + (1-u)v P1 + uv P2   (u: P0->P3, v: P0->P1)
    // matching kernel psf_splat.cl:214-216 (pl=mix(P0,P3,u), pr=mix(P1,P2,u)).
    // The partials MUST use the SAME convention, or cross(dPdu,dPdv) flips sign
    // (the validate_eval 1-cos=2.0 bug: these were written for the transposed
    // convention B, P=(1-u)(1-v)P0+u(1-v)P1+uv P2+(1-u)v P3).
    Vec3 dPbdu{ (P3.x - P0.x)*(1.0f - v) + (P2.x - P1.x)*v,
                (P3.y - P0.y)*(1.0f - v) + (P2.y - P1.y)*v,
                (P3.z - P0.z)*(1.0f - v) + (P2.z - P1.z)*v };
    Vec3 dPbdv{ (P1.x - P0.x)*(1.0f - u) + (P2.x - P3.x)*u,
                (P1.y - P0.y)*(1.0f - u) + (P2.y - P3.y)*u,
                (P1.z - P0.z)*(1.0f - u) + (P2.z - P3.z)*u };
    // displacement-direction partials (bilinear, UNnormalized) -- used for
    // the exact derivative of the NORMALIZED direction field below.
    // Same convention A as the position: n = (1-u)(1-v)N0 + u(1-v)N3 + (1-u)v N1 + uv N2.
    Vec3 dNdu{ (N3.x - N0.x)*(1.0f - v) + (N2.x - N1.x)*v,
               (N3.y - N0.y)*(1.0f - v) + (N2.y - N1.y)*v,
               (N3.z - N0.z)*(1.0f - v) + (N2.z - N1.z)*v };
    Vec3 dNdv{ (N1.x - N0.x)*(1.0f - u) + (N2.x - N3.x)*u,
               (N1.y - N0.y)*(1.0f - u) + (N2.y - N3.y)*u,
               (N1.z - N0.z)*(1.0f - u) + (N2.z - N3.z)*u };
    // exact derivative of the NORMALIZED direction field:
    //   Nd = n/|n|  =>  dNd = (dn - Nd*dot(Nd,dn))/|n|
    // (n = unnormalized bilinear, |n| = its length). This makes dPdu/dPdv
    // the EXACT derivative of P = Pb + h*Nd, so EvaluateNormal (cross of the
    // two) matches a finite-difference of Evaluate() to O(e^2) -- which is
    // what validate_eval checks.
    Vec3 n_un{ nld.x + (nrd.x - nld.x)*v,
               nld.y + (nrd.y - nld.y)*v,
               nld.z + (nrd.z - nld.z)*v };
    const float nlen = std::sqrt(n_un.x*n_un.x + n_un.y*n_un.y + n_un.z*n_un.z);
    const float inv_nlen = nlen > 1e-12f ? 1.0f/nlen : 0.0f;
    const float dndu_dot = dNdu.x*Nd.x + dNdu.y*Nd.y + dNdu.z*Nd.z;
    const float dndv_dot = dNdv.x*Nd.x + dNdv.y*Nd.y + dNdv.z*Nd.z;
    Vec3 dNd_du{ (dNdu.x - Nd.x*dndu_dot)*inv_nlen,
                 (dNdu.y - Nd.y*dndu_dot)*inv_nlen,
                 (dNdu.z - Nd.z*dndu_dot)*inv_nlen };
    Vec3 dNd_dv{ (dNdv.x - Nd.x*dndv_dot)*inv_nlen,
                 (dNdv.y - Nd.y*dndv_dot)*inv_nlen,
                 (dNdv.z - Nd.z*dndv_dot)*inv_nlen };
    // height partials (deform + seam)
    float dhu, dhv;
    deform_dh(rec->coef, u, v, dhu, dhv);
    float shu, shv;
    stitch_dh(stitch, u, v, s_scale, shu, shv);
    dhu += shu; dhv += shv;
    // P = Pb + h*Nd  =>  dP = dPb + dh*Nd + h*dNd  (exact)
    dPdu = { dPbdu.x + dhu*Nd.x + h*dNd_du.x,
             dPbdu.y + dhu*Nd.y + h*dNd_du.y,
             dPbdu.z + dhu*Nd.z + h*dNd_du.z };
    dPdv = { dPbdv.x + dhv*Nd.x + h*dNd_dv.x,
             dPbdv.y + dhv*Nd.y + h*dNd_dv.y,
             dPbdv.z + dhv*Nd.z + h*dNd_dv.z };
}

Vec3 Patch::Evaluate(float u, float v) const {
    Vec3 P, du, dv;
    EvaluateDerivs(u, v, P, du, dv);
    return P;
}

Vec3 Patch::EvaluateNormal(float u, float v) const {
    Vec3 P, du, dv;
    EvaluateDerivs(u, v, P, du, dv);
    return norm3(cross3(du, dv));
}

Vec3 Patch::EvaluateNormalField(float u, float v) const {
    const AnchorRecord* A = pool->Blob()->data();
    Vec3 N0 = dequant_dir(A[rec->anchor_idx[0]]);
    Vec3 N1 = dequant_dir(A[rec->anchor_idx[1]]);
    Vec3 N2 = dequant_dir(A[rec->anchor_idx[2]]);
    Vec3 N3 = dequant_dir(A[rec->anchor_idx[3]]);
    Vec3 nld = lerp3(N0, N3, u);
    Vec3 nrd = lerp3(N1, N2, u);
    return norm3(lerp3(nld, nrd, v));
}

float Patch::EvaluateHeight(float u, float v) const {
    return deform_h(rec->coef, u, v) + stitch_h(stitch, u, v, s_scale);
}

Vec3 Patch::EvaluateColor(float u, float v) const {
    const uint8_t* cp = rec->color_pattern;   // 3 channels x 6 bytes
    Vec3 c;
    c.x = channel_color(cp + 0, u, v);
    c.y = channel_color(cp + 6, u, v);
    c.z = channel_color(cp + 12, u, v);
    return c;
}

void Patch::AddColorK(uint32_t pid, float k) {
    // Add `k` to the contribution of pattern `pid` on every channel where it
    // is already in the 4-pattern set (Morph slider). Find the matching
    // pattern index in each channel and bump its K by k*63 (clamped).
    for (int ch = 0; ch < 3; ++ch) {
        uint8_t* c = rec->color_pattern + ch*6;
        uint8_t idx[4] = { c[0] >> 4, c[0] & 15u, c[1] >> 4, c[1] & 15u };
        int8_t*  K  = reinterpret_cast<int8_t*>(c + 2);
        for (int i = 0; i < 4; ++i) {
            if (idx[i] == (pid & 15u)) {
                int nv = (int)K[i] + (int)(k * 63.0f + (k >= 0 ? 0.5f : -0.5f));
                if (nv > 127) nv = 127; else if (nv < -128) nv = -128;
                K[i] = (int8_t)nv;
            }
        }
    }
    dirty = true;
}

void Patch::SetCoefficient(int i, float value) {
    if (i < 0 || i > 7) return;
    rec->coef[i] = value;
    dirty = true;
}

float Patch::GetCoefficient(int i) const {
    if (i < 0 || i > 7) return 0.0f;
    return rec->coef[i];
}

void Patch::AddDisplacementMode(const float k[8]) {
    for (int i = 0; i < 8; ++i) rec->coef[i] += k[i];
    dirty = true;
}

void Patch::PinEdgeToCurve(PatchEdge, float, float) {
    // TODO via SeamGraph -- edge pinning is handled by the seam layer, not
    // the patch. No-op for now (keeps the API surface stable).
}

void Patch::HeightGrad(float u, float v, float& dh_du, float& dh_dv) const {
    float dhu, dhv;
    deform_dh(rec->coef, u, v, dhu, dhv);
    float shu, shv;
    stitch_dh(stitch, u, v, s_scale, shu, shv);
    dh_du = dhu + shu;
    dh_dv = dhv + shv;
}

} // namespace DeliRium