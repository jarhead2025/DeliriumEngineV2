// =============================================================================
// SeamBridge.cpp - Hermite blend bridge over an un-stitchable seam.
// =============================================================================
#include "delirium/psf/SeamBridge.h"
#include <cmath>

namespace DeliRium {

// edge (u,v) for edge index e at parameter t  (matches Seam edge layout)
static void edge_uv(int e, float t, float& u, float& v) {
    switch (e) {
        case 0: u = t; v = 0; break;   // v0
        case 1: u = t; v = 1; break;   // v1
        case 2: u = 0; v = t; break;   // u0
        default:u = 1; v = t; break;   // u1
    }
}

// the inward parametric direction at an edge: which of (du,dv) points INTO
// the patch interior (so its world derivative is the cross-boundary tangent)
static void inward_dir(int e, float& du, float& dv) {
    switch (e) {
        case 0: du = 0; dv = +1; break;  // v0 edge -> +v goes inward
        case 1: du = 0; dv = -1; break;  // v1 edge -> -v inward
        case 2: du = +1; dv = 0; break;  // u0 edge -> +u inward
        default:du = -1; dv = 0; break;  // u1 edge -> -u inward
    }
}

// cubic Hermite basis and derivatives
static inline void hermite(float s, float& h00, float& h01,
                           float& h10, float& h11) {
    const float s2 = s*s, s3 = s2*s;
    h00 = 2*s3 - 3*s2 + 1;
    h01 = -2*s3 + 3*s2;
    h10 = s3 - 2*s2 + s;
    h11 = s3 - s2;
}
static inline void hermite_d(float s, float& d00, float& d01,
                             float& d10, float& d11) {
    const float s2 = s*s;
    d00 = 6*s2 - 6*s;
    d01 = -6*s2 + 6*s;
    d10 = 3*s2 - 4*s + 1;
    d11 = 3*s2 - 2*s;
}

// sample patch edge curve A(t): position + cross-boundary tangent T(t).
// T points OUT of the patch into the gap (we negate the inward derivative),
// scaled so the Hermite blend has a natural fullness (proportional to the
// gap so the bridge bulges sensibly rather than collapsing flat).
static void edge_curve(const Patch& P, int e, float t, float tan_scale,
                       Vec3& C, Vec3& T) {
    float u, v; edge_uv(e, t, u, v);
    Vec3 pos, dPdu, dPdv;
    P.EvaluateDerivs(u, v, pos, dPdu, dPdv);
    C = pos;
    float du, dv; inward_dir(e, du, dv);
    // outward tangent = -(inward parametric derivative)
    T = Vec3{ -(dPdu.x*du + dPdv.x*dv),
              -(dPdu.y*du + dPdv.y*dv),
              -(dPdu.z*du + dPdv.z*dv) };
    T.x *= tan_scale; T.y *= tan_scale; T.z *= tan_scale;
}

// B's parameter for A's t, honoring the seam's tb0..tb1 correspondence
static inline float b_param(const Seam& s, float t) {
    return s.tb0 + (s.tb1 - s.tb0) * t;
}

// shared core: evaluate position + both surface derivatives at (ss,tt)
static void eval_full(const PSFBody& body, const Seam& s, float ss, float tt,
                      Vec3& S, Vec3& dSds, Vec3& dSdt) {
    const Patch& A = body.Patches()[s.patch_a];
    const Patch& B = body.Patches()[s.patch_b];

    // tangent scale ~ distance between the two edges (so the bulge matches
    // the gap width; a flat ruled surface would crease at the joins)
    Vec3 ca0, ta0, cb0, tb0v;
    edge_curve(A, s.edge_a, tt, 1.0f, ca0, ta0);
    edge_curve(B, s.edge_b, b_param(s, tt), 1.0f, cb0, tb0v);
    const float gap = std::sqrt((cb0.x-ca0.x)*(cb0.x-ca0.x)
        + (cb0.y-ca0.y)*(cb0.y-ca0.y) + (cb0.z-ca0.z)*(cb0.z-ca0.z));
    // normalize each tangent and rescale to ~gap so H10/H11 contribute a
    // proportionate bulge; guard against zero-length tangents
    auto norm_scale = [&](Vec3 T)->Vec3 {
        const float l = std::sqrt(T.x*T.x+T.y*T.y+T.z*T.z);
        const float k = (l > 1e-9f) ? (gap / l) : 0.0f;
        return Vec3{T.x*k, T.y*k, T.z*k};
    };
    const Vec3 Ta = norm_scale(ta0);
    // B's outward tangent points the opposite way across the gap; flip so
    // both tangents are oriented A->B consistently for the blend
    const Vec3 Tb = norm_scale(Vec3{-tb0v.x, -tb0v.y, -tb0v.z});

    float h00,h01,h10,h11, d00,d01,d10,d11;
    hermite(ss, h00,h01,h10,h11);
    hermite_d(ss, d00,d01,d10,d11);

    S = Vec3{ h00*ca0.x + h01*cb0.x + h10*Ta.x + h11*Tb.x,
              h00*ca0.y + h01*cb0.y + h10*Ta.y + h11*Tb.y,
              h00*ca0.z + h01*cb0.z + h10*Ta.z + h11*Tb.z };
    dSds = Vec3{ d00*ca0.x + d01*cb0.x + d10*Ta.x + d11*Tb.x,
                 d00*ca0.y + d01*cb0.y + d10*Ta.y + d11*Tb.y,
                 d00*ca0.z + d01*cb0.z + d10*Ta.z + d11*Tb.z };

    // dS/dt by central difference along the seam (curves are cheap)
    const float e = 1e-3f;
    Vec3 cap,tap,cbp,tbp, cam,tam,cbm,tbm;
    const float tp = std::min(1.0f, tt+e), tm = std::max(0.0f, tt-e);
    edge_curve(A, s.edge_a, tp, 1.0f, cap, tap);
    edge_curve(B, s.edge_b, b_param(s,tp), 1.0f, cbp, tbp);
    edge_curve(A, s.edge_a, tm, 1.0f, cam, tam);
    edge_curve(B, s.edge_b, b_param(s,tm), 1.0f, cbm, tbm);
    auto blendpos = [&](const Vec3& a, const Vec3& b){
        return Vec3{ h00*a.x + h01*b.x, h00*a.y + h01*b.y, h00*a.z + h01*b.z }; };
    const Vec3 Sp = blendpos(cap, cbp), Sm = blendpos(cam, cbm);
    const float inv = 1.0f / (tp - tm);
    dSdt = Vec3{ (Sp.x-Sm.x)*inv, (Sp.y-Sm.y)*inv, (Sp.z-Sm.z)*inv };
}

BridgeSample SeamBridge::Evaluate(const PSFBody& body, const Seam& s,
                                  float ss, float tt) {
    Vec3 S, dSds, dSdt;
    eval_full(body, s, ss, tt, S, dSds, dSdt);
    // analytic normal = dS/ds x dS/dt
    Vec3 n{ dSds.y*dSdt.z - dSds.z*dSdt.y,
            dSds.z*dSdt.x - dSds.x*dSdt.z,
            dSds.x*dSdt.y - dSds.y*dSdt.x };
    const float l = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
    if (l > 1e-12f) { n.x/=l; n.y/=l; n.z/=l; }
    return { S, n };
}

bool SeamBridge::NeedsBridge(const PSFBody& body, const Seam& s, float tol) {
    const Patch& A = body.Patches()[s.patch_a];
    const Patch& B = body.Patches()[s.patch_b];
    const Vec3 sp = body.BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    float gmax = 0;
    for (int i = 1; i < 9; ++i) {
        const float t = i / 9.0f;
        float ua,va,ub,vb;
        edge_uv(s.edge_a, t, ua, va);
        edge_uv(s.edge_b, b_param(s, t), ub, vb);
        const Vec3 pa = A.Evaluate(ua, va), pb = B.Evaluate(ub, vb);
        gmax = std::fmax(gmax, std::sqrt((pa.x-pb.x)*(pa.x-pb.x)
            + (pa.y-pb.y)*(pa.y-pb.y) + (pa.z-pb.z)*(pa.z-pb.z)));
    }
    return gmax > tol * diag;
}

float SeamBridge::BoundaryGap(const PSFBody& body, const Seam& s) {
    const Patch& A = body.Patches()[s.patch_a];
    const Patch& B = body.Patches()[s.patch_b];
    float gmax = 0;
    for (int i = 0; i <= 8; ++i) {
        const float t = i / 8.0f;
        // bridge boundary at s=0 must equal A's edge; s=1 must equal B's
        const BridgeSample b0 = Evaluate(body, s, 0.0f, t);
        const BridgeSample b1 = Evaluate(body, s, 1.0f, t);
        float ua,va,ub,vb;
        edge_uv(s.edge_a, t, ua, va);
        edge_uv(s.edge_b, b_param(s, t), ub, vb);
        const Vec3 pa = A.Evaluate(ua, va), pb = B.Evaluate(ub, vb);
        gmax = std::fmax(gmax, std::sqrt((b0.P.x-pa.x)*(b0.P.x-pa.x)
            + (b0.P.y-pa.y)*(b0.P.y-pa.y) + (b0.P.z-pa.z)*(b0.P.z-pa.z)));
        gmax = std::fmax(gmax, std::sqrt((b1.P.x-pb.x)*(b1.P.x-pb.x)
            + (b1.P.y-pb.y)*(b1.P.y-pb.y) + (b1.P.z-pb.z)*(b1.P.z-pb.z)));
    }
    return gmax;
}

float SeamBridge::G1Error(const PSFBody& body, const Seam& s) {
    const Patch& A = body.Patches()[s.patch_a];
    const Patch& B = body.Patches()[s.patch_b];
    float worst = 0;
    for (int i = 1; i < 8; ++i) {
        const float t = i / 8.0f;
        // bridge cross-tangent at s=0 vs A's outward edge tangent
        Vec3 S, dSds, dSdt; eval_full(body, s, 0.0f, t, S, dSds, dSdt);
        Vec3 ca, ta; edge_curve(A, s.edge_a, t, 1.0f, ca, ta);
        auto cos_ang = [](Vec3 x, Vec3 y){
            const float lx=std::sqrt(x.x*x.x+x.y*x.y+x.z*x.z);
            const float ly=std::sqrt(y.x*y.x+y.y*y.y+y.z*y.z);
            if (lx<1e-9f||ly<1e-9f) return 1.0f;
            return (x.x*y.x+x.y*y.y+x.z*y.z)/(lx*ly);
        };
        worst = std::fmax(worst, 1.0f - std::fabs(cos_ang(dSds, ta)));
        // at s=1 vs B's outward tangent
        Vec3 S2, dSds2, dSdt2; eval_full(body, s, 1.0f, t, S2, dSds2, dSdt2);
        Vec3 cb, tb; edge_curve(B, s.edge_b, b_param(s,t), 1.0f, cb, tb);
        worst = std::fmax(worst, 1.0f - std::fabs(cos_ang(dSds2, tb)));
    }
    return worst;
}

} // namespace DeliRium
