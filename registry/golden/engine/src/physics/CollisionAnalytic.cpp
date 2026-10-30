#include "delirium/physics/CollisionAnalytic.h"
#include <cmath>
#include <algorithm>

namespace DeliRium {

// ---- quaternion helpers (unit q) -------------------------------------------
static Vec3 qrot(const Quat& q, const Vec3& v) {
    const Vec3 u{q.x, q.y, q.z};
    const Vec3 c{u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x};
    const Vec3 cc{u.y*c.z-u.z*c.y, u.z*c.x-u.x*c.z, u.x*c.y-u.y*c.x};
    return { v.x + 2*(q.w*c.x + cc.x),
             v.y + 2*(q.w*c.y + cc.y),
             v.z + 2*(q.w*c.z + cc.z) };
}
static Vec3 qrot_inv(const Quat& q, const Vec3& v) {
    return qrot({-q.x, -q.y, -q.z, q.w}, v);
}

void CollisionAnalytic::BodyVsPlane(const PSFBody& body, const Transform& xf,
                                    const Vec3& n, float d,
                                    std::vector<Contact>& out, int maxc) {
    // plane to LOCAL space: n_l . p = d_l   (uniform scale=1 assumed v1)
    const Vec3 nl = qrot_inv(xf.rotation, n);
    const float dl = d - (n.x*xf.position.x + n.y*xf.position.y
                        + n.z*xf.position.z);
    const auto& P = body.Patches();
    auto& pool = body.AnchorsConst();
    const Vec3 sp = body.BBoxSpan();
    const float margin = 0.05f * std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);

    std::vector<Contact> cand;
    for (uint32_t pid = 0; pid < P.size(); ++pid) {
        // broad: any anchor near/below the plane?
        const SurfaceRecord& S = body.SurfaceBlob()[pid];
        float dmin = 1e30f;
        for (int k = 0; k < 4; ++k) {
            const Vec3 a = pool.Position(S.anchor_idx[k]);
            dmin = std::fmin(dmin, a.x*nl.x + a.y*nl.y + a.z*nl.z - dl);
        }
        if (dmin > margin) continue;
        // Newton-descend f(u,v) = P(u,v).nl  from 2x2 seeds (projected)
        for (int si = 0; si < 2; ++si)
            for (int sj = 0; sj < 2; ++sj) {
                float u = 0.25f + 0.5f*si, v = 0.25f + 0.5f*sj;
                for (int it = 0; it < 10; ++it) {
                    Vec3 q, du, dv;
                    P[pid].EvaluateDerivs(u, v, q, du, dv);
                    const float gu = du.x*nl.x + du.y*nl.y + du.z*nl.z;
                    const float gv = dv.x*nl.x + dv.y*nl.y + dv.z*nl.z;
                    const float g2 = gu*gu + gv*gv;
                    if (g2 < 1e-14f) break;        // flat: any point works
                    // gradient descent with normalized step (f is ~linear:
                    // minimum sits on the boundary; project each step)
                    const float step = 0.25f;
                    u = std::fmin(1.f, std::fmax(0.f, u - step*gu/std::sqrt(g2)));
                    v = std::fmin(1.f, std::fmax(0.f, v - step*gv/std::sqrt(g2)));
                }
                Vec3 q = P[pid].Evaluate(u, v);
                const float depth = dl - (q.x*nl.x + q.y*nl.y + q.z*nl.z);
                if (depth <= 0) continue;
                const Vec3 wq = qrot(xf.rotation, q);
                cand.push_back({{wq.x+xf.position.x, wq.y+xf.position.y,
                                 wq.z+xf.position.z}, n, depth, pid, u, v});
            }
    }
    // keep the deepest, spatially deduped
    std::sort(cand.begin(), cand.end(),
              [](const Contact& a, const Contact& b){ return a.depth > b.depth; });
    const float dd = 0.04f * std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    for (const Contact& c : cand) {
        bool dup = false;
        for (const Contact& k : out) {
            const float dx=c.point.x-k.point.x, dy=c.point.y-k.point.y,
                        dz=c.point.z-k.point.z;
            if (dx*dx+dy*dy+dz*dz < dd*dd) { dup = true; break; }
        }
        if (!dup) out.push_back(c);
        if ((int)out.size() >= maxc) break;
    }
}

bool CollisionAnalytic::Raycast(const PSFBody& body, const Transform& xf,
                                const Vec3& o, const Vec3& dir, Contact& out) {
    const Vec3 ol = qrot_inv(xf.rotation,
        {o.x-xf.position.x, o.y-xf.position.y, o.z-xf.position.z});
    const Vec3 dl = qrot_inv(xf.rotation, dir);
    SurfaceHit h;
    if (!body.Raycast(ol, dl, h)) return false;
    const Vec3 wp = qrot(xf.rotation, h.p);
    out.point = {wp.x+xf.position.x, wp.y+xf.position.y, wp.z+xf.position.z};
    out.normal = qrot(xf.rotation,
        body.Patches()[h.patch].EvaluateNormal(h.u, h.v));
    out.depth = 0; out.patch = h.patch; out.u = h.u; out.v = h.v;
    return true;
}

} // namespace DeliRium
