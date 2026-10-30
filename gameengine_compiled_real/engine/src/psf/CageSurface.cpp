// =============================================================================
// CageSurface.cpp - anchor-cage surface (the reframe).
// =============================================================================
#include "delirium/psf/CageSurface.h"
#include <cmath>
#include <unordered_set>

namespace DeliRium {

static inline Vec3 mix3(const Vec3& a, const Vec3& b, float t) {
    return Vec3{a.x+(b.x-a.x)*t, a.y+(b.y-a.y)*t, a.z+(b.z-a.z)*t};
}
static inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
static inline float len3(const Vec3& a){ return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
static inline Vec3 norm3(const Vec3& a){ float l=len3(a); return l>1e-9f?Vec3{a.x/l,a.y/l,a.z/l}:Vec3{0,0,1}; }

CageSurface::CageSurface(const PSFBody& body) : body_(body) { BuildCage(); }

void CageSurface::BuildCage() {
    const auto& S = body_.SurfaceBlob();
    const uint32_t na = (uint32_t)body_.AnchorBlob().size();
    auto& pool = const_cast<PSFBody&>(body_).Anchors();

    // anchor adjacency from patch edges
    std::vector<std::vector<uint16_t>> nbr(na);
    auto link = [&](uint16_t a, uint16_t b){
        if (a==b) return;
        for (uint16_t x : nbr[a]) if (x==b) return;
        nbr[a].push_back(b); nbr[b].push_back(a);
    };
    for (const auto& s : S) {
        const uint16_t* a = s.anchor_idx;
        link(a[0],a[1]); link(a[1],a[2]); link(a[2],a[3]); link(a[3],a[0]);
    }

    // derived normal per anchor: sum of cross products of consecutive
    // neighbor edges (a discrete surface normal from the cage)
    anchor_normal_.assign(na, Vec3{0,0,1});
    anchor_bulge_.assign(na, 0.0f);
    for (uint16_t a = 0; a < na; ++a) {
        const Vec3 p = pool.Position(a);
        const auto& ns = nbr[a];
        if (ns.size() < 3) { anchor_normal_[a] = pool.Normal(a); continue; }
        Vec3 nrm{0,0,0};
        for (size_t i = 0; i < ns.size(); ++i) {
            const Vec3 e1{pool.Position(ns[i]).x-p.x, pool.Position(ns[i]).y-p.y,
                          pool.Position(ns[i]).z-p.z};
            const Vec3 e2{pool.Position(ns[(i+1)%ns.size()]).x-p.x,
                          pool.Position(ns[(i+1)%ns.size()]).y-p.y,
                          pool.Position(ns[(i+1)%ns.size()]).z-p.z};
            const Vec3 c = cross3(e1, e2);
            nrm.x+=c.x; nrm.y+=c.y; nrm.z+=c.z;
        }
        Vec3 dn = norm3(nrm);
        // orient outward using the stored normal as a hint
        const Vec3 hint = pool.Normal(a);
        if (dn.x*hint.x+dn.y*hint.y+dn.z*hint.z < 0) { dn.x=-dn.x; dn.y=-dn.y; dn.z=-dn.z; }
        anchor_normal_[a] = dn;
        // bulge = signed distance of this anchor from the mean of neighbors,
        // along the derived normal (discrete mean curvature proxy)
        Vec3 mean{0,0,0};
        for (uint16_t x : ns) { mean.x+=pool.Position(x).x; mean.y+=pool.Position(x).y; mean.z+=pool.Position(x).z; }
        mean.x/=ns.size(); mean.y/=ns.size(); mean.z/=ns.size();
        anchor_bulge_[a] = (p.x-mean.x)*dn.x + (p.y-mean.y)*dn.y + (p.z-mean.z)*dn.z;
    }
}

Vec3 CageSurface::Evaluate(uint32_t s, float u, float v) const {
    const uint16_t* a = body_.SurfaceBlob()[s].anchor_idx;
    auto& pool = const_cast<PSFBody&>(body_).Anchors();
    const Vec3 A0=pool.Position(a[0]), A1=pool.Position(a[1]),
               A2=pool.Position(a[2]), A3=pool.Position(a[3]);
    // bilinear base of the 4 corner anchors
    const Vec3 base = mix3(mix3(A0,A3,u), mix3(A1,A2,u), v);
    // derived normal interpolated
    const Vec3 n = norm3(mix3(mix3(anchor_normal_[a[0]],anchor_normal_[a[3]],u),
                              mix3(anchor_normal_[a[1]],anchor_normal_[a[2]],u), v));
    // fullness from cage curvature, interpolated and windowed so it VANISHES
    // at edges (seam continuity) and peaks at center
    const float b = anchor_bulge_[a[0]]*(1-u)*(1-v) + anchor_bulge_[a[3]]*u*(1-v)
                  + anchor_bulge_[a[1]]*(1-u)*v     + anchor_bulge_[a[2]]*u*v;
    const float w = 16.0f * u*(1-u)*v*(1-v);   // 0 at edges, 1 at center
    const float h = -0.5f * b * w;             // reconstruct the bulge
    return Vec3{base.x+n.x*h, base.y+n.y*h, base.z+n.z*h};
}

Vec3 CageSurface::EvaluateNormal(uint32_t s, float u, float v) const {
    const float e = 1e-3f;
    const Vec3 pu0=Evaluate(s,std::fmax(0,u-e),v), pu1=Evaluate(s,std::fmin(1,u+e),v);
    const Vec3 pv0=Evaluate(s,u,std::fmax(0,v-e)), pv1=Evaluate(s,u,std::fmin(1,v+e));
    const Vec3 du{pu1.x-pu0.x,pu1.y-pu0.y,pu1.z-pu0.z};
    const Vec3 dv{pv1.x-pv0.x,pv1.y-pv0.y,pv1.z-pv0.z};
    return norm3(cross3(du,dv));
}

float CageSurface::MaxSeamGap() const {
    // enumerate shared edges by anchor-pair key; both patches evaluate it
    const auto& S = body_.SurfaceBlob();
    std::unordered_map<uint64_t, std::vector<std::pair<uint32_t,int>>> edges;
    static const int EC[4][2] = {{0,3},{1,2},{0,1},{3,2}};
    for (uint32_t s = 0; s < S.size(); ++s)
        for (int e = 0; e < 4; ++e) {
            uint16_t c0=S[s].anchor_idx[EC[e][0]], c1=S[s].anchor_idx[EC[e][1]];
            uint64_t key = c0<c1 ? (uint64_t(c0)<<16|c1) : (uint64_t(c1)<<16|c0);
            edges[key].push_back({s,e});
        }
    auto euv=[](int e,float t,float&u,float&v){switch(e){case 0:u=t;v=0;break;
        case 1:u=t;v=1;break;case 2:u=0;v=t;break;default:u=1;v=t;}};
    const Vec3 sp=body_.BBoxSpan(); const float diag=std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    float gmax=0;
    for (auto& kv : edges) {
        if (kv.second.size()!=2) continue;
        auto [sa,ea]=kv.second[0]; auto [sb,eb]=kv.second[1];
        // shared endpoints -> determine if edge b runs same or flipped
        uint16_t a0=body_.SurfaceBlob()[sa].anchor_idx[EC[ea][0]];
        uint16_t b0=body_.SurfaceBlob()[sb].anchor_idx[EC[eb][0]];
        bool flip = (a0 != b0);
        for (float t=0.1f;t<0.95f;t+=0.2f){
            float ua,va,ub,vb; euv(ea,t,ua,va); euv(eb,flip?1-t:t,ub,vb);
            Vec3 pa=Evaluate(sa,ua,va), pb=Evaluate(sb,ub,vb);
            gmax=std::fmax(gmax,len3(Vec3{pa.x-pb.x,pa.y-pb.y,pa.z-pb.z}));
        }
    }
    return gmax/diag;
}

float CageSurface::FidelityVsCurrent() const {
    const auto& P = body_.Patches();
    double err=0; int n=0;
    for (uint32_t s=0;s<P.size();s+=3)
        for (int i=0;i<=4;++i)for(int j=0;j<=4;++j){
            const float u=i/4.f,v=j/4.f;
            const Vec3 a=Evaluate(s,u,v), b=P[s].Evaluate(u,v);
            err+=(a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z); ++n;
        }
    const Vec3 sp=body_.BBoxSpan(); const float diag=std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    return std::sqrt(err/n)/diag;
}

Vec3 CageSurface::CornerBulgeDir(uint16_t a) const { return anchor_normal_[a]; }

} // namespace DeliRium
