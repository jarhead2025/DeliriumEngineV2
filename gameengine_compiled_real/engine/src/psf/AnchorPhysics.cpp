// =============================================================================
// AnchorPhysics.cpp
// =============================================================================
#include "delirium/psf/AnchorPhysics.h"
#include <cmath>
#include <algorithm>

namespace DeliRium {

static inline Vec3 sub(const Vec3&a,const Vec3&b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static inline Vec3 cross(const Vec3&a,const Vec3&b){
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};}
static inline float dot(const Vec3&a,const Vec3&b){return a.x*b.x+a.y*b.y+a.z*b.z;}

void AnchorPhysics::Initialize(const std::vector<Vec3>& rest,
        const std::vector<std::array<uint32_t,4>>& patch_anchors,
        float stiffness) {
    rest_ = rest; cur_ = rest;
    patch_anchors_ = patch_anchors;
    n_ = (int)rest.size();
    stiffness_ = stiffness;
    Factorize();
    rest_volume_ = Volume();
    BuildRestPenetration();
}

// patch plane (centroid, normal) from anchors
static void patch_plane(const std::array<uint32_t,4>& pa,
                        const std::vector<Vec3>& pos, Vec3& c, Vec3& nrm) {
    c = Vec3{0,0,0};
    for (int i=0;i<4;++i){ c.x+=pos[pa[i]].x; c.y+=pos[pa[i]].y; c.z+=pos[pa[i]].z; }
    c.x/=4; c.y/=4; c.z/=4;
    Vec3 d1=sub(pos[pa[3]],pos[pa[0]]), d2=sub(pos[pa[2]],pos[pa[1]]);
    nrm=cross(d1,d2); float nl=std::sqrt(dot(nrm,nrm));
    if (nl>1e-9f){nrm.x/=nl;nrm.y/=nl;nrm.z/=nl;}
}

// penetration of q's anchors into p's inner half-space, given positions
static float pair_penetration(const std::array<uint32_t,4>& p,
                              const std::array<uint32_t,4>& q,
                              const std::vector<Vec3>& pos) {
    Vec3 c,nrm; patch_plane(p,pos,c,nrm);
    float maxpen=-1e30f;
    for (int j=0;j<4;++j){
        float sd=dot(sub(pos[q[j]],c),nrm);
        maxpen=std::max(maxpen,-sd);
    }
    return maxpen;
}

void AnchorPhysics::BuildRestPenetration() {
    const size_t np=patch_anchors_.size();
    rest_pen_.assign(np*np, -1e30f);
    // median patch size for the lateral threshold
    std::vector<float> sizes;
    for (auto& pa : patch_anchors_) {
        Vec3 lo=rest_[pa[0]], hi=rest_[pa[0]];
        for (int i=1;i<4;++i){ const Vec3&v=rest_[pa[i]];
            lo.x=std::min(lo.x,v.x);lo.y=std::min(lo.y,v.y);lo.z=std::min(lo.z,v.z);
            hi.x=std::max(hi.x,v.x);hi.y=std::max(hi.y,v.y);hi.z=std::max(hi.z,v.z);}
        sizes.push_back(std::max({hi.x-lo.x,hi.y-lo.y,hi.z-lo.z}));
    }
    std::sort(sizes.begin(),sizes.end());
    patch_size_ = sizes.empty()?1.0f:sizes[sizes.size()/2];
    for (size_t i=0;i<np;++i)
        for (size_t j=0;j<np;++j) if (i!=j)
            rest_pen_[i*np+j]=pair_penetration(patch_anchors_[i],patch_anchors_[j],rest_);
}

// Build a sparse-graph Laplacian-like stiffness from shared anchors and
// factorize it once (dense Cholesky here; sparse in production). Anchors that
// share a patch are coupled. K = stiffness * (D - A) + small identity.
void AnchorPhysics::Factorize() {
    std::vector<float> K((size_t)n_*n_, 0.0f);
    std::vector<int> deg(n_, 0);
    for (auto& pa : patch_anchors_) {
        for (int i=0;i<4;++i) for (int j=0;j<4;++j) if (i!=j) {
            uint32_t a=pa[i], b=pa[j];
            if (a<(uint32_t)n_ && b<(uint32_t)n_) {
                K[(size_t)a*n_+b] -= stiffness_;
                deg[a] += 0;  // count via diagonal below
            }
        }
    }
    // diagonal = -sum of off-diagonal (graph Laplacian) + regularizer so it's
    // positive-definite (anchors can move but are tethered to rest).
    for (int a=0;a<n_;++a) {
        float off=0;
        for (int b=0;b<n_;++b) if (b!=a) off += -K[(size_t)a*n_+b];
        K[(size_t)a*n_+a] = off + 1.0f;   // +1 tether to rest
    }
    // Cholesky factorization K = L L^T (dense)
    L_.assign((size_t)n_*n_, 0.0f);
    for (int i=0;i<n_;++i) {
        for (int j=0;j<=i;++j) {
            float s = K[(size_t)i*n_+j];
            for (int k=0;k<j;++k) s -= L_[(size_t)i*n_+k]*L_[(size_t)j*n_+k];
            if (i==j) L_[(size_t)i*n_+j] = std::sqrt(std::max(s,1e-8f));
            else      L_[(size_t)i*n_+j] = s / L_[(size_t)j*n_+j];
        }
    }
}

const std::vector<Vec3>& AnchorPhysics::Step(const std::vector<Vec3>& forces) {
    // solve K u = f by back-substitution on the prefactored L L^T, per axis.
    auto solve_axis = [&](int axis){
        std::vector<float> b(n_), y(n_), x(n_);
        for (int i=0;i<n_;++i)
            b[i] = (axis==0?forces[i].x:axis==1?forces[i].y:forces[i].z);
        // forward: L y = b
        for (int i=0;i<n_;++i){ float s=b[i];
            for (int k=0;k<i;++k) s-=L_[(size_t)i*n_+k]*y[k];
            y[i]=s/L_[(size_t)i*n_+i]; }
        // backward: L^T x = y
        for (int i=n_-1;i>=0;--i){ float s=y[i];
            for (int k=i+1;k<n_;++k) s-=L_[(size_t)k*n_+i]*x[k];
            x[i]=s/L_[(size_t)i*n_+i]; }
        return x;
    };
    std::vector<float> ux=solve_axis(0), uy=solve_axis(1), uz=solve_axis(2);
    for (int i=0;i<n_;++i)
        cur_[i] = { rest_[i].x+ux[i], rest_[i].y+uy[i], rest_[i].z+uz[i] };

    // VOLUME CONSERVATION: scale displacement about the centroid so volume
    // returns to rest_volume_ (single scalar correction).
    float v = Volume();
    if (v > 1e-6f && rest_volume_ > 1e-6f) {
        float s = std::cbrt(rest_volume_ / v);   // uniform scale to fix volume
        // centroid
        Vec3 c{0,0,0};
        for (auto& a : cur_){ c.x+=a.x; c.y+=a.y; c.z+=a.z; }
        c.x/=n_; c.y/=n_; c.z/=n_;
        for (auto& a : cur_) {
            a.x = c.x + (a.x-c.x)*s;
            a.y = c.y + (a.y-c.y)*s;
            a.z = c.z + (a.z-c.z)*s;
        }
    }
    return cur_;
}

float AnchorPhysics::PatchFlux(size_t p, const std::vector<Vec3>& a) const {
    // flux contribution of the quad patch via divergence theorem. Normals
    // must point OUTWARD from the object centroid for the signed sum to give
    // the enclosed volume.
    const auto& pa = patch_anchors_[p];
    Vec3 v0=a[pa[0]], v1=a[pa[1]], v2=a[pa[2]], v3=a[pa[3]];
    // object centroid (for outward orientation)
    Vec3 oc{0,0,0};
    for (auto& an : a){ oc.x+=an.x; oc.y+=an.y; oc.z+=an.z; }
    oc.x/=a.size(); oc.y/=a.size(); oc.z/=a.size();
    auto tri_flux=[&](const Vec3&A,const Vec3&B,const Vec3&C){
        Vec3 n=cross(sub(B,A),sub(C,A));      // 2*area*normal
        Vec3 cen{(A.x+B.x+C.x)/3,(A.y+B.y+C.y)/3,(A.z+B.z+C.z)/3};
        // orient outward
        if (dot(n, sub(cen,oc)) < 0) { n.x=-n.x; n.y=-n.y; n.z=-n.z; }
        return dot(cen,n)/6.0f;
    };
    return tri_flux(v0,v1,v3) + tri_flux(v0,v3,v2);
}

float AnchorPhysics::Volume() const {
    float v=0;
    for (size_t p=0;p<patch_anchors_.size();++p) v += PatchFlux(p, cur_);
    return std::fabs(v);   // outward-oriented fluxes sum to enclosed volume
}

std::vector<uint8_t> AnchorPhysics::CrossCull(float tol) const {
    const size_t np = patch_anchors_.size();
    std::vector<uint8_t> culled(np, 0);
    // centroids now
    std::vector<Vec3> cen(np);
    for (size_t p=0;p<np;++p){ Vec3 c,n; patch_plane(patch_anchors_[p],cur_,c,n); cen[p]=c; }
    // A real cross = penetration INCREASED vs rest by more than tol*patch_size
    // AND the patches are laterally close (< patch_size). Absolute penetration
    // is meaningless on a closed curved body (opposite-side patches already
    // penetrate at rest); the CHANGE from rest is the true signal -- which is
    // also the user's point: we know how the deformation started.
    const float lat_thresh = patch_size_;
    const float pen_thresh = tol * patch_size_;
    for (size_t p=0;p<np;++p) {
        for (size_t q=0;q<np;++q) {
            if (p==q) continue;
            bool adjacent=false;
            for (int i=0;i<4&&!adjacent;++i) for (int j=0;j<4;++j)
                if (patch_anchors_[p][i]==patch_anchors_[q][j]){adjacent=true;break;}
            if (adjacent) continue;
            float pen_now = pair_penetration(patch_anchors_[p],patch_anchors_[q],cur_);
            float pen_rest = rest_pen_[p*np+q];
            float increase = pen_now - pen_rest;          // change from rest
            Vec3 lat=sub(cen[q],cen[p]);
            float latd=std::sqrt(dot(lat,lat));
            if (increase > pen_thresh && latd < lat_thresh) culled[q]=1;
        }
    }
    return culled;
}

} // namespace DeliRium
