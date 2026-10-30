// =============================================================================
// LocalFEM.cpp - Near/expensive scale: a dense local FEM solved ON the
// patch basis (isogeometric: the patches ARE the elements). Shares the
// element matrices with the modal tool (corner hats + 8 deformation
// coefficients per patch, anchors coupling neighbors). Boundary DOFs of
// the region are CLAMPED (the rest of the body holds the region in place),
// interior DOFs evolve under an implicit Newmark-beta step -- so it is
// unconditionally stable at the small step sizes gameplay uses.
// =============================================================================
#include "delirium/physics/LocalFEM.h"
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstdio>

namespace DeliRium {

namespace {
struct Elem { double K[12][12]; double M[12][12]; bool ready=false; } g_elem;

void corner_hats(double u, double v, double w[4]) {
    w[0]=(1-u)*(1-v); w[1]=(1-u)*v; w[2]=u*v; w[3]=u*(1-v);
}
void corner_hats_du(double u, double v, double w[4]) {
    w[0]=-(1-v); w[1]=-v; w[2]=v; w[3]=(1-v);
}
void corner_hats_dv(double u, double v, double w[4]) {
    w[0]=-(1-u); w[1]=(1-u); w[2]=u; w[3]=-u;
}
void cv_basis(double u, double v, double B[8]) {
    const double bu=u*(1-u), bv=v*(1-v), au=2*u-1, av=2*v-1;
    B[0]=bu*(1-v); B[1]=bu*v; B[2]=bv*(1-u); B[3]=bv*u;
    B[4]=bu*au*(1-v); B[5]=bu*au*v; B[6]=bv*av*(1-u); B[7]=bv*av*u;
}
void cv_basis_du(double u, double v, double B[8]) {
    const double bv=v*(1-v), au=2*u-1; const double dbu=1-2*u, dcu=dbu*au+2*u*(1-u);
    B[0]=dbu*(1-v); B[1]=dbu*v; B[2]=-bv; B[3]=bv;
    B[4]=dcu*(1-v); B[5]=dcu*v; B[6]=-bv*au; B[7]=bv*au;  // av term folded
    const double av=2*v-1; B[6]=-bv*av; B[7]=bv*av;
}
void cv_basis_dv(double u, double v, double B[8]) {
    const double bu=u*(1-u), av=2*v-1; const double dbv=1-2*v, dcv=dbv*av+2*v*(1-v);
    B[0]=-bu; B[1]=bu; B[2]=dbv*(1-u); B[3]=dbv*u;
    const double au=2*u-1;
    B[4]=-bu*au; B[5]=bu*au; B[6]=dcv*(1-u); B[7]=dcv*u;
}

void build_elem() {
    if (g_elem.ready) return;
    const double gp[4]={0.0694318442,0.3300094782,0.6699905218,0.9305681558};
    const double gw[4]={0.1739274226,0.3260725774,0.3260725774,0.1739274226};
    for (int i=0;i<12;++i) for(int j=0;j<12;++j){g_elem.K[i][j]=0;g_elem.M[i][j]=0;}
    for (int gi=0; gi<4; ++gi)
        for (int gj=0; gj<4; ++gj) {
            const double u=gp[gi], v=gp[gj], w=gw[gi]*gw[gj];
            double hb[4],hdu[4],hdv[4],B[8],Bdu[8],Bdv[8];
            corner_hats(u,v,hb); corner_hats_du(u,v,hdu); corner_hats_dv(u,v,hdv);
            cv_basis(u,v,B); cv_basis_du(u,v,Bdu); cv_basis_dv(u,v,Bdv);
            double val[12],du[12],dv[12];
            for(int k=0;k<4;++k){val[k]=hb[k];du[k]=hdu[k];dv[k]=hdv[k];}
            for(int k=0;k<8;++k){val[4+k]=B[k];du[4+k]=Bdu[k];dv[4+k]=Bdv[k];}
            for(int a=0;a<12;++a)for(int b=0;b<12;++b){
                g_elem.K[a][b]+=w*(du[a]*du[b]+dv[a]*dv[b]);
                g_elem.M[a][b]+=w*val[a]*val[b];
            }
        }
    g_elem.ready=true;
}

// One-time LU factorization with partial pivoting, stored in compact form
// (unit-L multipliers in the lower triangle, U in the upper). Aeff is
// constant across steps (linear FEM, fixed dt, K/M assembled once), so we
// factor ONCE and reuse; each Step then costs O(n^2) substitution instead
// of O(n^3) elimination. Mathematically identical to dense_solve -- same
// pivot, same decomposition -- it just stops redoing the factorization.
bool lu_factor(const std::vector<double>& A, int n,
               std::vector<double>& LU, std::vector<int>& piv) {
    LU = A; piv.resize(n);
    for (int c=0;c<n;++c) piv[c]=c;
    for (int c=0;c<n;++c){
        int p=c; for(int r=c+1;r<n;++r) if(std::fabs(LU[r*n+c])>std::fabs(LU[p*n+c])) p=r;
        if(std::fabs(LU[p*n+c])<1e-14) return false;
        if(p!=c){for(int k=0;k<n;++k) std::swap(LU[c*n+k],LU[p*n+k]); std::swap(piv[c],piv[p]);}
        for(int r=c+1;r<n;++r){const double f=LU[r*n+c]/LU[c*n+c];
            LU[r*n+c]=f;                       // store L multiplier
            for(int k=c+1;k<n;++k) LU[r*n+k]-=f*LU[c*n+k];}
    }
    return true;
}
void lu_solve(const std::vector<double>& LU, const std::vector<int>& piv,
              const std::vector<double>& b, std::vector<double>& x, int n) {
    for(int i=0;i<n;++i) x[i]=b[piv[i]];       // apply row permutation
    for(int i=0;i<n;++i){double s=x[i];        // forward sub (unit L)
        for(int k=0;k<i;++k) s-=LU[i*n+k]*x[k]; x[i]=s;}
    for(int i=n-1;i>=0;--i){double s=x[i];     // back sub (U)
        for(int k=i+1;k<n;++k) s-=LU[i*n+k]*x[k]; x[i]=s/LU[i*n+i];}
}
} // namespace

struct LocalFEM::Impl {
    PSFBody* body=nullptr;
    std::vector<uint32_t> patches;
    std::vector<uint16_t> anchors;
    std::unordered_map<uint16_t,int> anchor_dof;
    std::vector<uint8_t> clamped;
    int n_dof=0;
    std::vector<double> Kg, Mg;
    std::vector<double> x, vel, x0;
    std::vector<Vec3> anchor_rest, anchor_nrm;
    std::vector<double> coef_rest;
    float zeta=0.04f;
    // Cached factorization of Aeff = M + dt*beta*K + dt^2*K (constant for
    // a given dt: K/M are assembled once, the solver is linear).
    std::vector<double> lu; std::vector<int> piv;
    double lu_dt=-1.0, lu_beta=-1.0;
    bool lu_ok=false;
};

LocalFEM::LocalFEM() : impl(new Impl) {}
LocalFEM::~LocalFEM() { delete impl; }

void LocalFEM::BindRegion(PSFBody& body, const Vec3& center, float radius) {
    build_elem();
    Impl& I = *impl; I.body=&body; I.patches.clear(); I.anchors.clear();
    I.anchor_dof.clear();
    auto& pool = body.AnchorsConst();
    auto patch_center=[&](uint32_t pid){
        const SurfaceRecord& s=body.SurfaceBlob()[pid]; Vec3 c{};
        for(int k=0;k<4;++k){Vec3 a=pool.Position(s.anchor_idx[k]);
            c.x+=a.x*0.25f;c.y+=a.y*0.25f;c.z+=a.z*0.25f;} return c; };
    std::unordered_set<uint16_t> in_region;
    for (uint32_t pid=0; pid<body.SurfaceBlob().size(); ++pid) {
        const Vec3 c=patch_center(pid);
        const float d=std::sqrt((c.x-center.x)*(c.x-center.x)
            +(c.y-center.y)*(c.y-center.y)+(c.z-center.z)*(c.z-center.z));
        if (d<=radius) {
            I.patches.push_back(pid);
            for(int k=0;k<4;++k) in_region.insert(body.SurfaceBlob()[pid].anchor_idx[k]);
        }
    }
    for (uint16_t a : in_region) { I.anchor_dof[a]=(int)I.anchors.size();
        I.anchors.push_back(a); }
    const int na=(int)I.anchors.size();
    I.n_dof = na + 8*(int)I.patches.size();

    std::unordered_set<uint32_t> region_set(I.patches.begin(), I.patches.end());
    std::vector<int> ext_ref(pool.Count(),0);
    for (uint32_t pid=0; pid<body.SurfaceBlob().size(); ++pid) {
        if (region_set.count(pid)) continue;
        for(int k=0;k<4;++k) ext_ref[body.SurfaceBlob()[pid].anchor_idx[k]]++;
    }
    I.clamped.assign(I.n_dof, 0);
    for (int i=0;i<na;++i) if (ext_ref[I.anchors[i]]>0) I.clamped[i]=1;

    I.anchor_rest.resize(na); I.anchor_nrm.resize(na);
    for (int i=0;i<na;++i){ I.anchor_rest[i]=pool.Position(I.anchors[i]);
        I.anchor_nrm[i]=pool.Normal(I.anchors[i]); }
    I.x.assign(I.n_dof,0.0); I.vel.assign(I.n_dof,0.0);
    I.x0.assign(I.n_dof,0.0);
    I.coef_rest.assign(8*I.patches.size(), 0.0);
    for (size_t e=0;e<I.patches.size();++e)
        for(int c=0;c<8;++c)
            I.coef_rest[8*e+c]=body.Patches()[I.patches[e]].GetCoefficient(c);
}

void LocalFEM::AssembleSystem(const Material& m) {
    Impl& I=*impl; const int n=I.n_dof;
    I.Kg.assign(size_t(n)*n,0.0); I.Mg.assign(size_t(n)*n,0.0);
    auto& pool=I.body->AnchorsConst();
    // Real stress-strain curve, if the material provides one: use its
    // INITIAL (zero-strain) tangent modulus. Honest scope note: Step()
    // below reuses this same Kg every frame (linear FEM, constant
    // stiffness) -- so this is the correct value to use ONCE at assembly,
    // but does NOT make the material genuinely nonlinear under large
    // deformation. True strain-dependent stiffness would need Kg
    // reassembled each step from the CURRENT accumulated strain (I.x) --
    // a separate, larger change to Step(), not done here.
    const float E = m.stress_strain_curve
                  ? m.stress_strain_curve->InitialModulus()
                  : m.youngs_modulus;
    const float stiff = E * 1e-5f;   // scale to usable freqs (unchanged)
    const int na=(int)I.anchors.size();
    for (size_t e=0;e<I.patches.size();++e){
        const SurfaceRecord& s=I.body->SurfaceBlob()[I.patches[e]];
        int ld[12];
        for(int k=0;k<4;++k) ld[k]=I.anchor_dof[s.anchor_idx[k]];
        for(int c=0;c<8;++c) ld[4+c]=na+8*(int)e+c;
        Vec3 A[4]; for(int k=0;k<4;++k) A[k]=pool.Position(s.anchor_idx[k]);
        const float area=0.5f*(
            std::sqrt(std::pow((A[1].y-A[0].y)*(A[3].z-A[0].z)-(A[1].z-A[0].z)*(A[3].y-A[0].y),2.f)
                     +std::pow((A[1].z-A[0].z)*(A[3].x-A[0].x)-(A[1].x-A[0].x)*(A[3].z-A[0].z),2.f)
                     +std::pow((A[1].x-A[0].x)*(A[3].y-A[0].y)-(A[1].y-A[0].y)*(A[3].x-A[0].x),2.f)));
        const double ar=std::max((double)area,1e-9);
        for(int a=0;a<12;++a)for(int b=0;b<12;++b){
            I.Kg[size_t(ld[a])*n+ld[b]] += stiff*g_elem.K[a][b];
            I.Mg[size_t(ld[a])*n+ld[b]] += ar*g_elem.M[a][b];
        }
    }
    for(int i=0;i<n;++i) I.Mg[size_t(i)*n+i]+=1e-9;
}

void LocalFEM::Step(float dt) {
    Impl& I=*impl; const int n=I.n_dof;
    if (n==0) return;
    const double beta = 2.0*I.zeta/60.0;
    // Aeff = M + dt*beta*K + dt^2*K is constant for a fixed dt (K/M are
    // assembled once, linear solver). Factor it the first time we see this
    // (dt,beta) pair, then every step is O(n^2) substitution only.
    if (!I.lu_ok || I.lu_dt!=dt || I.lu_beta!=beta){
        std::vector<double> Aeff(size_t(n)*n);
        for (int i=0;i<n;++i)
            for(int j=0;j<n;++j)
                Aeff[size_t(i)*n+j] = I.Mg[size_t(i)*n+j]
                    + (dt*beta + dt*dt)*I.Kg[size_t(i)*n+j];
        for(int i=0;i<n;++i) if(I.clamped[i]){
            for(int j=0;j<n;++j){ Aeff[size_t(i)*n+j]=0; Aeff[size_t(j)*n+i]=0; }
            Aeff[size_t(i)*n+i]=1;
        }
        I.lu_ok = lu_factor(Aeff, n, I.lu, I.piv);
        I.lu_dt=dt; I.lu_beta=beta;
        if (!I.lu_ok) return;
    }
    std::vector<double> rhs(n);
    for (int i=0;i<n;++i){
        double mvn=0, kx=0;
        for(int j=0;j<n;++j){
            mvn += I.Mg[size_t(i)*n+j]*I.vel[j];
            kx  += I.Kg[size_t(i)*n+j]*I.x[j];
        }
        rhs[i] = I.clamped[i] ? 0.0 : (mvn - dt*kx);
    }
    std::vector<double> vnew(n);
    lu_solve(I.lu, I.piv, rhs, vnew, n);
    for(int i=0;i<n;++i){ if(I.clamped[i]){I.vel[i]=0; continue;}
        I.vel[i]=vnew[i]; I.x[i]+=dt*vnew[i]; }
}

void LocalFEM::WriteBack() {
    Impl& I=*impl; if(!I.body) return;
    const int na=(int)I.anchors.size();
    auto& pool=I.body->Anchors();
    for(int i=0;i<na;++i){ if(I.clamped[i]) continue;
        const Vec3 p=I.anchor_rest[i], nrm=I.anchor_nrm[i];
        const double d=I.x[i];
        pool.SetPosition(I.anchors[i], {float(p.x+d*nrm.x),
            float(p.y+d*nrm.y), float(p.z+d*nrm.z)});
    }
    auto& patches=const_cast<std::vector<Patch>&>(I.body->Patches());
    for(size_t e=0;e<I.patches.size();++e)
        for(int c=0;c<8;++c)
            patches[I.patches[e]].SetCoefficient(c,
                float(I.coef_rest[8*e+c] + I.x[na+8*e+c]));
    I.body->MarkAnchorsDirty();
    std::vector<uint32_t> ids(I.patches.begin(), I.patches.end());
    I.body->Seams().Build(*I.body);
    I.body->Seams().RestitchAround(*I.body, ids);
}

float LocalFEM::EstimateError(uint32_t patch_id) const {
    const Impl& I=*impl; if(!I.body) return 0;
    return I.body->Seams().MaxGap(*I.body);
}

void LocalFEM::Excite(uint32_t patch_local_index, float amount) {
    Impl& I=*impl;
    if (patch_local_index>=I.patches.size()) return;
    const int na=(int)I.anchors.size();
    I.x[na+8*patch_local_index+0]+=amount;
}

int   LocalFEM::DofCount() const   { return impl->n_dof; }
int   LocalFEM::PatchCount() const { return (int)impl->patches.size(); }
float LocalFEM::Energy() const {
    const Impl& I=*impl; const int n=I.n_dof; double e=0;
    for(int i=0;i<n;++i){
        double kx=0, mv=0;
        for(int j=0;j<n;++j){ kx+=I.Kg[size_t(i)*n+j]*I.x[j];
                              mv+=I.Mg[size_t(i)*n+j]*I.vel[j]; }
        e += 0.5*I.x[i]*kx + 0.5*I.vel[i]*mv;
    }
    return (float)e;
}

} // namespace DeliRium
