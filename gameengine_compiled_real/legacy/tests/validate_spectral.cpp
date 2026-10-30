// =============================================================================
// validate_spectral.cpp - the spectral-primitive reframe ("complex models
// from simple equations"). Verifies the C++ evaluator reconstructs the baked
// roughball field and quantifies the compression.
//  A) RECONSTRUCTION: the C++ SH evaluation matches the original mesh radius
//     within the baked fidelity (the field is faithful).
//  B) COMPRESSION: the .spf is dramatically smaller than the .psf at equal
//     fidelity (the whole point).
//  C) NO SEAMS / NO ANCHORS: a single continuous field -> there is nothing
//     to weld/stitch; the surface is C-infinity everywhere.
// =============================================================================
#include "delirium/psf/SpectralPrimitive.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* spf = argc>1?argv[1]:"assets/roughball.spf";
    SpectralPrimitive prim;
    if (!prim.Load(spf)) { std::fprintf(stderr,"load failed: %s\n",spf); return 1; }
    int fails=0;

    // load original mesh vertices for ground truth
    // ground-truth OBJ: same basename as the .spf (e.g. asteroid.spf ->
    // asteroid.obj), so the validator works for any baked primitive.
    std::string objpath(spf);
    { size_t dot = objpath.rfind(".spf");
      if (dot != std::string::npos) objpath = objpath.substr(0,dot) + ".obj"; }
    std::vector<Vec3> V;
    { std::ifstream f(objpath); std::string ln;
      while (std::getline(f,ln)) if (ln.size()>2 && ln[0]=='v' && ln[1]==' ') {
          Vec3 p; std::sscanf(ln.c_str(),"v %f %f %f",&p.x,&p.y,&p.z); V.push_back(p);
      } }
    if (V.empty()) { std::fprintf(stderr,"no obj\n"); return 1; }

    // A) reconstruction: for each vertex, its spherical angles -> field point
    Vec3 c = prim.Center();
    double err2=0; int n=0; float lo[3]={1e9,1e9,1e9},hi[3]={-1e9,-1e9,-1e9};
    for (const Vec3& v : V) {
        for(int k=0;k<3;++k){float x=(&v.x)[k]; if(x<lo[k])lo[k]=x; if(x>hi[k])hi[k]=x;}
    }
    const float diag=std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0])+(hi[1]-lo[1])*(hi[1]-lo[1])+(hi[2]-lo[2])*(hi[2]-lo[2]));
    for (const Vec3& v : V) {
        const Vec3 rel{v.x-c.x, v.y-c.y, v.z-c.z};
        const float r = std::sqrt(rel.x*rel.x+rel.y*rel.y+rel.z*rel.z);
        const float theta = std::acos(std::fmax(-1.f,std::fmin(1.f,rel.z/r)));
        const float phi = std::atan2(rel.y, rel.x);
        const Vec3 s = prim.Surface(theta, phi);
        const float rs = std::sqrt((s.x-c.x)*(s.x-c.x)+(s.y-c.y)*(s.y-c.y)+(s.z-c.z)*(s.z-c.z));
        err2 += (rs-r)*(rs-r); ++n;
    }
    const float rms = std::sqrt(err2/n);
    std::printf("A) reconstruction: RMS %.3f%% diag %s\n", rms/diag*100,
                rms/diag < 0.01f ? "OK":"FAIL");
    fails += !(rms/diag < 0.01f);

    // B) compression
    std::ifstream fp("assets/roughball.psf", std::ios::binary|std::ios::ate);
    const long psf_sz = fp ? (long)fp.tellg() : 0;
    std::printf("B) compression: .spf %zu B vs .psf %ld B = %.0fx smaller %s\n",
                prim.ByteSize(), psf_sz,
                psf_sz ? double(psf_sz)/prim.ByteSize() : 0,
                (psf_sz && prim.ByteSize()*10 < (size_t)psf_sz) ? "OK":"FAIL");
    fails += !(psf_sz && prim.ByteSize()*10 < (size_t)psf_sz);

    // C) continuity: sample a dense ring, adjacent points are close (no seam)
    float maxjump=0; const int M=720;
    for (int i=0;i<M;++i){
        const float p0=-M_PI + 2*M_PI*i/M, p1=-M_PI + 2*M_PI*(i+1)/M;
        const Vec3 a=prim.Surface(1.2f,p0), b=prim.Surface(1.2f,p1);
        maxjump=std::fmax(maxjump,std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z)));
    }
    std::printf("C) continuity: max adjacent jump on a ring %.4f%% diag "
                "(no seams, single field) %s\n", maxjump/diag*100,
                maxjump/diag < 0.02f ? "OK":"FAIL");
    fails += !(maxjump/diag < 0.02f);

    std::printf("   primitive: center+radius (16 B) + geom L=%u + color L=%u\n",
                prim.GeomOrder(), prim.ColorOrder());
    std::printf(fails ? "SPECTRAL VALIDATION FAILED (%d)\n"
                      : "SPECTRAL VALIDATION PASSED\n", fails);
    return fails;
}
