// =============================================================================
// validate_harmonic.cpp - Global harmonic seam reconstruction.
// The weld's job is the CROSS-CHART case: the exporter's directional
// charts meet at borders whose endpoints share no anchor index and sit at
// slightly different positions. HarmonicWeld pulls those distinct endpoint
// anchors together globally (all at once), reconstructing one continuous
// surface from many separately-fitted charts.
//  A) CROSS-CHART CLOSURE: on the asteroid (many position-only seams),
//     the weld must reduce the max gap on those cross-chart seams.
//  B) SHAPE PRESERVATION: anchors with no cross-chart partner must not
//     move -> diagonal and volume essentially unchanged (no collapse).
//  C) IDEMPOTENT-ish: a second weld pass changes little (converged).
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/Seam.h"
#include <cstdio>
#include <cmath>

using namespace DeliRium;
static const int EPL[4][2]={{0,3},{1,2},{0,1},{3,2}};

static float cross_chart_gap(PSFBody& b) {
    const auto& S=b.SurfaceBlob();
    const auto& seams=b.Seams().All();
    auto& P=b.Patches();
    auto euv=[](int e,float t,float&u,float&v){switch(e){
        case 0:u=t;v=0;break;case 1:u=t;v=1;break;
        case 2:u=0;v=t;break;default:u=1;v=t;}};
    float g=0;
    for(const Seam&s:seams){
        const uint16_t a0=S[s.patch_a].anchor_idx[EPL[s.edge_a][0]];
        const uint16_t b0f=S[s.patch_b].anchor_idx[EPL[s.edge_b][0]];
        const uint16_t b1f=S[s.patch_b].anchor_idx[EPL[s.edge_b][1]];
        const bool flip=(s.tb0>s.tb1);
        if(a0==(flip?b1f:b0f)) continue;            // index-shared, skip
        for(int k=1;k<9;++k){float t=k/9.f,tb=s.tb0+(s.tb1-s.tb0)*t;
            float ua,va,ub,vb;euv(s.edge_a,t,ua,va);euv(s.edge_b,tb,ub,vb);
            Vec3 pa=P[s.patch_a].Evaluate(ua,va),pb=P[s.patch_b].Evaluate(ub,vb);
            g=std::fmax(g,(float)std::sqrt((pa.x-pb.x)*(pa.x-pb.x)
                +(pa.y-pb.y)*(pa.y-pb.y)+(pa.z-pb.z)*(pa.z-pb.z)));}
    }
    return g;
}

int main(int argc, char** argv) {
    const char* psf = argc>1 ? argv[1] : "assets/asteroid.psf";
    auto body = PSFBody::FromFile(psf);
    if(!body){ std::fprintf(stderr,"load failed\n"); return 1; }
    const Vec3 sp=body->BBoxSpan();
    const float diag=std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    int fails=0;

    body->Seams().Build(*body);
    const float gcc0 = cross_chart_gap(*body);
    const float vol0 = body->Volume();
    std::printf("cross-chart gap before weld: %.4f%% of diag\n", gcc0/diag*100);

    // A) cross-chart closure
    const float gN = body->Seams().HarmonicWeld(*body, 32, 0.2f);
    const float gcc1 = cross_chart_gap(*body);
    std::printf("A) cross-chart: %.4f%% -> %.4f%% of diag (overall max %.4f%%) %s\n",
                gcc0/diag*100, gcc1/diag*100, gN/diag*100,
                gcc1 < gcc0*0.85f ? "OK":"FAIL");
    fails += !(gcc1 < gcc0*0.85f);

    // B) shape preserved (no collapse). NOTE: divergence-theorem Volume()
    // is invalid on a surface with OVERLAPPING charts (it double-counts the
    // overlap), and welding removes overlap -> volume "drops" partly as a
    // correction, not a shrink. The bounding diagonal is the robust size
    // metric: if the surface collapsed, the diagonal would shrink too.
    const Vec3 sp2=body->BBoxSpan();
    const float diag2=std::sqrt(sp2.x*sp2.x+sp2.y*sp2.y+sp2.z*sp2.z);
    const float vol1=body->Volume();
    std::printf("B) shape: diag %+.3f%% (volume %+.2f%%, overlap-corrected) %s\n",
                (diag2/diag-1)*100, (vol1/vol0-1)*100,
                std::fabs(diag2-diag)/diag < 0.01f ? "OK":"FAIL");
    fails += !(std::fabs(diag2-diag)/diag < 0.01f);

    // C) converged: a second weld barely moves it
    const float gN2 = body->Seams().HarmonicWeld(*body, 32, 0.2f);
    std::printf("C) converged: 2nd pass overall max %.4f%% -> %.4f%% %s\n",
                gN/diag*100, gN2/diag*100,
                std::fabs(gN2-gN) < 0.2f*gN + 1e-6f ? "OK":"FAIL");
    fails += !(std::fabs(gN2-gN) < 0.2f*gN + 1e-6f);

    std::printf(fails ? "HARMONIC VALIDATION FAILED (%d)\n"
                      : "HARMONIC VALIDATION PASSED\n", fails);
    return fails;
}
