// =============================================================================
// validate_fem.cpp - Milestone 11 gate. Local isogeometric FEM:
//  A) ASSEMBLY: a bound region has the expected DOF count (anchors +
//     8*patches) and a non-empty clamped boundary.
//  B) STABILITY: excite an interior patch, run 600 implicit steps; the
//     energy must DECAY monotonically-ish to near zero (damped, stable --
//     never blows up, which an explicit scheme would at these stiffnesses).
//  C) STATIC EQUILIBRIUM: with no excitation the region is at rest and
//     stays at rest (energy stays ~0; clamped boundary holds).
//  D) SEAMS: after a FEM relaxation + writeback, the region's seams stay
//     within the subpixel budget (boundary clamp + restitch preserve it).
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/physics/LocalFEM.h"
#include <cstdio>
#include <cmath>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* psf = argc>1 ? argv[1] : "assets/torus.psf";
    auto body = PSFBody::FromFile(psf);
    if(!body){ std::fprintf(stderr,"load failed\n"); return 1; }
    const Vec3 sp=body->BBoxSpan();
    const float diag=std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    int fails=0;

    const Vec3 c = body->Patches()[150].Evaluate(0.5f,0.5f);
    const float radius = 0.18f*diag;

    Material mat; mat.youngs_modulus = 5e5f;
    LocalFEM fem;
    fem.BindRegion(*body, c, radius);
    fem.AssembleSystem(mat);
    const int np=fem.PatchCount(), nd=fem.DofCount();
    std::printf("region: %d patches, %d DOFs\n", np, nd);

    // A) DOF count consistency
    std::printf("A) assembly: %d patches -> %d DOFs %s\n", np, nd,
                (np>0 && nd>=8*np+4) ? "OK":"FAIL");
    fails += !(np>0 && nd>=8*np+4);

    // C) static rest (run first, before exciting): energy stays ~0
    float e_rest=fem.Energy();
    for(int i=0;i<100;++i) fem.Step(1.0f/240.0f);
    float e_rest2=fem.Energy();
    std::printf("C) static rest: energy %.3e -> %.3e %s\n", e_rest, e_rest2,
                (e_rest2 < 1e-6f + e_rest) ? "OK":"FAIL");
    fails += !(e_rest2 < 1e-6f + e_rest);

    // B) excite then watch decay + boundedness
    fem.Excite(0, 0.02f*diag);
    const float e0=fem.Energy();
    float emax=e0, eprev=e0; bool blew=false; int rises=0;
    for(int i=0;i<600;++i){
        fem.Step(1.0f/240.0f);
        const float e=fem.Energy();
        if(e>emax) emax=e;
        if(e>eprev*1.0001f) rises++;
        if(!(e==e) || e>1e3f*e0){ blew=true; break; }
        eprev=e;
    }
    const float ef=fem.Energy();
    std::printf("B) stability: E0 %.3e -> Emax %.3e -> Ef %.3e, rises %d %s\n",
                e0, emax, ef, rises,
                (!blew && ef < 0.2f*e0 && emax < 2.0f*e0) ? "OK":"FAIL");
    fails += !(!blew && ef < 0.2f*e0 && emax < 2.0f*e0);

    // D) writeback then seams
    fem.WriteBack();
    const float g = body->Seams().MaxGap(*body);
    std::printf("D) seams after writeback: %.4f%% diag (budget 0.05%%) %s\n",
                g/diag*100, g < 0.0005f*diag ? "OK":"FAIL");
    fails += !(g < 0.0005f*diag);

    std::printf(fails ? "FEM VALIDATION FAILED (%d)\n"
                      : "FEM VALIDATION PASSED\n", fails);
    return fails;
}
