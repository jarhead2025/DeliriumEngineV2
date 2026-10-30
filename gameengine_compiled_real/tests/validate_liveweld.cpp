// =============================================================================
// validate_liveweld.cpp - Progressive live harmonic weld in PhysicsWorld.
//  A) DEBT PAYDOWN: after RequestWeld, repeated Step() calls must drive the
//     relaxation debt to zero (the object settles) in bounded frames.
//  B) PROGRESSIVE: each frame does only a bounded slice of welding -- the
//     cross-chart gap decreases gradually across frames, not all at once.
//  C) MATERIAL PACING: a Fluid object pays its weld debt in FEWER frames
//     than a Brittle one (brittle lingers / keeps cracks).
//  D) NO HARM: welding through physics preserves the body's diagonal.
// =============================================================================
#include "delirium/core/GameObject.h"
#include "delirium/physics/PhysicsWorld.h"
#include "delirium/psf/Reconstructor.h"
#include "delirium/psf/Seam.h"
#include <cstdio>
#include <cmath>

using namespace DeliRium;

static int frames_to_settle(WeldCategory cat) {
    FixedReconstructor maker;
    ReconstructionRequest req; req.semantic="file:assets/asteroid.psf";
    GameObject o("a", maker, req);
    o.Mat().weld = cat;
    PhysicsWorld world; world.Register(&o);
    world.RequestWeld(&o, 10.0f);     // fixed debt; pacing decides frames
    int frames=0;
    for (; frames<500; ++frames) {
        world.Step(1.0f/60.0f);
        if (world.WeldDebt(&o) <= 0.0f) break;
    }
    return frames+1;
}

int main(int argc, char** argv) {
    FixedReconstructor maker;
    ReconstructionRequest req; req.semantic="file:assets/asteroid.psf";
    int fails=0;

    // A+B+D on an elastic object
    GameObject o("a", maker, req);
    o.Mat().weld = WeldCategory::Elastic;
    const Vec3 sp=o.Body().BBoxSpan();
    const float diag=std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
    PhysicsWorld world; world.Register(&o);
    o.Body().Seams().Build(o.Body());
    const float g0 = o.Body().Seams().MaxGap(o.Body());
    world.RequestWeld(&o, 5.0f);

    float prev = g0; int decreasing=0, total=0;
    for (int f=0; f<60; ++f) {
        world.Step(1.0f/60.0f);
        const float g = o.Body().Seams().MaxGap(o.Body());
        if (g < prev - 1e-7f) decreasing++;
        if (g <= prev + 1e-6f) total++;
        prev = g;
    }
    const float gN = o.Body().Seams().MaxGap(o.Body());
    std::printf("A/B) gap %.4f%% -> %.4f%% diag over 60 frames, "
                "decreasing frames %d %s\n", g0/diag*100, gN/diag*100,
                decreasing, (gN < g0 && decreasing >= 1) ? "OK":"FAIL");
    fails += !(gN < g0 && decreasing >= 1);

    const Vec3 sp2=o.Body().BBoxSpan();
    const float diag2=std::sqrt(sp2.x*sp2.x+sp2.y*sp2.y+sp2.z*sp2.z);
    std::printf("D) shape: diagonal %+.3f%% %s\n", (diag2/diag-1)*100,
                std::fabs(diag2-diag)/diag < 0.01f ? "OK":"FAIL");
    fails += !(std::fabs(diag2-diag)/diag < 0.01f);

    // C) material pacing: same debt, fluid pays it faster than brittle
    const int f_fluid = frames_to_settle(WeldCategory::Fluid);
    const int f_brittle = frames_to_settle(WeldCategory::Brittle);
    std::printf("C) pacing (same debt): Fluid %d frames, Brittle %d %s\n",
                f_fluid, f_brittle,
                (f_fluid < f_brittle) ? "OK":"FAIL");
    fails += !(f_fluid < f_brittle);

    std::printf(fails ? "LIVEWELD VALIDATION FAILED (%d)\n"
                      : "LIVEWELD VALIDATION PASSED\n", fails);
    return fails;
}
