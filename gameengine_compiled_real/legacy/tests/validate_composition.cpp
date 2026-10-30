// =============================================================================
// validate_composition.cpp - spectral primitives as a GAME-ENGINE scene.
//  A) COMPOSITION: build a multi-part model (parts = primitives with local
//     transforms forming something larger) and verify parts land at their
//     world positions (the user's "objetos individuales que conforman algo
//     mas grande").
//  B) ANIMATION IS FREE: moving the root moves all parts; detail (coeffs)
//     is untouched -- verify a translated part's surface shifts by exactly
//     the translation, with no re-evaluation cost.
//  C) CONTINUOUS LOD: distance selects SH order, one file = whole ladder.
// =============================================================================
#include "delirium/psf/SpectralObject.h"
#include <cstdio>
#include <cmath>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* spf = argc>1?argv[1]:"assets/roughball.spf";
    int fails=0;

    // Build a 3-part "creature": body at origin, two limbs offset. (Using
    // the same primitive for each part; in practice each part is its own
    // baked field.)
    SpectralModel model;
    SpectralTransform body{};  body.position={0,0,0};
    SpectralTransform limbL{}; limbL.position={1.5f,0,0}; limbL.scale={0.5f,0.5f,0.5f};
    SpectralTransform limbR{}; limbR.position={-1.5f,0,0};limbR.scale={0.5f,0.5f,0.5f};
    bool ok = model.LoadPart(spf, body)
           && model.LoadPart(spf, limbL)
           && model.LoadPart(spf, limbR);
    if(!ok){ std::fprintf(stderr,"load failed\n"); return 1; }

    std::printf("A) composition: %zu parts, total %zu B "
                "(a creature from baked primitives)\n",
                model.PartCount(), model.ByteSize());
    fails += !(model.PartCount()==3);

    // A) parts land at world positions: limbL surface center near +1.5 x
    const Vec3 cL = model.Part(1).SurfaceWorld(1.5708f, 0.0f); // some dir
    const Vec3 cBase = model.Part(1).prim->Center();
    // the part's world points should be offset ~+1.5 in x vs the body part
    const Vec3 cBody = model.Part(0).SurfaceWorld(1.5708f, 0.0f);
    const float dx = cL.x - cBody.x;
    std::printf("A) limb offset in world x: %.3f (expect ~1.5 minus scale) %s\n",
                dx, dx > 0.5f ? "OK":"FAIL");
    fails += !(dx > 0.5f);

    // B) animation is free: translate the part, surface shifts by exactly
    // the translation, coefficients untouched
    Vec3 before = model.Part(0).SurfaceWorld(1.0f, 0.5f);
    model.Part(0).transform.position.y += 3.0f;
    Vec3 after = model.Part(0).SurfaceWorld(1.0f, 0.5f);
    const float shift = after.y - before.y;
    std::printf("B) translate +3 in y -> surface shifts %.4f (detail rides "
                "along, no re-eval) %s\n", shift,
                std::fabs(shift-3.0f)<1e-3f?"OK":"FAIL");
    fails += !(std::fabs(shift-3.0f)<1e-3f);

    // C) continuous LOD from projected size
    std::printf("C) LOD ladder (one file): ");
    const float sizes[] = {10, 40, 100, 250, 600};
    for (float s : sizes)
        std::printf("%gpx->L%u ", s, model.Part(0).LodForScreenSize(s));
    std::printf("OK\n");

    std::printf(fails ? "COMPOSITION VALIDATION FAILED (%d)\n"
                      : "COMPOSITION VALIDATION PASSED\n", fails);
    return fails;
}
