// =============================================================================
// psf_harmonize - post-process a .psf so shared-edge deformation
// coefficients agree (closes index-shared seam cracks to ~0), optionally
// runs the harmonic weld, and re-saves. The exporter fits each chart leaf's
// deformation independently, leaving shared edges bowed differently; this
// fixes that at the record level so the asset loads crack-free.
//   usage: psf_harmonize in.psf out.psf [--weld]
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/Seam.h"
#include "delirium/io/PSFFile.h"
#include "delirium/physics/Material.h"
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace DeliRium;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr,
        "usage: psf_harmonize in.psf out.psf [--weld]\n"); return 1; }
    const bool do_weld = (argc > 3 && std::strcmp(argv[3], "--weld") == 0);

    auto body = PSFBody::FromFile(argv[1]);
    if (!body) { std::fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    const Vec3 sp = body->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);

    body->Seams().Build(*body);
    std::printf("max seam gap loaded:        %.3f%% diag\n",
                body->Seams().MaxGap(*body)/diag*100);

    const float g1 = body->Seams().HarmonizeEdgeCoefs(*body);
    std::printf("after edge-coef harmonize:  %.3f%% diag\n", g1/diag*100);

    if (do_weld) {
        body->Seams().Build(*body);
        const float g2 = body->Seams().HarmonicWeld(*body,
                            WeldProfileFor(WeldCategory::Elastic));
        std::printf("after harmonic weld:        %.3f%% diag\n", g2/diag*100);
    }
    body->Seams().Build(*body);
    body->Seams().FitAll(*body);
    std::printf("after final stitch:         %.3f%% diag\n",
                body->Seams().MaxGap(*body)/diag*100);

    // write back through PSFFile from the (modified) body blobs
    PSFFile out;
    out.version = 3;
    out.bbox_min = body->BBoxMin();
    out.bbox_span = body->BBoxSpan();
    out.s_scale = body->StitchScale();
    out.anchors = body->AnchorBlob();
    out.surfaces = body->SurfaceBlob();
    out.stitch = body->StitchBlob();
    if (!out.Save(argv[2])) { std::fprintf(stderr,"save failed\n"); return 1; }
    std::printf("wrote %s (%zu patches)\n", argv[2], out.surfaces.size());
    return 0;
}
