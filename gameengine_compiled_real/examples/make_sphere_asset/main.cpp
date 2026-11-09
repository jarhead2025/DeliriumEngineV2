// FILEPATH: examples/make_sphere_asset/main.cpp
// Bake a CLOSED sphere and write it as a valid binary .psf asset.
//
// WHY THIS TOOL EXISTS (G18): `assets/sphere.psf` was a LEGACY TEXT file
// -- a human-readable description of a "SPHERICAL" parametric surface with
// `#` comments, a `PATCH 0 / TYPE SPHERICAL / COEFFS ...` block and six
// orientation anchors. The binary loader expects the v3 container
// ("PSF\0" magic, 64-byte header, 64-byte AnchorRecord[] and
// SurfaceRecord[], then an optional stitch section). The text file failed
// the magic check, so the gallery's scene N reported a MISLEADING
// "MISSING" message -- the asset was not missing, it was the wrong format.
//
// The sibling assets (torus.psf, asteroid.psf, roughball.psf) are all
// valid binaries; only sphere.psf was broken.
//
// WHY BAKED RATHER THAN HAND-BUILT: a hand-written sphere would have to
// get the bbox, the s_scale, the 16-bit position quantisation, the normal
// encoding and the corner-vanishing coefficients exactly right -- every
// one of them has a measured failure mode (a zero bbox_span collapses the
// mesh to a point; a zero s_scale makes every seam un-stitchable). The
// engine's own baker produces all of that correctly, and the sphere it
// produces is a CLOSED body (two mirrored sheets), not a dome.
//
// Usage:
//   make_sphere_asset [--out assets/sphere.psf] [--grid N] [--radius R]
//
// The tool writes the file and then READS IT BACK through the normal
// loader, so a success means the asset is loadable -- not merely written.
#include "delirium/ai/ElementSystem.h"
#include "delirium/io/PSFFile.h"
#include "delirium/psf/PSFBody.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace DeliRium;

namespace {

// A closed sphere = an upper ellipsoid + a lower one with the SAME origin
// and a NEGATIVE z scale. That is the measured closure rule (see
// docs/GOALS.md G7 and the closed-body reference): the negative scale
// flips the sheet down from the seam plane it shares with its parent, so
// the body spans exactly [-r, +r] with no gap and no inside-out shell.
std::unique_ptr<Element> BuildClosedSphere(float r) {
    auto root = std::make_unique<Element>();
    root->kind = "sphere";
    root->color = {0.25f, 0.55f, 0.95f};   // the legacy file's SOLID colour
    root->substance = std::make_unique<EllipsoidSubstance>(r, r, r);

    auto lower = std::make_unique<Element>();
    lower->kind = "sphere_lower";
    lower->substance = std::make_unique<EllipsoidSubstance>(r, r, r);
    lower->origin = {0.f, 0.f, 0.f};        // SAME origin as the parent
    lower->scale  = {1.f, 1.f, -1.f};       // flips the sheet downward
    lower->color = {0.25f, 0.55f, 0.95f};
    root->AddChild(std::move(lower));
    return root;
}

} // namespace

int main(int argc, char** argv) {
    std::string out = "assets/sphere.psf";
    int grid = 48;
    float radius = 0.5f;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* d) -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string(d);
        };
        if (a == "--out") out = next("assets/sphere.psf");
        else if (a == "--grid") grid = std::atoi(next("48").c_str());
        else if (a == "--radius") radius = float(std::atof(next("0.5").c_str()));
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }
    if (grid < 4) grid = 4;

    // --- bake --------------------------------------------------------------
    auto tree = BuildClosedSphere(radius);
    std::string why;
    auto body = ElementSystem::BakeBody(*tree, grid, 1.0f, why);
    if (!body) {
        std::fprintf(stderr, "bake failed: %s\n", why.c_str());
        return 1;
    }
    std::printf("baked: %zu anchors, %zu surfaces\n",
                body->AnchorBlob().size(), body->SurfaceBlob().size());

    // --- assemble the container --------------------------------------------
    PSFFile f;
    f.version = 3;
    f.anchors  = body->AnchorBlob();
    f.surfaces = body->SurfaceBlob();
    // The stitch section: 8 int8 per surface. The baked body carries the
    // seam coefficients inside the anchors' curve_params, and the surface
    // slots are zero for a freshly baked body -- which is what a fresh
    // body should carry.
    f.stitch.assign(body->SurfaceBlob().size() * 8, 0);

    // The bbox and s_scale MUST be the BAKER'S, not a recomputation.
    //
    // WHY: the anchor positions are QUANTISED against the box the baker
    // used. A different box makes every stored position dequantise to a
    // different world point. Measured: the baker's box is 1.1732 wide
    // (with its 5% margin); recomputing a tight box from the anchor
    // positions gave 1.0000, and every position then shrank by 14.7% --
    // the sphere's z-span went from +-0.4998 to +-0.4260 and it stopped
    // being a sphere.
    //
    // The two failure modes this avoids are both measured:
    //   - a ZERO bbox_span collapses the mesh to a point;
    //   - a WRONG bbox_span silently misplaces every anchor.
    f.bbox_min  = body->BBoxMin();
    f.bbox_span = body->BBoxSpan();
    // A full-scale int8 stitch moves an edge by 1.2% of the diagonal --
    // the canonical derivation (tools/psf_encode.py in the proven tree).
    const float diag = std::sqrt(f.bbox_span.x * f.bbox_span.x +
                                 f.bbox_span.y * f.bbox_span.y +
                                 f.bbox_span.z * f.bbox_span.z);
    f.s_scale = 0.012f * diag / 127.0f;
    std::printf("bbox min (%.4f %.4f %.4f) span (%.4f %.4f %.4f) s_scale %.8f\n",
                f.bbox_min.x, f.bbox_min.y, f.bbox_min.z,
                f.bbox_span.x, f.bbox_span.y, f.bbox_span.z, f.s_scale);

    if (!f.Save(out)) {
        std::fprintf(stderr, "save failed: %s\n", out.c_str());
        return 1;
    }
    std::printf("wrote %s\n", out.c_str());

    // --- read it back ------------------------------------------------------
    // A write that produced an unloadable file is not a success. The
    // gallery's scene N failed with a misleading "MISSING" message for
    // exactly this reason: the file existed, but it was not a format the
    // loader could read.
    PSFFile check;
    std::string err;
    if (!PSFFile::Load(out, check, &err)) {
        std::fprintf(stderr, "READ-BACK FAILED: %s\n", err.c_str());
        return 1;
    }
    std::printf("read back: %zu anchors, %zu surfaces, stitch %zu bytes\n",
                check.anchors.size(), check.surfaces.size(), check.stitch.size());
    if (check.anchors.size() != f.anchors.size() ||
        check.surfaces.size() != f.surfaces.size()) {
        std::fprintf(stderr, "READ-BACK MISMATCH\n");
        return 1;
    }
    // And it must load as a BODY.
    auto reloaded = PSFBody::FromFile(out.c_str());
    if (!reloaded) {
        std::fprintf(stderr, "the written file does not load as a body\n");
        return 1;
    }
    std::printf("loads as a body: %zu anchors, %zu surfaces\n",
                reloaded->AnchorBlob().size(), reloaded->SurfaceBlob().size());

    // --- it must be a CLOSED SPHERE, not a dome ---------------------------
    // "Closed" is checked on the RELOADED file, so it verifies the written
    // asset rather than the in-memory body.
    //
    // The two properties that distinguish a closed body from a dome:
    //   - it is SYMMETRIC about z = 0 (a dome's lower half is missing);
    //   - its z-extent matches its x-extent (a dome's z-span is ~half its
    //     x-span).
    //
    // WHY THE GEOMETRIC EXTENT, NOT BBoxSpan(): the bbox carries a 5%
    // MARGIN on every side (the baker adds it so the quantised positions
    // have room), so `BBoxSpan().x` is 1.1732 for a unit sphere. The
    // comparison must use the extent of the ANCHORS themselves.
    float zmin = 1e30f, zmax = -1e30f;
    float xmin = 1e30f, xmax = -1e30f;
    const auto& pool = reloaded->AnchorsConst();
    for (size_t i = 0; i < reloaded->AnchorBlob().size(); ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        if (p.z < zmin) zmin = p.z;
        if (p.z > zmax) zmax = p.z;
        if (p.x < xmin) xmin = p.x;
        if (p.x > xmax) xmax = p.x;
    }
    const float xspan = xmax - xmin;
    const float zspan = zmax - zmin;
    const bool symmetric = std::fabs(zmin + zmax) < 0.02f * xspan;
    const bool full = zspan > 0.9f * xspan;
    std::printf("z range [%+.4f, %+.4f]  span %.4f\n", zmin, zmax, zspan);
    std::printf("x range [%+.4f, %+.4f]  span %.4f   ratio z/x %.4f\n",
                xmin, xmax, xspan, zspan / xspan);
    std::printf("symmetric about z=0: %s   spans the full width: %s\n",
                symmetric ? "YES" : "NO", full ? "YES" : "NO");
    if (!symmetric || !full) {
        std::fprintf(stderr, "the written asset is NOT a closed sphere\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
