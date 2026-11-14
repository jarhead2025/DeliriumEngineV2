// Bake the MODEL-GENERATED cube and write it as a valid binary .psf asset.
//
// WHY THIS TOOL EXISTS: the user asked to "test a cube render with the
// local model" to shed light on the render-pipeline glitches the torus
// shows. A cube is the sharpest lens: flat faces with CONSTANT normals
// turn any normal/shading artifact into crisp edges or missing faces.
//
// The tree it bakes is the one the model actually produced for the cube
// document (/tmp/doc_cube.txt, logged in /tmp/cube_run2.txt): a grouping
// body with TWO "box" sheets -- an upper and a mirrored lower with
// scale [1,1,-1] -- i.e. the model applied the closed-body rule BY ITSELF.
//
// The tool writes the file and then READS IT BACK through the normal
// loader, so a success means the asset is loadable -- not merely written.
#include "delirium/ai/ElementSystem.h"
#include "delirium/io/PSFFile.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace DeliRium;

namespace {

// The model's own tree, materialised. A "box" substance models the upper
// half of the solid (its z >= 0 part), so the closed body is the upper
// sheet plus a mirrored lower one at the SAME origin -- the same recipe
// the model emitted and the same one the sphere tool documents.
std::unique_ptr<Element> BuildModelCube(float sx, float sy, float sz) {
    auto root = std::make_unique<Element>();
    root->kind = "body";
    root->color = {0.549f, 0.549f, 0.549f};   // the model's steel grey
    root->substance = std::make_unique<BoxSubstance>(sx, sy, sz);

    auto lower = std::make_unique<Element>();
    lower->kind = "box_lower";
    lower->substance = std::make_unique<BoxSubstance>(sx, sy, sz);
    lower->origin = {0.f, 0.f, 0.f};          // SAME origin as the parent
    lower->scale  = {1.f, 1.f, -1.f};         // flips the sheet downward
    lower->color = {0.549f, 0.549f, 0.549f};
    root->AddChild(std::move(lower));
    return root;
}

} // namespace

int main(int argc, char** argv) {
    std::string out = "assets/cube.psf";
    int grid = 48;
    float sx = 0.5f, sy = 0.5f, sz = 0.5f;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* d) -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string(d);
        };
        if (a == "--out") out = next("assets/cube.psf");
        else if (a == "--grid") grid = std::atoi(next("48").c_str());
        else if (a == "--sx") sx = float(std::atof(next("0.5").c_str()));
        else if (a == "--sy") sy = float(std::atof(next("0.5").c_str()));
        else if (a == "--sz") sz = float(std::atof(next("0.5").c_str()));
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }
    if (grid < 4) grid = 4;

    // --- bake --------------------------------------------------------------
    auto tree = BuildModelCube(sx, sy, sz);
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
    f.stitch.assign(body->SurfaceBlob().size() * 8, 0);

    // The bbox and s_scale MUST be the BAKER'S, not a recomputation --
    // the anchor positions are QUANTISED against the baker's box, and a
    // different box silently misplaces every anchor (the G18 lesson,
    // measured at a 14.7% shrink).
    f.bbox_min  = body->BBoxMin();
    f.bbox_span = body->BBoxSpan();
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
    PSFFile check;
    std::string err;
    if (!PSFFile::Load(out, check, &err)) {
        std::fprintf(stderr, "READ-BACK FAILED: %s\n", err.c_str());
        return 1;
    }
    std::printf("read back: %zu anchors, %zu surfaces\n",
                check.anchors.size(), check.surfaces.size());

    auto reloaded = PSFBody::FromFile(out.c_str());
    if (!reloaded) {
        std::fprintf(stderr, "the written file does not load as a body\n");
        return 1;
    }

    // --- it must be a CLOSED cube, not a single sheet ----------------------
    // Symmetric about z = 0, and the z-extent matches the x-extent.
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
    std::printf("z range [%+.4f, %+.4f] span %.4f\n", zmin, zmax, zspan);
    std::printf("x range [%+.4f, %+.4f] span %.4f   ratio z/x %.4f\n",
                xmin, xmax, xspan, zspan / xspan);
    std::printf("symmetric about z=0: %s   spans the full width: %s\n",
                symmetric ? "YES" : "NO", full ? "YES" : "NO");
    if (!symmetric || !full) {
        std::fprintf(stderr, "the written asset is NOT a closed cube\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
