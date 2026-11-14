// Diagnostic: WHERE did the cube's side walls go?
//
// The model-generated cube bakes to 4608 anchors / 4418 surfaces and the
// anchor extent is a perfect cube (x/y/z span 1.0, symmetric). But the
// render shows TWO OVERLAPPING FLAT SHEETS with no side walls. This probe
// measures the anchors' and surfaces' actual 3D structure to name the
// stage that dropped the walls.
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "assets/cube.psf";
    auto body = PSFBody::FromFile(path);
    if (!body) { std::printf("cannot load %s\n", path); return 1; }

    const size_t n = body->AnchorBlob().size();
    const auto& pool = body->AnchorsConst();
    std::printf("anchors: %zu   surfaces: %zu\n", n, body->SurfaceBlob().size());

    // 1) The anchor extent (already verified -- re-measure for context).
    float mn[3] = {1e30f,1e30f,1e30f}, mx[3] = {-1e30f,-1e30f,-1e30f};
    for (size_t i = 0; i < n; ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        for (int a = 0; a < 3; ++a) {
            const float v = a == 0 ? p.x : (a == 1 ? p.y : p.z);
            if (v < mn[a]) mn[a] = v;
            if (v > mx[a]) mx[a] = v;
        }
    }
    std::printf("anchor extent: x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]\n",
                mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);

    // 2) THE KEY QUESTION: for anchors near the cube's TOP (z near +0.5),
    //    do they cover the full x/y footprint? And are there anchors on the
    //    SIDE faces (|z| < 0.4 but far from the z-axis)?
    //
    // A real cube's surface has THREE kinds of points:
    //   - top/bottom faces: |z| ~ 0.5, x/y anywhere in [-0.5,0.5]
    //   - side walls: |x| or |y| ~ 0.5 with |z| anywhere in [-0.5,0.5]
    // If the side walls are missing, the second set is EMPTY.
    int top_like = 0, bottom_like = 0, side_like = 0;
    int near_axis = 0;
    for (size_t i = 0; i < n; ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        if (p.z > 0.48f) ++top_like;
        else if (p.z < -0.48f) ++bottom_like;
        else if (std::fabs(p.x) > 0.48f || std::fabs(p.y) > 0.48f) ++side_like;
        else ++near_axis;
    }
    std::printf("distribution: top %d  bottom %d  SIDE-WALL %d  interior %d\n",
                top_like, bottom_like, side_like, near_axis);

    // 3) The wall-band question. The BoxSubstance comment says "the side
    //    walls hang DOWN from the footprint's edge". Measure where the
    //    substance's surface actually is at a footprint-edge point:
    //    for x near +0.5, what z values do anchors take?
    std::printf("\nanchors with |x| > 0.45 (the +x/-x walls):\n");
    int wall_x = 0;
    float wzmin = 1e30f, wzmax = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        const Vec3 p = pool.Position(uint16_t(i));
        if (std::fabs(p.x) > 0.45f) {
            ++wall_x;
            if (p.z < wzmin) wzmin = p.z;
            if (p.z > wzmax) wzmax = p.z;
        }
    }
    std::printf("  count %d  z range [%.4f, %.4f]\n", wall_x, wzmin, wzmax);

    // 4) Surface-level: how many patches have a normal pointing HORIZONTALLY
    //    (a wall)? A cube has 6 faces: 2 horizontal (top/bottom), 4 vertical
    //    walls. If the walls are missing, no patch normal is horizontal.
    int horiz = 0, vert = 0, other = 0;
    const auto& patches = body->Patches();
    for (const Patch& p : patches) {
        const Vec3 nrm = p.EvaluateNormal(0.5f, 0.5f);
        const float nz = std::fabs(nrm.z);
        if (nz > 0.8f) ++vert;      // normal mostly +z or -z: a top/bottom face
        else if (nz < 0.2f) ++horiz; // normal mostly horizontal: a wall
        else ++other;
    }
    std::printf("\npatch normals: wall-like %d  top/bottom-like %d  mixed %d\n",
                horiz, vert, other);
    std::printf("(a closed cube should show ~4/6 of patches wall-like, ~2/6 top/bottom)\n");
    return 0;
}
