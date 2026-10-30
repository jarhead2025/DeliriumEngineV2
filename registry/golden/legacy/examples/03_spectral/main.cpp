// =============================================================================
// example_spectral/main.cpp - render a spectral primitive (.spf) on the GPU
// via SpectralRenderer and write a PPM. This is the integrated path (engine
// classes), as opposed to the standalone tools/spf_gpu_smoketest.c.
//
//   ./example_spectral assets/roughball.spf [out.ppm]
//
// Prints the GPU name, coverage, and writes the image. If you see coverage
// well above zero and a recognizable shape in the PPM, the spectral renderer
// works end to end on your hardware.
// =============================================================================
#include "delirium/psf/SpectralPrimitive.h"
#include "delirium/render/SpectralRenderer.h"
#include "delirium/render/Camera.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* spf = argc > 1 ? argv[1] : "assets/roughball.spf";
    const char* out = argc > 2 ? argv[2] : "spectral_out.ppm";
    const int W = 512, H = 512;

    SpectralPrimitive prim;
    if (!prim.Load(spf)) { std::fprintf(stderr, "load failed: %s\n", spf); return 1; }
    std::printf("loaded %s: geom L=%u, color L=%u, %zu bytes\n",
                spf, prim.GeomOrder(), prim.ColorOrder(), prim.ByteSize());

    SpectralRenderer rnd;
    if (!rnd.Initialize(W, H, "kernels")) {
        std::fprintf(stderr, "renderer init failed (no OpenCL device?)\n");
        return 1;
    }
    std::printf("GPU: %s  (64-bit atomics: %s)\n",
                rnd.DeviceName().c_str(), rnd.HasLongAtomics() ? "yes":"no");

    // camera looking at the primitive
    Camera cam;
    Vec3 c = prim.Center();
    cam.position = { c.x + 2.6f, c.y - 2.2f, c.z + 1.6f };
    cam.target   = c;
    cam.up       = { 0, 0, 1 };

    float cw[3] = { c.x, c.y, c.z };
    rnd.BeginFrame();
    rnd.DrawPrimitive(prim, cw, 1.0f, 0 /*full geom*/, 0 /*full color*/, cam);
    rnd.EndFrame();

    const std::vector<uint8_t>& img = rnd.Image();
    unsigned long covered = 0;
    for (int i = 0; i < W*H; ++i)
        if (!(img[i*4]==18 && img[i*4+1]==18 && img[i*4+2]==24)) covered++;
    std::printf("coverage: %lu / %d pixels (%.1f%%)\n",
                covered, W*H, 100.0*covered/(W*H));

    FILE* f = fopen(out, "wb");
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W*H; ++i) fwrite(&img[i*4], 1, 3, f);
    fclose(f);
    std::printf("wrote %s\n", out);

    rnd.Shutdown();
    std::printf(covered > 1000 ? "\nSPECTRAL RENDER OK\n"
                               : "\nWARNING: near-empty image\n");
    return covered > 1000 ? 0 : 2;
}
