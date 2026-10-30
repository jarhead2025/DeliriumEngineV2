// =============================================================================
// recon_diag - HEADLESS diagnostic for the black recon window.
//
// No SDL, no Engine loop. Builds the SAME four element-system bodies the
// recon window builds (orb / bump / wave / mound at the 2x2 grid) plus the
// torus as a CONTROL, then answers three concrete questions:
//
//   Q1  CAMERA: is each body in front of the camera and inside the frustum?
//       (projects the body's bbox center + 8 corners to screen pixels,
//        prints the dot(view_dir, to_body) and the march near/far range)
//   Q2  GEOMETRY: does a camera ray actually hit each body? (Raycast)
//   Q3  RENDER: does each render path (march / raymarch / splat) produce
//       ANY non-background pixel for the generated bodies -- and for the
//       torus control? (renders, saves PPM, counts lit pixels)
//
//   Build:  cmake --build build_delirium --target delirium_recon_diag
//   Run:    ./build_dir/delirium_recon_diag            (from project root)
//
// Reading the output:
//   * Q1 says "BEHIND" or corners off-screen  -> camera/placement bug
//   * Q1 OK, Q2 says "no hit"                 -> records/geometry bug
//   * Q1+Q2 OK, Q3 torus lit but bodies 0     -> generated-records bug
//   * Q3 torus ALSO 0                         -> renderer/env bug (device?)
// =============================================================================
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/Reconstructor.h"
#include "delirium/core/Types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

using namespace DeliRium;

static void setenv_(const char* k, const char* v) { ::setenv(k, v, 1); }
static void unsetenv_(const char* k) { ::unsetenv(k); }

// Same projection SplatRenderer::BuildSampleGrids uses.
static void Project(const float mvp[16], const Vec3& P, int W, int H,
                    float& x, float& y, float& w) {
    const float px = mvp[0]*P.x + mvp[1]*P.y + mvp[2]*P.z + mvp[3];
    const float py = mvp[4]*P.x + mvp[5]*P.y + mvp[6]*P.z + mvp[7];
    const float pw = mvp[12]*P.x + mvp[13]*P.y + mvp[14]*P.z + mvp[15];
    w = pw;
    const float ww = pw > 1e-6f ? pw : 1e-6f;
    x = (px/ww*0.5f + 0.5f) * (W - 1);
    y = (0.5f - py/ww*0.5f) * (H - 1);
}

// Exact CPU copies of kernels/psf_splat.cl deform_h / stitch_h.
static float deform_h(const float c[8], float u, float v) {
    const float bu = u*(1.f-u), bv = v*(1.f-v);
    const float au = 2.f*u-1.f, av = 2.f*v-1.f;
    const float w = 1.f-v, x = 1.f-u;
    return c[0]*bu*w + c[1]*bu*v + c[2]*bv*x + c[3]*bv*u
         + c[4]*bu*au*w + c[5]*bu*au*v + c[6]*bv*av*x + c[7]*bv*av*u;
}
static float stitch_h(const int8_t* e, float u, float v, float s_scale) {
    const float tu = u*(1.f-u), tv = v*(1.f-v);
    return (1.f-v)*tu*(e[0]*s_scale + e[1]*s_scale*u)
         +      v *tu*(e[2]*s_scale + e[3]*s_scale*u)
         + (1.f-u)*tv*(e[4]*s_scale + e[5]*s_scale*v)
         +      u *tv*(e[6]*s_scale + e[7]*s_scale*v);
}

// Dump the raw records of a body with the renderer's exact math, so the
// black-screen cause is visible in stdout: coefficient magnitude (surface
// displacement), deformed-center vs bbox, normal direction (back-face cull),
// and the decoded color at patch center (the 8x-bug check).
static void DumpRecords(const char* name, const PSFBody& b) {
    const auto& A = b.AnchorBlob();
    const auto& S = b.SurfaceBlob();
    const auto& St = b.StitchBlob();
    const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    std::printf("\n-- records: %s (anchors=%zu surfaces=%zu stitch_bytes=%zu) --\n",
                name, A.size(), S.size(), St.size());
    std::printf("   bbox min=(%.3f,%.3f,%.4f) span=(%.3f,%.3f,%.4f)\n",
                mn.x, mn.y, mn.z, sp.x, sp.y, sp.z);
    float cmax = 0.f; int cmax_patch = -1; float cmax_c = -1.f;
    for (size_t s = 0; s < S.size(); ++s)
        for (int k = 0; k < 8; ++k) {
            const float a = std::fabs(S[s].coef[k]);
            if (a > cmax) { cmax = a; cmax_patch = (int)s; cmax_c = S[s].coef[k]; }
        }
    std::printf("   max |coef| = %.5f (patch %d, diag=%.3f)  %s\n",
                cmax, cmax_patch, diag,
                cmax > 0.05f*diag ? "  <-- HUGE: surface displaced off the bbox!" : "ok");
    // deformed center + normal + color for the first 2 patches
    for (size_t s = 0; s < std::min<size_t>(2, S.size()); ++s) {
        const SurfaceRecord& r = S[s];
        const uint8_t* ab = (const uint8_t*)A.data();
        float P[3] = {0,0,0}, N[3] = {0,0,0};
        for (int k = 0; k < 4; ++k) {
            const uint8_t* a = ab + (size_t)r.anchor_idx[k]*64;
            uint16_t px, py, pz;
            std::memcpy(&px, a + 32, 2); std::memcpy(&py, a + 34, 2); std::memcpy(&pz, a + 36, 2);
            const float wgt = 0.25f;   // 4-corner average
            P[0] += (mn.x + px/65535.f*sp.x) * wgt;
            P[1] += (mn.y + py/65535.f*sp.y) * wgt;
            P[2] += (mn.z + pz/65535.f*sp.z) * wgt;
            N[0] += (a[44]*2.f/255.f - 1.f) * wgt;   // kernel: n*(2/255)-1
            N[1] += (a[45]*2.f/255.f - 1.f) * wgt;
            N[2] += (a[46]*2.f/255.f - 1.f) * wgt;
        }
        const float nl = std::sqrt(N[0]*N[0]+N[1]*N[1]+N[2]*N[2]);
        if (nl > 1e-9f) { N[0]/=nl; N[1]/=nl; N[2]/=nl; }
        const float u = 0.5f, v = 0.5f;
        const float h = deform_h(r.coef, u, v)
                      + stitch_h(St.empty() ? nullptr : St.data() + s*8, u, v, b.StitchScale());
        const float Pq[3] = {P[0]+h*N[0], P[1]+h*N[1], P[2]+h*N[2]};
        // color at center, EXACT kernel math: sum_k K[k]/63 * pattern_eval(idx_k, u, v)
        float col[3];
        for (int ch = 0; ch < 3; ++ch) {
            const uint8_t* cp = r.color_pattern + ch*6;
            const int idx[4] = { (int)(cp[0] >> 4), (int)(cp[0] & 0xF),
                                 (int)(cp[1] >> 4), (int)(cp[1] & 0xF) };
            const int8_t* K = (const int8_t*)(cp + 2);
            float val = 0.f;
            for (int k = 0; k < 4; ++k) {
                float pe = 1.f;
                switch (idx[k] & 15) {
                    case 1: pe = u; break;  case 2: pe = v; break;
                    case 3: pe = u*v; break; case 4: pe = u*u; break;
                    case 5: pe = v*v; break; case 6: pe = u*u*v; break;
                    case 7: pe = u*v*v; break;
                    case 8: pe = std::cos(3.14159265f*u); break;
                    case 9: pe = std::cos(3.14159265f*v); break;
                    default: pe = 1.f; break;   // idx 0
                }
                val += (float)K[k] / 63.f * pe;
            }
            col[ch] = std::max(0.f, std::min(1.f, val));
        }
        std::printf("   patch %zu: anchors=%u,%u,%u,%u  h(0.5,0.5)=%.5f  N=(%.2f,%.2f,%.2f)\n",
                    s, r.anchor_idx[0], r.anchor_idx[1], r.anchor_idx[2], r.anchor_idx[3], h, N[0], N[1], N[2]);
        std::printf("            center=(%.3f,%.3f,%.4f)  deformed=(%.3f,%.3f,%.4f)  color=(%.3f,%.3f,%.3f)\n",
                    P[0], P[1], P[2], Pq[0], Pq[1], Pq[2], col[0], col[1], col[2]);
        if (Pq[0] < mn.x - 0.1f*diag || Pq[0] > mn.x + sp.x + 0.1f*diag ||
            Pq[1] < mn.y - 0.1f*diag || Pq[1] > mn.y + sp.y + 0.1f*diag ||
            Pq[2] < mn.z - 0.1f*diag || Pq[2] > mn.z + sp.z + 0.1f*diag)
            std::printf("            <-- DEFORMED CENTER OUTSIDE BBOX (rays may miss)\n");
        if (N[2] < -0.5f)
            std::printf("            <-- NORMAL POINTS DOWN (-z): top view sees back faces (cull!)\n");
    }
}

// Count pixels in a P6 PPM that differ from the renderer background (18,18,24).
// Returns lit count; total = W*H. -1 on read failure.
static long CountLit(const std::string& path, long& total) {
    std::ifstream f(path);   // text==binary on Linux; no <ios> needed
    if (!f) return -1;
    std::string magic; f >> magic;
    if (magic != "P6") { std::fprintf(stderr, "  (not a P6 PPM: %s)\n", path.c_str()); return -1; }
    auto read_int = [&]() {
        int v = 0; char c;
        f.get(c);
        while (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '#') f.get(c);
        while (c >= '0' && c <= '9') { v = v*10 + (c - '0'); f.get(c); }
        f.unget();
        return v;
    };
    const int W = read_int(), H = read_int(), mv = read_int();
    (void)mv;
    f.get(); // single whitespace after maxval
    std::vector<unsigned char> px((size_t)W * H * 3);
    f.read((char*)px.data(), (std::streamsize)px.size());
    total = (long)W * H;
    long lit = 0;
    for (size_t i = 0; i < px.size(); i += 3) {
        const int dr = (int)px[i]     - 18;
        const int dg = (int)px[i + 1] - 18;
        const int db = (int)px[i + 2] - 24;
        if (std::abs(dr) + std::abs(dg) + std::abs(db) > 24) ++lit;
    }
    return lit;
}

struct Spec { const char* name; const char* semantic; Vec3 pos; uint32_t patches; };

int main(int argc, char** argv) {
    int W = 1024, H = 768;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--size" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%dx%d", &W, &H) != 2) {
                std::fprintf(stderr, "bad --size\n"); return 2;
            }
        }
    }

    SplatRenderer renderer;
    if (!renderer.Initialize(W, H, nullptr, "kernels")) {
        std::fprintf(stderr, "SplatRenderer::Initialize failed (OpenCL / kernels?)\n");
        return 1;
    }
    std::printf("== recon_diag (headless) ==\n");
    std::printf("device: %s   size: %dx%d\n", renderer.DeviceName().c_str(), W, H);

    // ---- build the same four bodies as the recon window -------------------
    const Spec specs[] = {
        {"orb",   "orb",   {-0.7f, -0.7f, 0.f}, 900},
        {"bump",  "bump",   { 0.7f, -0.7f, 0.f}, 900},
        {"wave",  "wave",   {-0.7f,  0.7f, 0.f}, 900},
        {"mound", "mound",  { 0.7f,  0.7f, 0.f}, 900},
    };
    NeuralReconstructor maker;
    std::vector<std::unique_ptr<PSFBody>> bodies;
    for (const Spec& sp : specs) {
        ReconstructionRequest req;
        req.semantic = sp.semantic;
        req.max_patches = sp.patches;
        req.tolerance_pct = 2.0f;
        req.world_offset = sp.pos;
        auto b = maker.Reconstruct(req);
        if (!b) { std::fprintf(stderr, "reconstruction of '%s' FAILED\n", sp.semantic); return 1; }
        bodies.push_back(std::move(b));
    }
    // ---- record dump: exact kernel math on the raw generated records ------
    for (size_t i = 0; i < bodies.size(); ++i)
        DumpRecords(specs[i].name, *bodies[i]);

    // control: the torus (known-good records, known-good render)
    auto torus = PSFBody::FromFile("assets/torus.psf");
    if (!torus) std::fprintf(stderr, "(torus control unavailable: assets/torus.psf not found)\n");

    // ---- the EXACT recon-window camera ------------------------------------
    Camera cam;
    cam.position = {0.f, 3.2f, 4.6f};
    cam.target   = {0.f, 0.f, 0.f};
    cam.up       = {0.f, 1.f, 0.f};
    cam.fov_deg = 45.f;
    float mvp[16];
    cam.BuildMVP(mvp, float(W) / float(H));

    Vec3 f{cam.target.x-cam.position.x, cam.target.y-cam.position.y, cam.target.z-cam.position.z};
    const float fl = std::sqrt(f.x*f.x+f.y*f.y+f.z*f.z);
    f.x/=fl; f.y/=fl; f.z/=fl;
    std::printf("\n-- Q1: camera (window camera: pos(%.1f,%.1f,%.1f) -> target origin) --\n",
                cam.position.x, cam.position.y, cam.position.z);
    std::printf("   view dir = (%.3f, %.3f, %.3f)\n", f.x, f.y, f.z);

    // ---- Q1: projection + Q2: raycast per body ----------------------------
    // Q2 casts a ray STRAIGHT AT the body's bbox center (not the window
    // center pixel -- the bodies project off-center, so a center-pixel ray
    // would miss them and give a false "no hit").
    for (size_t i = 0; i < bodies.size(); ++i) {
        const PSFBody& b = *bodies[i];
        const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
        const Vec3 c{mn.x + sp.x*0.5f, mn.y + sp.y*0.5f, mn.z + sp.z*0.5f};
        const Vec3 to{c.x-cam.position.x, c.y-cam.position.y, c.z-cam.position.z};
        const float dot = to.x*f.x + to.y*f.y + to.z*f.z;
        const float dist = std::sqrt(to.x*to.x + to.y*to.y + to.z*to.z);
        Vec3 ro = cam.position;
        Vec3 rd{to.x, to.y, to.z};
        const float rdl = std::sqrt(rd.x*rd.x + rd.y*rd.y + rd.z*rd.z);
        if (rdl > 1e-9f) { rd.x/=rdl; rd.y/=rdl; rd.z/=rdl; }
        float cx, cy, cw;
        Project(mvp, c, W, H, cx, cy, cw);
        int onscreen = 0;
        for (int ox = 0; ox < 2; ++ox)
            for (int oy = 0; oy < 2; ++oy)
                for (int oz = 0; oz < 2; ++oz) {
                    Vec3 p{mn.x + ox*sp.x, mn.y + oy*sp.y, mn.z + oz*sp.z};
                    float x, y, w;
                    Project(mvp, p, W, H, x, y, w);
                    if (w > 0.f && x >= 0 && x < W && y >= 0 && y < H) ++onscreen;
                }
        SurfaceHit hit;
        const bool hitok = b.Raycast(ro, rd, hit);
        std::printf("   [%s] bbox=(%.2f,%.2f,%.3f)..(%.2f,%.2f,%.3f)  center=(%.2f,%.2f,%.3f)\n",
                    specs[i].name, mn.x, mn.y, mn.z, mn.x+sp.x, mn.y+sp.y, mn.z+sp.z, c.x, c.y, c.z);
        std::printf("      in_front=%s (dot=%.3f, dist=%.2f)  center_screen=(%6.1f,%6.1f) w=%.4f  corners_onscreen=%d/8  raycast_center=%s%s\n",
                    dot > 0.f ? "YES" : "NO/BEHIND", dot, dist, cx, cy, cw, onscreen,
                    hitok ? "HIT" : "no hit", hitok ? "" : "  <-- Q2 FAIL");
    }

    // ---- Q3: render each path, count non-background pixels ----------------
    auto env_march    = [](){ setenv_("DELIRIUM_MARCH", "1"); unsetenv_("DELIRIUM_RAYMARCH"); };
    auto env_raymarch = [](){ setenv_("DELIRIUM_RAYMARCH", "1"); setenv_("DELIRIUM_MARCH", "0"); };
    auto env_splat    = [](){ setenv_("DELIRIUM_MARCH", "0"); unsetenv_("DELIRIUM_RAYMARCH"); };
    struct Path { const char* name; void (*env)(void); std::string file; };
    const Path paths[] = {
        {"march",    env_march,    "/tmp/recon_diag_march.ppm"},
        {"raymarch", env_raymarch, "/tmp/recon_diag_raymarch.ppm"},
        {"splat",    env_splat,    "/tmp/recon_diag_splat.ppm"},
    };

    std::printf("\n-- Q3: render paths (window camera, all 4 bodies composited) --\n");
    for (const Path& p : paths) {
        p.env();
        bool ok = true;
        for (size_t i = 0; i < bodies.size(); ++i)
            if (!renderer.RenderBody(*bodies[i], cam, /*clear=*/(i == 0))) ok = false;
        renderer.Present();
        renderer.SaveFramePPM(p.file.c_str());
        long total = 0, lit = CountLit(p.file, total);
        std::printf("   %-9s render_ok=%s  lit_pixels=%6ld / %ld  (%.3f%%)  -> %s\n",
                    p.name, ok ? "yes" : "NO", lit, total,
                    total ? 100.0 * lit / (double)total : 0.0, p.file.c_str());
    }

    // ---- per-body framing camera: is it the camera or the records? -------
    // Each body rendered ALONE from a camera that looks straight at it
    // (from above, so the flat cards are not viewed edge-on). If a body
    // renders lit here but not in Q3 -> camera/placement. If black here too
    // -> the generated records themselves are the problem.
    auto frame_cam = [](const PSFBody& b, Camera& c) {
        const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
        const Vec3 tc{mn.x + sp.x*0.5f, mn.y + sp.y*0.5f, mn.z + sp.z*0.5f};
        const float td = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
        c.position = {tc.x + td*1.0f, tc.y + td*0.7f, tc.z + std::max(td*1.9f, 1.5f)};
        c.target = tc;
        c.up = {0.f, 1.f, 0.f};
        c.fov_deg = 45.f;
    };
    std::printf("\n-- per-body framing camera (body alone, viewed from above) --\n");
    for (size_t i = 0; i < bodies.size(); ++i) {
        Camera bcam;
        frame_cam(*bodies[i], bcam);
        for (const Path& p : paths) {
            p.env();
            const bool ok = renderer.RenderBody(*bodies[i], bcam, /*clear=*/true);
            renderer.Present();
            const std::string f3 = p.file + std::string(".") + specs[i].name;
            renderer.SaveFramePPM(f3.c_str());
            long total = 0, lit = CountLit(f3, total);
            std::printf("   %-5s %-9s render_ok=%s  lit_pixels=%6ld / %ld  (%.3f%%)\n",
                        specs[i].name, p.name, ok ? "yes" : "NO", lit, total,
                        total ? 100.0 * lit / (double)total : 0.0);
        }
    }

    // ---- control: torus with its own framing camera, each path ------------
    if (torus) {
        const Vec3 mn = torus->BBoxMin(), sp = torus->BBoxSpan();
        const Vec3 tc{mn.x + sp.x*0.5f, mn.y + sp.y*0.5f, mn.z + sp.z*0.5f};
        const float tdiag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
        Camera tcam;
        tcam.position = {tc.x + tdiag*1.6f, tc.y + tdiag*1.1f, tc.z + tdiag*1.9f};
        tcam.target = tc;
        tcam.up = {0.f, 1.f, 0.f};
        tcam.fov_deg = 45.f;
        std::printf("\n-- control: torus (known-good records), framed camera --\n");
        for (const Path& p : paths) {
            p.env();
            const bool ok = renderer.RenderBody(*torus, tcam, /*clear=*/true);
            renderer.Present();
            const std::string f2 = p.file + ".torus";
            renderer.SaveFramePPM(f2.c_str());
            long total = 0, lit = CountLit(f2, total);
            std::printf("   %-9s render_ok=%s  lit_pixels=%6ld / %ld  (%.3f%%)  -> %s\n",
                        p.name, ok ? "yes" : "NO", lit, total,
                        total ? 100.0 * lit / (double)total : 0.0, f2.c_str());
        }
    }

    std::printf("\nDone. PPMs in /tmp -- open them to see what each path actually drew.\n");
    renderer.Shutdown();
    return 0;
}