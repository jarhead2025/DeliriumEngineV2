// =============================================================================
// realtime_deform - HEADLESS real-time rendering benchmark WHILE deforming.
//
// Renders the torus continuously at a fixed timestep and drives a sustained
// deformation workload every frame, so we measure the TRUE per-frame cost of
// "deform + GPU flush + render" (not a static scene). Headless: no window,
// no display, no SDL. Times the CPU deform phase and the GPU render phase
// (which includes the FlushToGPU upload) separately, and reports
// mean / p50 / p95 / max + FPS per phase.
//
//   Build:  cmake --build build_delirium --target delirium_realtime_deform
//   Run:    ./build_delirium/delirium_realtime_deform            (project root)
//
// Options:
//   --workload modal|amr|dent|mixed   deformation to drive (default: modal)
//       modal  : modal impulse + Step + WriteBack every frame (topology fixed)
//       amr    : RefineWhere + CoarsenWhere at a rotating point (topology
//                changes; NO modal -- the baked ROM is invalid after splits)
//       dent   : ApplyRadialDent + RefitLocal at a rotating point (geometry)
//       mixed  : modal + ApplyRadialDent + periodic HarmonicWeld (topology
//                fixed -- RefitLocal is excluded because it changes the
//                patch/anchor count and would disable the modal WriteBack)
//   --path march|raymarch|splat       render path (default: march)
//   --frames N                        total frames        (default: 300)
//   --warmup N                        un-timed frames     (default: 30)
//   --size WxH                        frame size         (default: 1024x768)
//   --dt T                            fixed timestep     (default: 1/60)
//   --shot FRAME FILE                 save a PPM at FRAME (see the deform)
//   --orbit                           slow camera orbit (default: on)
//   --no-orbit                        fixed camera
//
// NOTE on modal vs AMR: ModalDynamics::WriteBack is fitted to the REST
// topology and silently no-ops once the patch/anchor count changes (the
// n_surf/n_anchor guard). So modal and amr are separate workloads here --
// that is the engine's real capability, not a benchmark limitation.
// =============================================================================
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/physics/ModalDynamics.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace DeliRium;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// POSIX setenv/unsetenv are in the global namespace on libstdc++.
static void setenv_(const char* k, const char* v) { ::setenv(k, v, 1); }
static void unsetenv_(const char* k) { ::unsetenv(k); }

// Frame a body from a spherical camera (yaw/pitch degrees, dist_factor*diag).
static Camera FrameCamera(const PSFBody& b, float yaw_deg, float pitch_deg,
                          float dist_factor) {
    const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
    Vec3 c{ mn.x + sp.x * 0.5f, mn.y + sp.y * 0.5f, mn.z + sp.z * 0.5f };
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    const float dist = (diag > 1e-6f ? diag : 1.f) * dist_factor;
    const float yaw = yaw_deg * 3.14159265f / 180.f;
    const float pit = pitch_deg * 3.14159265f / 180.f;
    Camera cam;
    cam.position = { c.x + dist * std::cos(pit) * std::sin(yaw),
                     c.y + dist * std::sin(pit),
                     c.z + dist * std::cos(pit) * std::cos(yaw) };
    cam.target = c;
    cam.up = {0.f, 1.f, 0.f};
    cam.fov_deg = 45.f;
    cam.near_p = 0.05f;
    cam.far_p = 100.f;
    return cam;
}

// A point on the body surface at spherical angles (degrees), used to rotate
// the deformation target around the torus over time.
static Vec3 SurfacePoint(const PSFBody& b, float yaw_deg, float pitch_deg) {
    const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
    Vec3 c{ mn.x + sp.x * 0.5f, mn.y + sp.y * 0.5f, mn.z + sp.z * 0.5f };
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    const float r = diag * 0.5f;
    const float yaw = yaw_deg * 3.14159265f / 180.f;
    const float pit = pitch_deg * 3.14159265f / 180.f;
    return { c.x + r * std::cos(pit) * std::sin(yaw),
             c.y + r * std::sin(pit),
             c.z + r * std::cos(pit) * std::cos(yaw) };
}

// Find the patch nearest a world point (for ApplyImpulseAt's surface index).
static uint32_t NearestPatch(const PSFBody& b, const Vec3& p) {
    auto& pool = b.AnchorsConst();
    uint32_t best = 0; float best_d = 1e30f;
    for (uint32_t pid = 0; pid < b.SurfaceBlob().size(); ++pid) {
        const auto& sr = b.SurfaceBlob()[pid];
        Vec3 ctr{};
        for (int k = 0; k < 4; ++k) {
            Vec3 a = pool.Position(sr.anchor_idx[k]);
            ctr.x += a.x*0.25f; ctr.y += a.y*0.25f; ctr.z += a.z*0.25f;
        }
        const float dx = ctr.x-p.x, dy = ctr.y-p.y, dz = ctr.z-p.z;
        const float d = dx*dx + dy*dy + dz*dz;
        if (d < best_d) { best_d = d; best = pid; }
    }
    return best;
}

struct Stats {
    std::vector<double> deform;   // ms per frame
    std::vector<double> render;   // ms per frame (includes GPU flush)
    std::vector<double> total;    // ms per frame
    void report(const char* phase) {
        if (total.empty()) return;
        auto pct = [](std::vector<double> v, double p) {
            std::sort(v.begin(), v.end());
            if (v.empty()) return 0.0;
            size_t i = (size_t)(p / 100.0 * (v.size() - 1));
            return v[i];
        };
        double mean = 0; for (double x : total) mean += x; mean /= total.size();
        double dmean = 0; for (double x : deform) dmean += x; dmean /= deform.size();
        double rmean = 0; for (double x : render) rmean += x; rmean /= render.size();
        double fps = 1000.0 / (mean > 1e-9 ? mean : 1e-9);
        std::printf("\n== %s : %zu timed frames ==\n", phase, total.size());
        std::printf("  deform (CPU)  mean %8.3f ms | p50 %7.3f | p95 %7.3f | max %8.3f\n",
                    dmean, pct(deform, 50), pct(deform, 95), *std::max_element(deform.begin(), deform.end()));
        std::printf("  render (GPU)  mean %8.3f ms | p50 %7.3f | p95 %7.3f | max %8.3f\n",
                    rmean, pct(render, 50), pct(render, 95), *std::max_element(render.begin(), render.end()));
        std::printf("  total/frame   mean %8.3f ms | p50 %7.3f | p95 %7.3f | max %8.3f  ->  %6.1f FPS\n",
                    mean, pct(total, 50), pct(total, 95), *std::max_element(total.begin(), total.end()), fps);
        std::printf("  budget 16.67 ms/frame (60 FPS): %s\n",
                    mean <= 16.67 ? "MEETS 60 FPS" : "BELOW 60 FPS");
    }
};

struct Cfg {
    std::string workload = "modal";
    std::string path = "march";
    int frames = 300, warmup = 30;
    int w = 1024, h = 768;
    double dt = 1.0 / 60.0;
    int shot_frame = -1; std::string shot_file;
    bool orbit = true;
};

} // namespace

int main(int argc, char** argv) {
    Cfg cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--workload" && i + 1 < argc) cfg.workload = argv[++i];
        else if (a == "--path" && i + 1 < argc) cfg.path = argv[++i];
        else if (a == "--frames" && i + 1 < argc) cfg.frames = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) cfg.warmup = std::atoi(argv[++i]);
        else if (a == "--size" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%dx%d", &cfg.w, &cfg.h) != 2)
                { std::fprintf(stderr, "bad --size\n"); return 2; }
        }
        else if (a == "--dt" && i + 1 < argc) cfg.dt = std::atof(argv[++i]);
        else if (a == "--shot" && i + 2 < argc) {
            cfg.shot_frame = std::atoi(argv[++i]); cfg.shot_file = argv[++i];
        }
        else if (a == "--no-orbit") cfg.orbit = false;
        else if (a == "--orbit") cfg.orbit = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    // render path -> env (RenderBody reads these every frame)
    if (cfg.path == "march") {
        setenv_("DELIRIUM_MARCH", "1"); unsetenv_("DELIRIUM_RAYMARCH");
    } else if (cfg.path == "raymarch") {
        setenv_("DELIRIUM_RAYMARCH", "1"); setenv_("DELIRIUM_MARCH", "0");
        setenv_("DELIRIUM_GLOW", "0.45");
    } else if (cfg.path == "splat") {
        setenv_("DELIRIUM_MARCH", "0"); setenv_("DELIRIUM_RAYMARCH", "0");
    } else {
        std::fprintf(stderr, "bad --path (march|raymarch|splat)\n"); return 2;
    }

    SplatRenderer renderer;
    if (!renderer.Initialize(cfg.w, cfg.h, nullptr, "kernels")) {
        std::fprintf(stderr, "SplatRenderer::Initialize failed (OpenCL / kernels?)\n");
        return 1;
    }

    auto body = PSFBody::FromFile("assets/torus.psf");
    if (!body) {
        std::fprintf(stderr, "could not load assets/torus.psf (run from project root)\n");
        return 1;
    }
    ModalDynamics dyn;
    const bool have_modes = dyn.LoadModes("assets/torus.modes", *body);
    const bool use_modal = (cfg.workload == "modal" || cfg.workload == "mixed") && have_modes;
    if ((cfg.workload == "modal" || cfg.workload == "mixed") && !have_modes)
        std::printf("(no modes file: modal workload degrades to rigid)\n");

    const Vec3 mn = body->BBoxMin(), sp = body->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    Camera cam = FrameCamera(*body, 30.f, 22.f, 2.6f);

    std::printf("== DeliRium realtime deform benchmark ==\n");
    std::printf("workload=%s  path=%s  frames=%d (warmup %d)  size=%dx%d  dt=%.4f  orbit=%s\n",
                cfg.workload.c_str(), cfg.path.c_str(), cfg.frames, cfg.warmup,
                cfg.w, cfg.h, cfg.dt, cfg.orbit ? "on" : "off");
    std::printf("body: %zu patches, %zu anchors, modes=%s\n",
                body->Patches().size(), body->AnchorBlob().size(),
                have_modes ? "yes" : "no");

    Stats st;
    const int n = cfg.frames;
    for (int f = 0; f < n; ++f) {
        // --- advance the deformation target around the torus ---
        const float ang = 360.0f * (f / (float)n);

        // --- DEFORM PHASE (CPU) ---
        Clock::time_point t0 = Clock::now();
        if (cfg.workload == "modal" || cfg.workload == "mixed") {
            if (use_modal) {
                // re-seed a small impulse every 10 frames to keep the wave alive
                if (f % 10 == 0) {
                    Vec3 p = SurfacePoint(*body, ang, 20.f);
                    uint32_t s = NearestPatch(*body, p);
                    dyn.ApplyImpulseAt(s, 0.5f, 0.5f, 0.35f);
                }
                dyn.Step((float)cfg.dt);
                dyn.WriteBack(*body);
            }
        }
        if (cfg.workload == "amr") {
            // refine a rotating region, coarsen the opposite side periodically
            Vec3 p = SurfacePoint(*body, ang, 15.f);
            if (f % 15 == 0)
                body->RefineWhere(p, 0.35f*diag, 0.10f*diag, 32);
            if (f % 60 == 30) {
                Vec3 q = SurfacePoint(*body, ang + 180.f, -15.f);
                body->CoarsenWhere(q, 0.40f*diag, 0.004f, 32);
            }
        }
        if (cfg.workload == "dent" || cfg.workload == "mixed") {
            if (f % 20 == 0) {
                Vec3 p = SurfacePoint(*body, ang, 25.f);
                body->ApplyRadialDent(p, 0.30f*diag, 0.08f*diag);
                // RefitLocal only in the pure 'dent' workload: it changes the
                // patch/anchor count, which would disable the modal WriteBack
                // in 'mixed' (the n_surf/n_anchor guard).
                if (cfg.workload == "dent")
                    body->RefitLocal(p, 0.45f*diag);
            }
        }
        if (cfg.workload == "mixed") {
            if (f % 45 == 0) {
                body->Seams().Build(*body);
                body->Seams().HarmonicWeld(*body, 16, 0.2f);
            }
        }
        const double t_deform = ms_since(t0);

        // --- RENDER PHASE (GPU; includes FlushToGPU upload) ---
        if (cfg.orbit) {
            // slow orbit: rotate the camera yaw a little each frame
            const float d = 0.5f * 3.14159265f / 180.f;
            const Vec3 c = cam.target;
            const float rx = cam.position.x - c.x, rz = cam.position.z - c.z;
            const float s = std::sin(d), co = std::cos(d);
            cam.position.x = c.x + (co*rx + s*rz);
            cam.position.z = c.z + (-s*rx + co*rz);
        }
        Clock::time_point t1 = Clock::now();
        if (!renderer.RenderBody(*body, cam, /*clear=*/true)) {
            std::fprintf(stderr, "RenderBody failed at frame %d\n", f);
            return 1;
        }
        const double t_render = ms_since(t1);

        // optional screenshot to SEE the deforming state
        if (cfg.shot_frame == f && !cfg.shot_file.empty()) {
            if (renderer.SaveFramePPM(cfg.shot_file.c_str()))
                std::printf("  [shot] frame %d -> %s\n", f, cfg.shot_file.c_str());
        }

        if (f >= cfg.warmup) {
            st.deform.push_back(t_deform);
            st.render.push_back(t_render);
            st.total.push_back(t_deform + t_render);
        }

        // rolling status every 30 frames
        if (f % 30 == 0) {
            body->Seams().Build(*body);
            const float gap = body->Seams().MaxGap(*body);
            std::printf("  f%4d  patches=%5zu anchors=%5zu seam=%6.3f%% diag  E=%8.4f  "
                        "deform=%6.2fms render=%6.2fms\n",
                        f, body->Patches().size(), body->AnchorBlob().size(),
                        gap / diag * 100.f, dyn.TotalEnergy(), t_deform, t_render);
        }
    }

    st.report((std::string("realtime deform [" + cfg.workload + " / " + cfg.path + "]")).c_str());
    renderer.Shutdown();
    return 0;
}