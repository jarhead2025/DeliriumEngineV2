// =============================================================================
// torus_live - LIVE toroidal deformation demo.
//
// Loads assets/torus.psf + assets/torus.modes into a GameObject (the engine
// steps the modal dynamics every frame), and drives a CONTINUOUS traveling
// wave: every 10 frames an impulse is injected at a point that rotates
// around the torus, so a wave perpetually propagates along the tube.
//
// The point of this demo: watch the march render path (fixed 2026-10-28:
// foot guarantee 0.5*foot -> foot, per-ray slab intersection) hold 60 FPS
// while the torus is continuously deforming, and poke/dent/fracture it
// live on top of the auto-wave.
//
//   Build:  cmake --build build_delirium --target delirium_torus_live
//   Run:    ./build_delirium/delirium_torus_live         (from project root)
//
// Controls:
//   Mouse look .......... aim (window center is the hit point)
//   Wheel ............... zoom
//   LMB ................. poke (modal impulse toward camera -> rings & decays)
//   RMB ................. dent (RefineWhere + radial dent, permanent)
//   R ................... refine  (AMR split patches at aim)
//   C ................... coarsen (AMR merge at aim)
//   F ................... fracture (detach a fragment at aim)
//   G ................... restitch (Seams Build + FitAll)
//   T ................... weld (HarmonicWeld global seam relaxation)
//   M ................... morph (color pattern K edit)
//   W ................... toggle auto-wave (default ON)
//   1 ................... render path: layered march (default)
//   2 ................... render path: GPU equation ray-march
//   3 ................... render path: classic splat
//   O ................... toggle auto-orbit
//   P ................... screenshot -> /tmp/torus_live_XXXX.ppm
//   Esc ................. quit
//   (WASD = move, Space/Ctrl = up/down, mouse look, wheel zoom)
//
// NOTE: the auto-wave uses PokeSurface (direct modal impulse at the nearest
// patch) rather than Poke (which raycasts + ClosestPoint), so it works even
// when the aim is off the surface. AMR (R/C) changes the patch/anchor count,
// which disables the modal WriteBack (n_surf/n_anchor guard) until the modes
// are reloaded -- the auto-wave then degrades to a no-op, same as the
// realtime_deform benchmark.
// =============================================================================
#include "delirium/core/Engine.h"
#include "delirium/core/GameObject.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

using namespace DeliRium;

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL

namespace {

// Nearest patch to a world point (patch-center distance) -- same as the
// realtime_deform benchmark. Used to aim the auto-wave impulse.
static uint32_t NearestPatch(const PSFBody& b, const Vec3& p) {
    auto& pool = b.AnchorsConst();
    uint32_t best = 0; float best_d = 1e30f;
    for (uint32_t pid = 0; pid < b.SurfaceBlob().size(); ++pid) {
        const auto& sr = b.SurfaceBlob()[pid];
        Vec3 ctr{};
        for (int k = 0; k < 4; ++k) {
            Vec3 a = pool.Position(sr.anchor_idx[k]);
            ctr.x += a.x * 0.25f; ctr.y += a.y * 0.25f; ctr.z += a.z * 0.25f;
        }
        const float dx = ctr.x - p.x, dy = ctr.y - p.y, dz = ctr.z - p.z;
        const float d = dx * dx + dy * dy + dz * dz;
        if (d < best_d) { best_d = d; best = pid; }
    }
    return best;
}

// A point on a sphere of radius r around the body center, at spherical
// angles (degrees). The torus tube centerline lies in the XZ plane, so
// sweeping yaw with pitch=0 rides the tube.
static Vec3 TubePoint(const PSFBody& b, float yaw_deg, float pitch_deg) {
    const Vec3 mn = b.BBoxMin(), sp = b.BBoxSpan();
    Vec3 c{ mn.x + sp.x * 0.5f, mn.y + sp.y * 0.5f, mn.z + sp.z * 0.5f };
    const float diag = std::sqrt(sp.x * sp.x + sp.y * sp.y + sp.z * sp.z);
    const float r = diag * 0.5f;   // tube centerline radius ~ outer radius
    const float yaw = yaw_deg * 3.14159265f / 180.f;
    const float pit = pitch_deg * 3.14159265f / 180.f;
    return { c.x + r * std::cos(pit) * std::sin(yaw),
             c.y + r * std::sin(pit),
             c.z + r * std::cos(pit) * std::cos(yaw) };
}

struct Demo {
    GameObject* torus = nullptr;
    SplatRenderer* renderer = nullptr;
    int w = 0, h = 0;
    float diag = 1.f;
    bool orbit = false;
    bool auto_wave = true;     // traveling wave, on by default
    float wave_yaw = 0.f;      // current injection angle (degrees)
    int frame = 0;
    int morph_pattern = 0;
    int shot = 0;
    int render_path = 1;       // 1 march, 2 raymarch, 3 splat

    // world point at the window center (the aim)
    bool aim(const Camera& cam, Vec3& out) const {
        Vec3 o, d;
        cam.PixelRay(w * 0.5f, h * 0.5f, w, h, o, d);
        SurfaceHit hit;
        if (!torus->Body().Raycast(o, d, hit)) return false;
        out = hit.p;
        return true;
    }

    void status(const char* what) {
        PSFBody& b = torus->Body();
        b.Seams().Build(b);
        const float g = b.Seams().MaxGap(b);
        std::printf("  [%s] patches=%zu anchors=%zu seam=%.4f%% diag  E=%.4f\n",
                    what, b.Patches().size(), b.AnchorBlob().size(),
                    g / diag * 100.f, torus->Dynamics().TotalEnergy());
    }

    void set_render_path(int p) {
        render_path = p;
        // RenderBody reads these env vars every frame, so this takes effect
        // on the very next frame.
        if (p == 1) {
            ::setenv("DELIRIUM_MARCH", "1", 1);
            ::unsetenv("DELIRIUM_RAYMARCH");
        } else if (p == 2) {
            ::setenv("DELIRIUM_RAYMARCH", "1", 1);
            ::setenv("DELIRIUM_MARCH", "0", 1);
        } else {
            ::setenv("DELIRIUM_MARCH", "0", 1);
            ::unsetenv("DELIRIUM_RAYMARCH");
        }
        std::printf("  [path] render path -> %s\n",
                    p == 1 ? "layered march" : p == 2 ? "GPU ray-march" : "classic splat");
    }

    // Per-frame auto-wave: every 10 frames, inject a modal impulse at a
    // point that has rotated 24 deg around the tube -> a wave that travels
    // around the torus forever. The engine's FixedUpdate steps the dynamics
    // and writes back; this only SEEDS energy.
    void tick_wave() {
        if (!auto_wave) return;
        if (frame % 10 != 0) return;
        wave_yaw += 24.0f;
        if (wave_yaw >= 360.f) wave_yaw -= 360.f;
        Vec3 p = TubePoint(torus->Body(), wave_yaw, 0.f);
        uint32_t s = NearestPatch(torus->Body(), p);
        torus->PokeSurface(s, 0.5f, 0.5f, 0.35f);
        if (frame % 60 == 0)
            std::printf("  [wave] impulse @ yaw %.0f deg  E=%.4f\n",
                        wave_yaw, torus->Dynamics().TotalEnergy());
    }
};

bool on_event(const void* evp, Demo* S) {
    const SDL_Event* ev = (const SDL_Event*)evp;
    if (ev->type == SDL_MOUSEBUTTONDOWN) {
        const Camera& cam = Engine::Instance().CurrentScene()->GetCamera();
        Vec3 p;
        if (!S->aim(cam, p)) return false;
        if (ev->button.button == SDL_BUTTON_LEFT) {
            // poke toward the camera (modal impulse -> rings & decays)
            Vec3 to_cam{cam.position.x - p.x, cam.position.y - p.y,
                        cam.position.z - p.z};
            float l = std::sqrt(to_cam.x * to_cam.x + to_cam.y * to_cam.y +
                                to_cam.z * to_cam.z);
            if (l > 1e-6f) { to_cam.x /= l; to_cam.y /= l; to_cam.z /= l; }
            S->torus->Poke(p, {to_cam.x * 0.6f, to_cam.y * 0.6f, to_cam.z * 0.6f});
            S->status("poke");
            return true;
        }
        if (ev->button.button == SDL_BUTTON_RIGHT) {
            S->torus->DentLocally(p, 0.22f * S->diag, 0.08f * S->diag);
            S->status("dent");
            return true;
        }
        return false;
    }
    if (ev->type == SDL_KEYDOWN) {
        const Camera& cam = Engine::Instance().CurrentScene()->GetCamera();
        Vec3 p;
        const bool hit = S->aim(cam, p);
        switch (ev->key.keysym.sym) {
        case SDLK_r:
            if (hit) {
                PSFBody& b = S->torus->Body();
                const size_t before = b.Patches().size();
                b.RefineWhere(p, 0.30f * S->diag, 0.08f * S->diag, 64);
                std::printf("  [refine] +%zu patches\n",
                            b.Patches().size() - before);
                S->status("refine");
            }
            return true;
        case SDLK_c:
            if (hit) {
                PSFBody& b = S->torus->Body();
                const size_t before = b.Patches().size();
                const int m = b.CoarsenWhere(p, 0.40f * S->diag, 0.004f, 64);
                std::printf("  [coarsen] %d merges, -%zu patches\n", m,
                            before - b.Patches().size());
                S->status("coarsen");
            }
            return true;
        case SDLK_f:
            if (hit) {
                { Vec3 f{cam.target.x-cam.position.x, cam.target.y-cam.position.y,
                         cam.target.z-cam.position.z};
                  float fl = std::sqrt(f.x*f.x+f.y*f.y+f.z*f.z);
                  if (fl>1e-9f){ f.x/=fl; f.y/=fl; f.z/=fl; }
                  S->torus->Fracture(p, 1.0f, f); }
                S->status("fracture");
            }
            return true;
        case SDLK_m:
            S->torus->Morph((uint32_t)(S->morph_pattern++ % 16), 0.25f);
            std::printf("  [morph] pattern %d\n", (S->morph_pattern - 1) % 16);
            return true;
        case SDLK_g:
            S->torus->RecalculateStructure();
            S->status("restitch");
            return true;
        case SDLK_t: {
            PSFBody& b = S->torus->Body();
            b.Seams().Build(b);
            const float gap = b.Seams().HarmonicWeld(b, 24, 0.2f);
            std::printf("  [weld] HarmonicWeld gap=%.4f%% diag\n",
                        gap / S->diag * 100.f);
            S->status("weld");
            return true;
        }
        case SDLK_w:
            S->auto_wave = !S->auto_wave;
            std::printf("  [wave] auto-wave %s\n", S->auto_wave ? "ON" : "OFF");
            return true;
        case SDLK_1: S->set_render_path(1); return true;
        case SDLK_2: S->set_render_path(2); return true;
        case SDLK_3: S->set_render_path(3); return true;
        case SDLK_o:
            S->orbit = !S->orbit;
            std::printf("  [orbit] %s\n", S->orbit ? "on" : "off");
            return true;
        case SDLK_p: {
            char path[128];
            std::snprintf(path, sizeof(path), "/tmp/torus_live_%04d.ppm", S->shot++);
            if (S->renderer->SaveFramePPM(path))
                std::printf("  [shot] %s\n", path);
            else
                std::printf("  [shot] save failed\n");
            return true;
        }
        default:
            return false;
        }
    }
    return false;
}

// Per-frame hook: auto-wave seeding + auto-orbit + fracture fragment.
// The engine renders the scene's objects FIRST (torus, clear=true), so this
// hook only adds the fragment on top (clear=false).
void on_frame(Demo* S, SplatRenderer& r, Camera& cam) {
    S->tick_wave();
    S->frame++;
    if (S->orbit) {
        const float d = 0.02f;
        const float s = std::sin(d), c = std::cos(d);
        const float rx = cam.position.x - cam.target.x;
        const float rz = cam.position.z - cam.target.z;
        cam.position.x = cam.target.x + (c * rx + s * rz);
        cam.position.z = cam.target.z + (-s * rx + c * rz);
    }
    PSFBody* frag = S->torus->Fragment();
    if (frag) r.RenderBody(*frag, cam, /*clear=*/false);
}

} // namespace

#endif // DELIRIUM_HAS_SDL

int main() {
    Engine& engine = Engine::Instance();
    if (!engine.Initialize(1024, 768, "DeliRium torus live")) return -1;
    SplatRenderer* rnd = engine.GetRenderer();
    if (!rnd) {
        std::fprintf(stderr, "no renderer (OpenCL?) -- nothing to show\n");
        return -1;
    }

    Scene* scene = engine.CreateScene("torus_live");
    // frame the torus (diag ~3.88): ~10 units out, slightly above
    scene->GetCamera().position = {0.f, 2.5f, 10.f};
    scene->GetCamera().target   = {0.f, 0.f, 0.f};
    scene->GetCamera().up       = {0.f, 1.f, 0.f};

    ReconstructionRequest req;
    req.semantic = "file:assets/torus.psf";
    req.material.youngs_modulus = 5e5f;
    GameObject* torus = scene->Create("torus", req);
    if (!torus->HasBody()) {
        std::printf("ERROR: could not load assets/torus.psf\n"
                    "Run from the PROJECT ROOT (the folder containing 'assets/').\n");
        return -1;
    }
    if (!torus->LoadModes("assets/torus.modes")) {
        std::printf("ERROR: could not load assets/torus.modes -- the auto-wave "
                    "needs modal modes.\n");
        return -1;
    }

    const Vec3 sp = torus->Body().BBoxSpan();
    const float diag = std::sqrt(sp.x * sp.x + sp.y * sp.y + sp.z * sp.z);

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    Demo S;
    S.torus = torus;
    S.renderer = rnd;
    S.w = 1024; S.h = 768;
    S.diag = diag;
    engine.SetInputHook([&S](const void* ev) { return on_event(ev, &S); });
    engine.SetRenderHook([&S](SplatRenderer& r, const Camera& cam) {
        // The engine binds this const ref to the scene's real (non-const)
        // camera, so the const_cast is safe and lets the orbit rotate it.
        on_frame(&S, r, const_cast<Camera&>(cam));
    });
    std::printf("Torus live ready. A continuous wave travels around the tube.\n"
                "Aim = window center (mouse-look).\n"
                "  LMB poke   RMB dent   R refine   C coarsen   F fracture\n"
                "  M morph    G restitch  T weld     W toggle wave\n"
                "  1/2/3 render path   O orbit   P screenshot   Esc quit\n");
#endif

    engine.Run();
    engine.Shutdown();
    return 0;
}