// =============================================================================
// playground - LIVE interactive playground for the DeliRium engine.
//
// Opens a window on the torus and lets you drive EVERY engine subsystem in
// real time and watch the GPU render the result. Aim = window center
// (mouse-look to move the aim, FPS-style). The modal physics runs on the
// engine's fixed-step loop, so pokes ring and decay live.
//
//   Build:  cmake --build build_delirium --target delirium_playground
//   Run:    ./build_delirium/delirium_playground          (from project root)
//
// Controls:
//   Mouse look .......... aim (window center is the hit point)
//   Wheel ............... zoom
//   LMB ................. poke (modal impulse toward camera -> rings & decays)
//   RMB ................. dent (RefineWhere + radial dent, permanent)
//   R ................... refine  (AMR split patches at aim)
//   C ................... coarsen (AMR merge at aim)
//   F ................... fracture (detach a fragment at aim; it flies off)
//   M ................... morph (color pattern K edit)
//   G ................... restitch (Seams Build + FitAll)
//   T ................... weld (HarmonicWeld global seam relaxation)
//   1 ................... render path: layered march (default)
//   2 ................... render path: GPU equation ray-march
//   3 ................... render path: classic splat
//   O ................... toggle auto-orbit
//   P ................... screenshot -> /tmp/playground_XXXX.ppm
//   Esc ................. quit
//   (WASD = move, Space/Ctrl = up/down, mouse look, wheel zoom)
//
// Every op prints a status line: patches, anchors, worst seam gap, modal
// energy. Watch for: seams opening at AMR size transitions, the fracture
// crack, the fragment flying off, and modal waves propagating.
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

struct Play {
    GameObject* torus = nullptr;
    SplatRenderer* renderer = nullptr;
    int w = 0, h = 0;
    int morph_pattern = 0;
    int shot = 0;
    float diag = 1.f;
    bool orbit = false;
    float orbit_yaw = 0.f;
    int render_path = 1;   // 1 march, 2 raymarch, 3 splat

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
};

bool on_event(const void* evp, Play* S) {
    const SDL_Event* ev = (const SDL_Event*)evp;
    if (ev->type == SDL_MOUSEBUTTONDOWN) {
        const Camera& cam = Engine::Instance().CurrentScene()->GetCamera();
        Vec3 p;
        if (!S->aim(cam, p)) return false;
        if (ev->button.button == SDL_BUTTON_LEFT) {
            // poke toward the camera (modal impulse -> rings & decays)
            Vec3 to_cam{cam.position.x - p.x, cam.position.y - p.y,
                        cam.position.z - p.z};
            float l = std::sqrt(to_cam.x*to_cam.x + to_cam.y*to_cam.y +
                                to_cam.z*to_cam.z);
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
        case SDLK_1: S->set_render_path(1); return true;
        case SDLK_2: S->set_render_path(2); return true;
        case SDLK_3: S->set_render_path(3); return true;
        case SDLK_o:
            S->orbit = !S->orbit;
            std::printf("  [orbit] %s\n", S->orbit ? "on" : "off");
            return true;
        case SDLK_p: {
            char path[128];
            std::snprintf(path, sizeof(path), "/tmp/playground_%04d.ppm", S->shot++);
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

// Per-frame hook: auto-orbit + composite the fracture fragment.
// `cam` is the SCENE camera (the engine passes it as a non-const ref, so
// the orbit can rotate it in place; the const in the hook signature is
// only the hook's declared type).
void on_frame(Play* S, SplatRenderer& r, Camera& cam) {
    if (S->orbit) {
        // rotate the camera position around the target (yaw only, keeps
        // the camera's height above the target)
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
    if (!engine.Initialize(1024, 768, "DeliRium playground")) return -1;
    SplatRenderer* rnd = engine.GetRenderer();
    if (!rnd) {
        std::fprintf(stderr, "no renderer (OpenCL?) -- nothing to show\n");
        return -1;
    }

    Scene* scene = engine.CreateScene("playground");
    // frame the torus (diag ~3.88): start ~10 units out, slightly above
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
    if (!torus->LoadModes("assets/torus.modes"))
        std::printf("(no modes file: object will render rigid)\n");

    const Vec3 sp = torus->Body().BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    Play S;
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
    std::printf("Playground ready. Aim = window center (mouse-look).\n"
                "  LMB poke   RMB dent   R refine   C coarsen   F fracture\n"
                "  M morph    G restitch  T weld     1/2/3 render path\n"
                "  O orbit    P screenshot   Esc quit\n");
#endif

    engine.Run();
    engine.Shutdown();
    return 0;
}