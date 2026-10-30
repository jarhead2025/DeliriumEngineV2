// =============================================================================
// 03_visual_stress - live visual stress test.
// Loads the torus, then lets you hammer it with every live operation and
// watch the GPU render the result in real time. Aim = window center
// (mouse-look to move the aim, FPS-style).
//
//   Left click   poke (modal impulse toward camera)
//   Right click  dent (RefineWhere + radial dent)
//   R            refine (AMR split patches at aim)
//   C            coarsen (AMR merge at aim)
//   F            fracture (detach a fragment at aim)
//   M            morph (color K pattern)
//   G            re-stitch everything (RecalculateStructure)
//   S            save frame to /tmp/stress_XXXX.ppm
//   Esc          quit
//
// Every op prints a status line: patches, anchors, worst seam gap,
// modal energy. Watch for seams opening, holes at fracture edges,
// shading creases at AMR size transitions, and the fragment flying off.
// =============================================================================
#include "delirium/core/Engine.h"
#include "delirium/core/GameObject.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"
#include <cmath>
#include <cstdio>
#include <string>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

using namespace DeliRium;

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL

namespace {
struct Stress {
    GameObject* torus = nullptr;
    SplatRenderer* renderer = nullptr;
    int w = 0, h = 0;
    int morph_pattern = 0;
    int shot = 0;
    float diag = 1.f;

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
};

bool on_event(const void* evp, Stress* S) {
    const SDL_Event* ev = (const SDL_Event*)evp;
    if (ev->type == SDL_MOUSEBUTTONDOWN) {
        const Camera& cam = Engine::Instance().CurrentScene()->GetCamera();
        Vec3 p;
        if (!S->aim(cam, p)) return false;
        if (ev->button.button == SDL_BUTTON_LEFT) {
            // poke toward the camera
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
                S->torus->Fracture(p, 1.0f);
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
        case SDLK_s: {
            char path[128];
            std::snprintf(path, sizeof(path), "/tmp/stress_%04d.ppm", S->shot++);
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
} // namespace

#endif // DELIRIUM_HAS_SDL

int main() {
    Engine& engine = Engine::Instance();
    if (!engine.Initialize(1024, 768, "DeliRium v0.3 - visual stress")) return -1;

    Scene* scene = engine.CreateScene("stress");
    scene->GetCamera().position = {2.6f, -2.2f, 1.6f};
    scene->GetCamera().target   = {0, 0, 0};
    scene->GetCamera().up       = {0, 0, 1};

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
    Stress S;
    S.torus = torus;
    S.renderer = engine.GetRenderer();
    S.w = 1024; S.h = 768;
    S.diag = diag;
    engine.SetInputHook([&S](const void* ev) { return on_event(ev, &S); });
    // composite the fracture fragment (if any) into the same frame
    engine.SetRenderHook([&S](SplatRenderer& r, const Camera& cam) {
        PSFBody* frag = S.torus->Fragment();
        if (frag) r.RenderBody(*frag, cam, false);
    });
    std::printf("Visual stress ready. Aim = window center (mouse-look).\n"
                "  LMB poke   RMB dent   R refine   C coarsen\n"
                "  F fracture  M morph   G restitch  S screenshot   Esc quit\n");
#endif

    engine.Run();
    engine.Shutdown();
    return 0;
}