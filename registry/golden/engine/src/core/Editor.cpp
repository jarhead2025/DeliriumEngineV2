// =============================================================================
// Editor.cpp - input handling + live HUD for the interactive editor.
// =============================================================================
#include "delirium/core/Editor.h"
#include "delirium/core/Scene.h"
#include "delirium/physics/PhysicsWorld.h"
#include <SDL2/SDL.h>
#include <cmath>

namespace DeliRium {

bool Editor::OnInput(const void* evp) {
    const SDL_Event& ev = *static_cast<const SDL_Event*>(evp);
    if (!target || !target->HasBody()) return false;
    PSFBody& body = target->Body();

    // ---- tool / material selection -----------------------------------
    if (ev.type == SDL_KEYDOWN) {
        switch (ev.key.keysym.sym) {
            case SDLK_1: tool = Tool::Dent;     std::snprintf(last_action,64,"tool: dent");    return true;
            case SDLK_2: tool = Tool::Poke;     std::snprintf(last_action,64,"tool: poke");    return true;
            case SDLK_3: tool = Tool::Refine;   std::snprintf(last_action,64,"tool: refine");  return true;
            case SDLK_4: tool = Tool::Coarsen;  std::snprintf(last_action,64,"tool: coarsen"); return true;
            case SDLK_5: tool = Tool::Fracture; std::snprintf(last_action,64,"tool: fracture");return true;
            case SDLK_6: tool = Tool::Morph;    std::snprintf(last_action,64,"tool: morph");   return true;
            case SDLK_7: tool = Tool::Weld;     std::snprintf(last_action,64,"tool: weld");    return true;
            case SDLK_m:                                       // cycle material
                mat_idx = (mat_idx + 1) % 5;
                target->Mat().weld = (WeldCategory)mat_idx;
                std::snprintf(last_action,64,"material: %s", MatName(mat_idx));
                return true;
            case SDLK_r:                                       // reset shape
                target->Dynamics().Reset();
                target->Dynamics().WriteBack(body);
                std::snprintf(last_action,64,"reset");
                return true;
            case SDLK_w: {                                     // weld now
                WeldProfile p = WeldProfileFor((WeldCategory)mat_idx);
                body.Seams().Build(body);
                const float g = body.Seams().HarmonicWeld(body, p);
                std::snprintf(last_action,64,"welded gap %.3f%%",
                              g / Diag() * 100);
                return true;
            }
            default: return false;
        }
    }

    // ---- mouse: apply the active tool at the picked surface point ----
    if (ev.type == SDL_MOUSEBUTTONDOWN) {
        Scene* sc = engine.CurrentScene();
        if (!sc) return false;
        Vec3 ro, rd;
        sc->GetCamera().PixelRay((float)ev.button.x, (float)ev.button.y,
                                 800, 600, ro, rd);
        SurfaceHit hit;
        if (!body.Raycast(ro, rd, hit)) return true;   // missed: consume
        const Vec3 P = hit.p;
        const float diag = Diag();

        switch (tool) {
            case Tool::Dent:
                target->DentLocally(P, 0.21f, 0.042f);
                std::snprintf(last_action,64,"dent @ patch %u", hit.patch);
                break;
            case Tool::Poke: {
                const Vec3 imp = { -rd.x*30, -rd.y*30, -rd.z*30 };
                target->Poke(P, imp);
                std::snprintf(last_action,64,"poke @ patch %u", hit.patch);
                break;
            }
            case Tool::Refine: {
                body.RefineWhere(P, 0.12f*diag, 0.06f*diag);
                std::snprintf(last_action,64,"refine -> %zu patches",
                              body.Patches().size());
                break;
            }
            case Tool::Coarsen: {
                const int n = body.CoarsenWhere(P, 0.18f*diag, 0.004f);
                std::snprintf(last_action,64,"coarsen %d -> %zu", n,
                              body.Patches().size());
                break;
            }
            case Tool::Fracture: {
                target->Fracture(P, 10.0f);
                std::snprintf(last_action,64,"fracture -> %zu + frag",
                              body.Patches().size());
                break;
            }
            case Tool::Morph: {
                morph_k += (ev.button.button == SDL_BUTTON_LEFT) ? 0.1f : -0.1f;
                target->Morph(0, (ev.button.button==SDL_BUTTON_LEFT)?0.1f:-0.1f);
                std::snprintf(last_action,64,"morph K0 %+.1f", morph_k);
                break;
            }
            case Tool::Weld: {                          // local request -> physics
                if (PhysicsWorld* pw = engine.GetPhysicsWorld())
                    pw->RequestWeld(target, 3.0f);
                std::snprintf(last_action,64,"weld requested");
                break;
            }
        }
        return true;
    }
    return false;
}

float Editor::Diag() const {
    const Vec3 sp = target->Body().BBoxSpan();
    return std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
}

std::string Editor::Hud() {
    if (!target || !target->HasBody()) return "no target";
    PSFBody& b = target->Body();
    const size_t np = b.Patches().size();
    b.Seams().Build(b);
    const float gap = b.Seams().MaxGap(b) / Diag() * 100.0f;
    const size_t bytes = 64 + np*64 + b.AnchorBlob().size()*48 + np*8;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%s [%s] | %zu patches | %.1f KB | gap %.3f%% | %s",
        ToolName(tool), MatName(mat_idx), np, bytes/1024.0f, gap, last_action);
    return std::string(buf);
}

} // namespace DeliRium
