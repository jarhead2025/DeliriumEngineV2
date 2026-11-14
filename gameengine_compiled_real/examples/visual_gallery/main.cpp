// =============================================================================
// visual_gallery - HEADLESS render gallery. No window, no display needed.
//
// Renders one PPM per engine subsystem / render path so you can SEE what each
// piece of the engine actually produces. Every scene loads a FRESH body so the
// scenes are independent (a dent in one never leaks into the next). Each scene
// prints a one-line "what to look for" note; a manifest is printed at the end.
//
//   Build:  cmake --build build_delirium --target delirium_visual_gallery
//   Run:    ./build_delirium/delirium_visual_gallery            (from project root)
//   Options:
//     --out DIR      output dir        (default: visual_gallery)
//     --size WxH     frame size        (default: 640x480)
//     --only A,B,C   render only scenes A,B,C (letters below)
//     --list         print the manifest and exit (no render)
//
// Scenes (letter = --only selector):
//   A  march_default      layered lateral march (the default visible path)
//   B  raymarch           GPU equation ray-march (glow / emissive look)
//   C  splat_classic      classic anisotropic splat (psf_splat.cl)
//   D  march_highres      march at UVN=12 (sharper, more samples)
//   E  amr_refine         AMR: dense patches on one side (live SplitPatch)
//   F  amr_lod            AMR: refine one region + coarsen another (LOD contrast)
//   G  dent               radial dent (anchors move along normals)
//   H  dent_refit         dent + RefitLocal (coefficients re-optimized)
//   I  mode0              ROM mode 0, static deflection
//   J  mode1              ROM mode 1, static deflection
//   K  modal_ring         modal impulse + N steps (wave frozen mid-flight)
//   L  fracture           detach a fragment + offset it (the crack)
//   M  morph_color        color-pattern K edit (Morph)
//   N  sphere             assets/sphere.psf
//   O  asteroid           assets/asteroid.psf
//   P  roughball          assets/roughball.psf
//   Q  closeup            tight camera on the torus
//   W  closeup_layered    Q's camera/body on the LAYERED path (the pixel-
//                          fidelity pair: same body, same camera, only the
//                          compositing differs)
//   X  closeup_classic    Q's camera/body on the CLASSIC splat (the
//                          reference: it never invents a blended colour)
//   R  orbit              torus from a low, wide angle
//   S  split_edges        a few splits (new patch edges / seams)
//   T  weld               HarmonicWeld (global seam relaxation)
// =============================================================================
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/physics/ModalDynamics.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

using namespace DeliRium;

// --- tiny helpers -----------------------------------------------------------
// POSIX setenv/unsetenv live in the global namespace, not std:: (libstdc++
// only exposes them via <cstdlib> as ::setenv under _GNU_SOURCE / POSIX).
static void setenv_(const char* k, const char* v) { ::setenv(k, v, 1); }
static void unsetenv_(const char* k) { ::unsetenv(k); }
// std::string has no single-char constructor: build it from (count, char).
static std::string L(char c) { return std::string(1, c); }

static void mkdir_p(const std::string& p) {
    if (p.empty()) return;
    std::string cur;
    for (size_t i = 0; i < p.size(); ++i) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size())
            ::mkdir(cur.c_str(), 0755);   // EEXIST is fine
    }
}

// Frame a body from a spherical camera: yaw/pitch in degrees, dist_factor
// scales the object diagonal. Center is the bbox center.
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

static std::unique_ptr<PSFBody> Load(const char* path) {
    return PSFBody::FromFile(path);
}

// Render one body and save it. Returns true on success.
static bool Shot(SplatRenderer& r, const std::string& outdir,
                 const std::string& name, PSFBody& body,
                 const Camera& cam, bool clear = true) {
    const std::string path = outdir + "/" + name;
    if (!r.RenderBody(body, cam, clear)) {
        std::printf("  [FAIL] %s: RenderBody returned false\n", name.c_str());
        return false;
    }
    if (!r.SaveFramePPM(path)) {
        std::printf("  [FAIL] %s: SaveFramePPM failed\n", name.c_str());
        return false;
    }
    return true;
}

// Aim at a point on the body from a camera direction; returns (patch,u,v).
static bool AimPoint(PSFBody& b, const Vec3& from, const Vec3& to,
                     uint32_t& s, float& u, float& v) {
    Vec3 d{ to.x - from.x, to.y - from.y, to.z - from.z };
    float l = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    if (l < 1e-9f) return false;
    d.x /= l; d.y /= l; d.z /= l;
    SurfaceHit h;
    if (!b.Raycast(from, d, h)) return false;
    s = h.patch; u = h.u; v = h.v;
    return true;
}

// --- scene definitions ------------------------------------------------------
// Each returns true if the PPM was written. `ctx` carries the shared renderer
// + output dir + asset paths. A fresh body is loaded per scene.
struct Ctx {
    SplatRenderer* r;
    std::string out;
    const char* torus = "assets/torus.psf";
    const char* modes = "assets/torus.modes";
    int w = 640, h = 480;
};

// A) default march path
static bool scene_march_default(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "A_march_default.ppm", *b, cam);
    std::printf("  [A] march_default      <- layered lateral march (default visible path)\n");
    return ok;
}

// B) GPU equation raymarch.
//
// THE PATH IS SET THROUGH THE API, NOT THE ENV VARS: the renderer's
// active path is a runtime value (`path_`) that RenderBody reads EVERY
// FRAME, because the editor needs to switch paths live. The env vars
// (DELIRIUM_MARCH / DELIRIUM_RAYMARCH) are read ONCE at Initialize, so
// setenv() here has NO effect -- before this fix, scene B silently
// rendered the MARCH path and the "raymarch" gallery image was a lie.
static bool scene_raymarch(Ctx& c) {
    c.r->SetPath(SplatRenderer::Path::Raymarch);
    setenv_("DELIRIUM_GLOW", "0.45");
    auto b = Load(c.torus);
    bool ok = false;
    if (b) {
        Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
        ok = Shot(*c.r, c.out, "B_raymarch.ppm", *b, cam);
    }
    unsetenv_("DELIRIUM_GLOW");
    c.r->SetPath(SplatRenderer::Path::March);
    std::printf("  [B] raymarch           <- GPU equation ray-march (emissive glow)\n");
    return ok;
}

// C) classic splat (the per-patch splat with ONE winner per pixel).
static bool scene_splat_classic(Ctx& c) {
    c.r->SetPath(SplatRenderer::Path::SplatClassic);
    auto b = Load(c.torus);
    bool ok = false;
    if (b) {
        Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
        ok = Shot(*c.r, c.out, "C_splat_classic.ppm", *b, cam);
    }
    c.r->SetPath(SplatRenderer::Path::March);
    std::printf("  [C] splat_classic      <- anisotropic per-patch splat (psf_splat.cl)\n");
    return ok;
}

// C2) LAYERED splat -- the user's distance-plane method.
//
// The impacts are bucketed into DISTANCE SHELLS along the view axis,
// each shell is averaged within itself, and the shells are stacked BACK
// TO FRONT so the nearest wins. This is scene C's exact counterpart:
// same body, same camera, same samples -- only the compositing differs.
// Comparing the two images side by side is how the method is judged.
static bool scene_splat_layered(Ctx& c) {
    c.r->SetPath(SplatRenderer::Path::SplatLayered);
    auto b = Load(c.torus);
    bool ok = false;
    if (b) {
        Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
        ok = Shot(*c.r, c.out, "C2_splat_layered.ppm", *b, cam);
    }
    c.r->SetPath(SplatRenderer::Path::March);
    std::printf("  [C2] splat_layered     <- distance-plane shells, averaged, stacked back-to-front\n");
    return ok;
}

// V) AREA-WEIGHTED splat -- the pixel-footprint pyramid.
//
// The classic splat lands every surface sample on EXACTLY ONE pixel. This
// path distributes each sample over the pixels its footprint covers -- the
// footprint grows linearly with the distance (a pixel subtends a fixed
// ANGLE, so it covers more world space further away) and stretches on a
// grazing surface -- and the pixel's colour is the AREA-WEIGHTED MEAN of
// everything that touched it. The weights are fixed-point integers summed
// with atomic_add and divided with an integer divide, so the result is
// exact and deterministic.
//
// It is scene C's exact counterpart: same body, same camera, same samples
// -- only the compositing differs. Comparing the images side by side is
// how the method is judged.
static bool scene_splat_area(Ctx& c) {
    c.r->SetPath(SplatRenderer::Path::SplatArea);
    auto b = Load(c.torus);
    bool ok = false;
    if (b) {
        Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
        ok = Shot(*c.r, c.out, "C3_splat_area.ppm", *b, cam);
    }
    c.r->SetPath(SplatRenderer::Path::March);
    std::printf("  [C3] splat_area        <- pixel-footprint area weighting\n");
    return ok;
}

// D) high-res march
static bool scene_march_highres(Ctx& c) {
    setenv_("DELIRIUM_MARCH_UVN", "12");
    setenv_("DELIRIUM_MARCH_STEPS", "64");
    setenv_("DELIRIUM_MARCH_FILL", "4");
    auto b = Load(c.torus);
    bool ok = false;
    if (b) {
        Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
        ok = Shot(*c.r, c.out, "D_march_highres.ppm", *b, cam);
    }
    unsetenv_("DELIRIUM_MARCH_UVN");
    unsetenv_("DELIRIUM_MARCH_STEPS");
    unsetenv_("DELIRIUM_MARCH_FILL");
    std::printf("  [D] march_highres      <- march UVN=12 (sharper; compare vs A)\n");
    return ok;
}

// E) AMR refine one side
static bool scene_amr_refine(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    const Vec3 mn = b->BBoxMin(), sp = b->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    // refine the +X side of the torus down to ~1/8 of the diagonal
    Vec3 center{ mn.x + sp.x, mn.y + sp.y*0.5f, mn.z + sp.z*0.5f };
    const size_t before = b->Patches().size();
    b->RefineWhere(center, 0.6f*diag, 0.12f*diag, 128);
    const size_t after = b->Patches().size();
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "E_amr_refine.ppm", *b, cam);
    std::printf("  [E] amr_refine         <- live SplitPatch on +X side (%zu -> %zu patches)\n",
                before, after);
    return ok;
}

// F) AMR LOD contrast: refine +X, coarsen -X
static bool scene_amr_lod(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    const Vec3 mn = b->BBoxMin(), sp = b->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    Vec3 cx{ mn.x + sp.x, mn.y + sp.y*0.5f, mn.z + sp.z*0.5f };
    Vec3 ccx{ mn.x, mn.y + sp.y*0.5f, mn.z + sp.z*0.5f };
    b->RefineWhere(cx, 0.6f*diag, 0.12f*diag, 128);
    const int merges = b->CoarsenWhere(ccx, 0.6f*diag, 0.004f, 128);
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "F_amr_lod.ppm", *b, cam);
    std::printf("  [F] amr_lod            <- fine on +X, coarse on -X (%d merges; watch LOD seams)\n",
                merges);
    return ok;
}

// G) radial dent
static bool scene_dent(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    const Vec3 mn = b->BBoxMin(), sp = b->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    Vec3 p{ mn.x + sp.x*0.5f, mn.y + sp.y, mn.z + sp.z*0.5f };  // top of the tube
    b->ApplyRadialDent(p, 0.35f*diag, 0.10f*diag);
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "G_dent.ppm", *b, cam);
    std::printf("  [G] dent               <- radial dent on top (compare vs A)\n");
    return ok;
}

// H) dent + local refit
static bool scene_dent_refit(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    const Vec3 mn = b->BBoxMin(), sp = b->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    Vec3 p{ mn.x + sp.x*0.5f, mn.y + sp.y, mn.z + sp.z*0.5f };
    b->ApplyRadialDent(p, 0.35f*diag, 0.10f*diag);
    b->RefitLocal(p, 0.5f*diag);
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "H_dent_refit.ppm", *b, cam);
    std::printf("  [H] dent_refit         <- dent + RefitLocal (coefs re-optimized)\n");
    return ok;
}

// I/J) static ROM mode deflections
static bool scene_mode(Ctx& c, int mode, char letter, const char* name) {
    (void)name;
    auto b = Load(c.torus);
    if (!b) return false;
    ModalDynamics dyn;
    bool have = dyn.LoadModes(c.modes, *b);
    if (have) {
        dyn.SetAmplitude(mode, 0.008f);   // small static deflection
        dyn.WriteBack(*b);
    }
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    std::string fn = L(letter) + "_mode" + std::to_string(mode) + ".ppm";
    bool ok = Shot(*c.r, c.out, fn, *b, cam);
    std::printf("  [%c] mode%d              <- ROM mode %d static%s\n", letter, mode, mode,
                have ? "" : " (no modes file!)");
    return ok;
}

// K) modal impulse, frozen mid-ring
static bool scene_modal_ring(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    ModalDynamics dyn;
    bool have = dyn.LoadModes(c.modes, *b);
    if (have) {
        // seed at the TOP of the torus: highest patch center (a ray from
        // above would pass through the hole). u=v=0.5 is the patch center.
        auto& pool = b->AnchorsConst();
        uint32_t best = 0; float best_y = -1e30f;
        for (uint32_t pid = 0; pid < b->SurfaceBlob().size(); ++pid) {
            const auto& sr = b->SurfaceBlob()[pid];
            float y = 0.f;
            for (int k = 0; k < 4; ++k) y += pool.Position(sr.anchor_idx[k]).y;
            if (y * 0.25f > best_y) { best_y = y * 0.25f; best = pid; }
        }
        dyn.ApplyImpulseAt(best, 0.5f, 0.5f, 0.35f);
        for (int i = 0; i < 24; ++i) {   // ~0.2 s of modal time
            dyn.Step(1.0f/120.0f);
            dyn.WriteBack(*b);
        }
    }
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "K_modal_ring.ppm", *b, cam);
    std::printf("  [K] modal_ring         <- impulse at top, 24 modal steps (frozen wave)\n");
    return ok;
}

// L) fracture: detach the far half, offset it
static bool scene_fracture(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    const Vec3 mn = b->BBoxMin(), sp = b->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    // split plane: vertical, through center, normal +Z. Detach the +Z half.
    const Vec3 n{0, 0, 1};
    Vec3 cen{ mn.x + sp.x*0.5f, mn.y + sp.y*0.5f, mn.z + sp.z*0.5f };
    const float d = cen.z;
    std::vector<uint32_t> ids;
    auto& pool = b->AnchorsConst();
    for (uint32_t pid = 0; pid < b->SurfaceBlob().size(); ++pid) {
        const auto& sr = b->SurfaceBlob()[pid];
        Vec3 ctr{};
        for (int k = 0; k < 4; ++k) {
            Vec3 a = pool.Position(sr.anchor_idx[k]);
            ctr.x += a.x*0.25f; ctr.y += a.y*0.25f; ctr.z += a.z*0.25f;
        }
        if (n.z*ctr.z - d > 0.f) ids.push_back(pid);
    }
    if (ids.empty() || ids.size() == b->SurfaceBlob().size()) {
        std::printf("  [L] fracture         <- no valid split plane, skipped\n");
        return false;
    }
    auto frag = b->DetachPatches(ids);
    if (!frag) {
        std::printf("  [L] fracture         <- DetachPatches failed\n");
        return false;
    }
    // offset the fragment away from the main body so the crack is visible.
    // Grow the fragment's bbox first so SetPosition doesn't clamp the
    // displaced anchors (same trick NPC::SetPose uses).
    frag->ExpandBBox(0.5f);
    auto& fpool = frag->Anchors();
    for (uint16_t a = 0; a < (uint16_t)frag->AnchorBlob().size(); ++a) {
        Vec3 p = fpool.Position(a);
        fpool.SetPosition(a, {p.x + 0.12f*diag, p.y + 0.06f*diag, p.z + 0.18f*diag});
    }
    frag->MarkAnchorsDirty();
    b->Seams().Build(*b);
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.9f);
    bool ok = Shot(*c.r, c.out, "L_fracture.ppm", *b, cam);
    if (ok) ok = Shot(*c.r, c.out, "L_fracture.ppm", *frag, cam, /*clear=*/false);
    std::printf("  [L] fracture           <- %zu patches detached + offset (the crack)\n", ids.size());
    return ok;
}

// M) color morph
static bool scene_morph_color(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    // bump pattern 9's K on every patch (pattern 9 is present on most of the
    // torus -- verified in the .psf color_pattern slots).
    for (auto& p : const_cast<std::vector<Patch>&>(b->Patches()))
        p.AddColorK(9, 0.6f);
    b->MarkAnchorsDirty();
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "M_morph_color.ppm", *b, cam);
    std::printf("  [M] morph_color        <- pattern-9 K +0.6 (compare vs A)\n");
    return ok;
}

// N/O/P) other assets
static bool scene_asset(Ctx& c, char letter, const char* path, const char* name) {
    auto b = Load(path);
    if (!b) {
        std::printf("  [%c] %-18s <- MISSING %s (skipped)\n", letter, name, path);
        return false;
    }
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, L(letter) + "_" + name + ".ppm", *b, cam);
    std::printf("  [%c] %-18s <- %s\n", letter, name, path);
    return ok;
}

// Q) closeup
static bool scene_closeup(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    Camera cam = FrameCamera(*b, 30.f, 22.f, 1.35f);
    bool ok = Shot(*c.r, c.out, "Q_closeup.ppm", *b, cam);
    std::printf("  [Q] closeup            <- tight camera (patch detail / seams)\n");
    return ok;
}

// W) closeup on the LAYERED path -- scene Q's EXACT counterpart.
//
// WHY THIS EXISTS: the pixel-fidelity question ("is the render faithful to
// the screen's pixel scale, or just a square when close up?") needs the
// SAME camera and the SAME body on both paths, so the only difference is
// the compositing. Q uses whatever path was last set (the gallery resets
// to March after each scene), so Q alone cannot compare the paths.
//
// AND THE ENV VAR CANNOT DO IT: the gallery calls SetPath() per scene,
// which marks the path EXPLICIT -- and an explicit path SUPERSEDES the
// env-var seed (SplatRenderer::Initialize only seeds when the path is not
// explicit). So DELIRIUM_SPLAT_LAYERS=1 --only Q renders the MARCH path and
// the numbers come out byte-identical to a classic run. That is the same
// trap as G19: before blaming a knob, check whether it reaches the thing.
static bool scene_closeup_layered(Ctx& c) {
    c.r->SetPath(SplatRenderer::Path::SplatLayered);
    auto b = Load(c.torus);
    if (!b) { c.r->SetPath(SplatRenderer::Path::March); return false; }
    Camera cam = FrameCamera(*b, 30.f, 22.f, 1.35f);   // Q's camera, exactly
    bool ok = Shot(*c.r, c.out, "W_closeup_layered.ppm", *b, cam);
    c.r->SetPath(SplatRenderer::Path::March);
    std::printf("  [W] closeup_layered    <- Q's camera/body on the LAYERED path\n");
    return ok;
}

// X) closeup on the CLASSIC splat -- the third member of the Q/W/X set.
//
// The pixel-fidelity comparison needs a REFERENCE: the classic path never
// invents a blended colour (a pixel is an impact's own colour or EMPTY),
// so it is the baseline the layered path is measured against.
static bool scene_closeup_classic(Ctx& c) {
    c.r->SetPath(SplatRenderer::Path::SplatClassic);
    auto b = Load(c.torus);
    if (!b) { c.r->SetPath(SplatRenderer::Path::March); return false; }
    Camera cam = FrameCamera(*b, 30.f, 22.f, 1.35f);   // Q's camera, exactly
    bool ok = Shot(*c.r, c.out, "X_closeup_classic.ppm", *b, cam);
    c.r->SetPath(SplatRenderer::Path::March);
    std::printf("  [X] closeup_classic    <- Q's camera/body on the CLASSIC splat\n");
    return ok;
}

// R) orbit
static bool scene_orbit(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    Camera cam = FrameCamera(*b, 200.f, -12.f, 3.0f);
    bool ok = Shot(*c.r, c.out, "R_orbit.ppm", *b, cam);
    std::printf("  [R] orbit              <- low, wide angle (back side of the torus)\n");
    return ok;
}

// S) a few splits (new patch edges)
static bool scene_split_edges(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    const Vec3 mn = b->BBoxMin(), sp = b->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    Vec3 p{ mn.x + sp.x*0.5f, mn.y + sp.y*0.5f, mn.z + sp.z };
    b->RefineWhere(p, 0.3f*diag, 0.2f*diag, 8);
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "S_split_edges.ppm", *b, cam);
    std::printf("  [S] split_edges        <- a few splits (new seams at child edges)\n");
    return ok;
}

// T) harmonic weld
static bool scene_weld(Ctx& c) {
    auto b = Load(c.torus);
    if (!b) return false;
    // perturb a ring of anchors to open seams, then weld them back.
    // Grow the bbox first so the displaced anchors don't clamp.
    b->ExpandBBox(0.1f);
    auto& pool = b->Anchors();
    const Vec3 mn = b->BBoxMin(), sp = b->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    for (uint16_t a = 0; a < (uint16_t)b->AnchorBlob().size(); ++a) {
        Vec3 p = pool.Position(a);
        // push anchors whose x is near the +X edge outward
        if (p.x > mn.x + sp.x*0.85f)
            pool.SetPosition(a, {p.x + 0.01f*diag, p.y, p.z});
    }
    b->MarkAnchorsDirty();
    b->Seams().Build(*b);
    const float gap = b->Seams().HarmonicWeld(*b, 24, 0.2f);
    Camera cam = FrameCamera(*b, 30.f, 22.f, 2.6f);
    bool ok = Shot(*c.r, c.out, "T_weld.ppm", *b, cam);
    std::printf("  [T] weld               <- perturbed +X anchors, HarmonicWeld (gap %.4f%% diag)\n",
                gap / diag * 100.f);
    return ok;
}

// --- main -------------------------------------------------------------------
struct SceneDef {
    char letter;
    const char* name;
    bool (*fn)(Ctx&);
};

static const SceneDef kScenes[] = {
    {'A', "march_default", scene_march_default},
    {'B', "raymarch",      scene_raymarch},
    {'C', "splat_classic", scene_splat_classic},
    {'U', "splat_layered", scene_splat_layered},
    {'V', "splat_area",    scene_splat_area},
    {'D', "march_highres", scene_march_highres},
    {'E', "amr_refine",    scene_amr_refine},
    {'F', "amr_lod",       scene_amr_lod},
    {'G', "dent",          scene_dent},
    {'H', "dent_refit",    scene_dent_refit},
    {'I', "mode0",         [](Ctx& c){ return scene_mode(c, 0, 'I', "mode0"); }},
    {'J', "mode1",         [](Ctx& c){ return scene_mode(c, 1, 'J', "mode1"); }},
    {'K', "modal_ring",    scene_modal_ring},
    {'L', "fracture",      scene_fracture},
    {'M', "morph_color",   scene_morph_color},
    {'N', "sphere",        [](Ctx& c){ return scene_asset(c, 'N', "assets/sphere.psf", "sphere"); }},
    {'O', "asteroid",      [](Ctx& c){ return scene_asset(c, 'O', "assets/asteroid.psf", "asteroid"); }},
    {'P', "roughball",     [](Ctx& c){ return scene_asset(c, 'P', "assets/roughball.psf", "roughball"); }},
    {'Q', "closeup",       scene_closeup},
    {'W', "closeup_layered", scene_closeup_layered},
    {'X', "closeup_classic", scene_closeup_classic},
    {'Y', "cube",          [](Ctx& c){ return scene_asset(c, 'Y', "assets/cube.psf", "cube"); }},
    {'R', "orbit",         scene_orbit},
    {'S', "split_edges",   scene_split_edges},
    {'T', "weld",          scene_weld},
};

int main(int argc, char** argv) {
    std::string outdir = "visual_gallery";
    int w = 640, h = 480;
    std::string only;
    bool list_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) outdir = argv[++i];
        else if (a == "--size" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%dx%d", &w, &h) != 2) {
                std::fprintf(stderr, "bad --size\n"); return 2;
            }
        }
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--list") list_only = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    std::printf("== DeliRium visual gallery ==\n");
    std::printf("output dir: %s/   size: %dx%d\n", outdir.c_str(), w, h);
    std::printf("scenes:\n");
    for (const SceneDef& s : kScenes)
        std::printf("  %c  %s\n", s.letter, s.name);

    if (list_only) return 0;

    // which scenes to run
    auto wanted = [&](char letter) -> bool {
        if (only.empty()) return true;
        return only.find(letter) != std::string::npos;
    };

    SplatRenderer renderer;
    if (!renderer.Initialize(w, h, nullptr, "kernels")) {
        std::fprintf(stderr, "SplatRenderer::Initialize failed (OpenCL / kernel dir?)\n");
        std::fprintf(stderr, "Run from the PROJECT ROOT (the folder containing 'kernels/').\n");
        return 1;
    }
    mkdir_p(outdir);

    Ctx ctx;
    ctx.r = &renderer;
    ctx.out = outdir;
    ctx.w = w; ctx.h = h;

    int ok = 0, fail = 0, skip = 0;
    for (const SceneDef& s : kScenes) {
        if (!wanted(s.letter)) { ++skip; continue; }
        std::printf("\n-- scene %c (%s) --\n", s.letter, s.name);
        if (s.fn(ctx)) ++ok; else ++fail;
    }

    std::printf("\n== gallery complete: %d ok, %d failed, %d skipped ==\n", ok, fail, skip);
    std::printf("PPMs are in %s/ -- open them in any image viewer.\n", outdir.c_str());
    renderer.Shutdown();
    return fail ? 1 : 0;
}