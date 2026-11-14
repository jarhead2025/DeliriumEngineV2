// FILEPATH: engine/src/core/Engine.cpp
// Interactive camera: WASD move, mouse look; planes/splat use Camera XYZ each frame.
#include "delirium/core/Engine.h"
#include "delirium/core/Scene.h"
#include "delirium/psf/Reconstructor.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

namespace DeliRium {

Engine& Engine::Instance() {
    static Engine inst;
    return inst;
}

bool Engine::Initialize(int w, int h, const std::string& title) {
    std::printf("Engine::Initialize %dx%d \"%s\"\n", w, h, title.c_str());
    running = true;
    window_w = w; window_h = h;

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init failed: %s (continuing headless)\n", SDL_GetError());
    } else {
        // CLAMP THE WINDOW TO THE DISPLAY.
        //
        // WHY: a window larger than the screen is not merely "too big" -- the
        // parts that fall off-screen are UNREACHABLE. A docked editor whose
        // AI console lives in the bottom strip then has its console on a
        // screen that does not exist: the user sees the top panels, the
        // console is simply gone, and the app looks broken. SDL happily
        // creates the oversized window; the damage is invisible until the
        // user tries to type a prompt.
        //
        // The window is still RESIZABLE, so clamping the INITIAL size is
        // enough: the user can grow it later if the display allows.
        int disp_w = w, disp_h = h;
        {
            SDL_DisplayMode dm;
            if (SDL_GetCurrentDisplayMode(0, &dm) == 0) {
                disp_w = dm.w;
                disp_h = dm.h;
                // Leave a little room so the window decorations (title bar)
                // do not push the bottom edge off the screen.
                const int cap_w = std::max(640, disp_w - 8);
                const int cap_h = std::max(480, disp_h - 48);
                if (w > cap_w || h > cap_h) {
                    std::printf("Engine: clamping window %dx%d -> %dx%d (display is %dx%d)\n",
                                w, h, std::min(w, cap_w), std::min(h, cap_h), disp_w, disp_h);
                    w = std::min(w, cap_w);
                    h = std::min(h, cap_h);
                    window_w = w;
                    window_h = h;
                }
            }
        }
        SDL_Window* win = SDL_CreateWindow(
            title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, 0);
        SDL_Renderer* rnd = win ? SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED) : nullptr;
        if (!win || !rnd) {
            std::fprintf(stderr, "SDL window/renderer failed: %s\n", SDL_GetError());
        } else {
            sdl_renderer = rnd;
            std::printf("Engine: SDL window %dx%d ready\n", w, h);
            std::printf("Controls: WASD move, Space/Ctrl up/down, mouse look, Esc quit\n");
            // In External mode the pointer stays visible: the editor needs to
            // click panels and type into fields, and grabs the pointer itself
            // only when it wants FPS-style look.
            if (camera_control == CameraControl::FPS)
                SDL_SetRelativeMouseMode(SDL_TRUE);
            renderer = std::make_unique<SplatRenderer>();
            if (!renderer->Initialize(w, h, static_cast<void*>(rnd), "kernels")) {
                std::fprintf(stderr, "SplatRenderer::Initialize failed (OpenCL?)\n");
                renderer.reset();
            } else {
                std::printf("Engine: SplatRenderer OpenCL path ready (%s)\n",
                            renderer->DeviceName().c_str());
            }
        }
    }
#else
    std::printf("Engine: headless (no DELIRIUM_HAS_SDL)\n");
    renderer = std::make_unique<SplatRenderer>();
    if (!renderer->Initialize(w, h, nullptr, "kernels")) {
        renderer.reset();
    }
#endif
    return true;
}

Scene* Engine::CreateScene(const std::string& name) {
    currentScene = std::make_unique<Scene>(name.empty() ? std::string("scene") : name);
    Camera& cam = currentScene->GetCamera();
    cam.position = {0.f, 0.8f, 3.5f};  // farther default; was too close/twitchy
    cam.target = {0.f, 0.f, 0.f};
    cam.up = {0.f, 1.f, 0.f};
    cam.fov_deg = 45.f;
    cam.near_p = 0.05f;
    cam.far_p = 50.f;
    std::printf("Engine::CreateScene(\"%s\") cam=(%.2f,%.2f,%.2f)\n",
                currentScene->sceneName.c_str(), cam.position.x, cam.position.y, cam.position.z);
    return currentScene.get();
}

void Engine::ApplyCameraControl() {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (camera_control == CameraControl::External) {
        // Release the grab AND show the cursor. A relative-mode grab hides
        // the pointer and captures it, so the cursor cannot leave the
        // window and every motion event is relative to a virtual centre.
        if (SDL_GetRelativeMouseMode() == SDL_TRUE) SDL_SetRelativeMouseMode(SDL_FALSE);
        SDL_ShowCursor(SDL_TRUE);
    }
#endif
}

IReconstructor& Engine::Reconstructor() {
    if (!reconstructor)
        reconstructor = std::make_unique<FixedReconstructor>();
    return *reconstructor;
}

// Capture the composited WINDOW to a PPM.
//
// WHY SDL_RenderReadPixels and not the GPU readback: the workspace paints
// its panels with SDL drawing calls INTO the same SDL_Renderer that
// presents the viewport texture. Reading the GPU framebuffer would show
// only the 3D image and miss every panel, so a screenshot taken that way
// could not prove the UI is on screen. Reading the SDL target after the
// overlay pass captures the final composited window.
//
// The read happens AFTER the last Present(), so the target holds the last
// painted frame.
bool Engine::SaveWindowPPM(const std::string& path) const {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (!sdl_renderer || window_w <= 0 || window_h <= 0) return false;
    SDL_Renderer* rnd = static_cast<SDL_Renderer*>(sdl_renderer);
    // SDL_RenderReadPixels needs the render target set. The renderer draws
    // to the default target, so ask for it explicitly.
    // (In SDL2 the render target IS a texture: SDL_RenderTarget is an alias
    // for SDL_Texture*, so there is no separate type to name here.)
    SDL_Texture* prev = SDL_GetRenderTarget(rnd);
    SDL_SetRenderTarget(rnd, nullptr);
    SDL_RenderFlush(rnd);
    std::vector<unsigned char> px(size_t(window_w) * size_t(window_h) * 4, 0);
    if (SDL_RenderReadPixels(rnd, nullptr, SDL_PIXELFORMAT_ABGR8888,
                             px.data(), window_w * 4) != 0) {
        std::fprintf(stderr, "SaveWindowPPM: SDL_RenderReadPixels failed: %s\n",
                     SDL_GetError());
        if (prev) SDL_SetRenderTarget(rnd, prev);
        return false;
    }
    if (prev) SDL_SetRenderTarget(rnd, prev);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "SaveWindowPPM: cannot open %s\n", path.c_str()); return false; }
    std::fprintf(f, "P6\n%d %d\n255\n", window_w, window_h);
    for (int y = 0; y < window_h; ++y) {
        for (int x = 0; x < window_w; ++x) {
            const unsigned char* p = &px[(size_t(y) * window_w + x) * 4];
            // ABGR8888 in memory is B,G,R,A.
            std::fputc(p[2], f); std::fputc(p[1], f); std::fputc(p[0], f);
        }
    }
    std::fclose(f);
    std::printf("Engine: saved window capture %s (%dx%d)\n",
                path.c_str(), window_w, window_h);
    return true;
#else
    (void)path;
    return false;
#endif
}

void Engine::SetReconstructor(std::unique_ptr<IReconstructor> r) {
    reconstructor = std::move(r);
}

void Engine::Run() {
    if (!running || !currentScene) return;
    Scene* scene = currentScene.get();
    const auto& objs = scene->Objects();
    std::printf("Engine::Run: %zu object(s)\n", objs.size());
    for (size_t i = 0; i < objs.size(); ++i) {
        if (!objs[i]) continue;
        std::printf("  [%zu] \"%s\" HasBody=%s\n", i, objs[i]->name.c_str(),
                    objs[i]->HasBody() ? "yes" : "no");
    }

    const float dt = 1.0f / 60.0f;
    Camera& cam = scene->GetCamera();

    float yaw = std::atan2(cam.target.x - cam.position.x, cam.target.z - cam.position.z);
    float pitch = 0.f;
    {
        Vec3 f{cam.target.x - cam.position.x, cam.target.y - cam.position.y,
               cam.target.z - cam.position.z};
        float fl = std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
        if (fl > 1e-6f) pitch = std::asin(std::max(-1.f, std::min(1.f, f.y / fl)));
    }
    const float move_speed = 1.2f;
    const float mouse_sens = 0.00045f;

    auto apply_look = [&]() {
        // In External mode the editor owns the camera completely: the engine
        // must not recompute cam.target from its own yaw/pitch, or it would
        // fight the editor's orbit every frame.
        if (camera_control != CameraControl::FPS) return;
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const float cy = std::cos(yaw), sy = std::sin(yaw);
        Vec3 forward{sy * cp, sp, cy * cp};
        cam.target = {cam.position.x + forward.x, cam.position.y + forward.y,
                      cam.position.z + forward.z};
        cam.up = {0.f, 1.f, 0.f};
    };
    apply_look();

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    std::printf("Engine::Run: INTERACTIVE (WASD + mouse, Esc quit)\n");
    int f = 0;
    auto fps_t0 = std::chrono::steady_clock::now();
    auto run_t0 = fps_t0;
    int fps_frames = 0;
    while (running) {
        auto frame_t0 = std::chrono::steady_clock::now();

        // AUTO-QUIT: stop the loop after the requested wall-clock time.
        // Polled here (before the frame's work) so the app always ends on a
        // COMPLETE frame -- a frame that was rendered and presented -- never
        // on a half-drawn one.
        if (auto_quit_seconds_ > 0.0) {
            const double elapsed_run =
                std::chrono::duration<double>(frame_t0 - run_t0).count();
            if (elapsed_run >= auto_quit_seconds_) {
                std::printf("Engine::Run: auto-quit after %.2f s (%d frames)\n",
                            elapsed_run, f);
                running = false;
                break;
            }
        }

        // AUTO-QUIT BY FRAME COUNT: the DETERMINISTIC counterpart to the
        // wall-clock quit above. A clock lands on whichever frame the clock
        // reaches, and the render is not frame-stable, so a pixel
        // measurement taken at a clock-defined frame is not reproducible.
        if (auto_quit_frames_ > 0 && f >= auto_quit_frames_) {
            std::printf("Engine::Run: auto-quit after %d frames\n", f);
            running = false;
            break;
        }

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            // The editor gets first refusal: UI focus must not rotate/zoom
            // the viewport or trigger Escape while editing a property.
            if (input_hook && input_hook(&ev)) continue;
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) running = false;
            // In External mode the camera belongs to the editor. The engine
            // must not rotate it (that would fight the editor's orbit) and
            // must not move it with WASD.
            if (camera_control != CameraControl::FPS) continue;
            if (ev.type == SDL_MOUSEMOTION) {
                yaw += ev.motion.xrel * mouse_sens;
                pitch -= ev.motion.yrel * mouse_sens;
                const float lim = 1.4f;
                if (pitch > lim) pitch = lim;
                if (pitch < -lim) pitch = -lim;
                apply_look();
            }
            if (ev.type == SDL_MOUSEWHEEL) {
                // Zoom along look direction (scroll up = closer)
                const float cp = std::cos(pitch), sp = std::sin(pitch);
                const float cy = std::cos(yaw), sy = std::sin(yaw);
                Vec3 forward{sy * cp, sp, cy * cp};
                const float zoom = float(ev.wheel.y) * 0.35f;
                cam.position.x += forward.x * zoom;
                cam.position.y += forward.y * zoom;
                cam.position.z += forward.z * zoom;
                apply_look();
            }
        }

        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const float cy = std::cos(yaw), sy = std::sin(yaw);
        Vec3 forward{sy * cp, sp, cy * cp};
        Vec3 right{cy, 0.f, -sy};
        float rl = std::sqrt(right.x * right.x + right.z * right.z);
        if (rl > 1e-6f) {
            right.x /= rl;
            right.z /= rl;
        }
        float sp_move = move_speed * dt;
        if (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]) sp_move *= 1.8f;

        auto move = [&](const Vec3& d, float s) {
            cam.position.x += d.x * s;
            cam.position.y += d.y * s;
            cam.position.z += d.z * s;
        };
        // Pointer-mode editor owns keyboard focus. Do not move the camera
        // from polled key state when its event handler consumes the keys.
        // In External mode the editor owns the camera outright -- WASD must
        // not move it behind the editor's back.
        if (SDL_GetRelativeMouseMode() == SDL_TRUE &&
            camera_control == CameraControl::FPS) {
            if (keys[SDL_SCANCODE_W]) move(forward, sp_move);
            if (keys[SDL_SCANCODE_S]) move(forward, -sp_move);
            if (keys[SDL_SCANCODE_D]) move(right, sp_move);
            if (keys[SDL_SCANCODE_A]) move(right, -sp_move);
            if (keys[SDL_SCANCODE_SPACE]) cam.position.y += sp_move;
            if (keys[SDL_SCANCODE_LCTRL] || keys[SDL_SCANCODE_RCTRL]) cam.position.y -= sp_move;
        }
        apply_look();

        // The workspace's frame hook runs here: it lays out the docked
        // panels, resizes the viewport, and runs the UI UPDATE pass
        // (hit-testing + widget callbacks) BEFORE the scene updates. The
        // PAINT pass happens in the overlay hook during Present().
        if (frame_hook) frame_hook();

        scene->FixedUpdate(fixedTimeStep);
        scene->Update(dt);
        if (renderer) {
            // MULTI-BODY COMPOSITING: body 0 clears the shared framebuffer
            // and every body renders into it (frontmost-wins atom_min), and
            // only the LAST body runs the full-screen resolve+infill
            // (final=true) so the post passes see the COMBINED image.
            //
            // The old code called RenderBody(body, cam) with the defaults
            // (clear=true, final=true) for EVERY body, so each object wiped
            // the previous object's pixels -- with two objects in the scene
            // only the last one was visible.
            //
            // The render list is built FIRST so the clear/final flags are
            // correct no matter how many bodies there are (and so a
            // fracture fragment -- a SEPARATE PSFBody on the same object --
            // composites into the same frame instead of being drawn after
            // the resolve, where it would be discarded).
            struct DrawItem { PSFBody* body; Mat4 model; float depth_bias; };
            std::vector<DrawItem> draws;
            for (const auto& o : objs) {
                if (!o || !o->HasBody()) continue;
                // The object's rigid placement lives OUTSIDE the frozen
                // 64-byte records (core/Transform.h) and is applied as a GPU
                // model matrix, so moving an object never re-quantizes an
                // anchor. Identity for every pre-existing harness.
                //
                // The matrix is stored BY VALUE: a pointer into a
                // std::vector would dangle as soon as the vector grew.
                const Mat4 m = o->GetTransform().ToMatrix();
                draws.push_back(DrawItem{ &o->Body(), m, 0.0f });
                // A fracture fragment is a separate body owned by the object.
                // Push it just behind the parent with a tiny NDC depth bias
                // so the cut rims (coplanar) do not flip the depth winner
                // frame to frame.
                if (PSFBody* frag = o->Fragment())
                    draws.push_back(DrawItem{ frag, m, 1e-4f });
            }
            for (size_t i = 0; i < draws.size(); ++i) {
                renderer->RenderBody(*draws[i].body, cam,
                                     /*clear=*/(i == 0),
                                     /*final=*/(i + 1 == draws.size()),
                                     /*depth_bias=*/draws[i].depth_bias,
                                     draws[i].model.m);
            }
            if (draws.empty()) {
                // No bodies: still resolve the (empty) framebuffer so the
                // viewport shows the background instead of the PREVIOUS
                // frame's stale image.
                renderer->RenderEmptyFrame();
            }
            if (render_hook) render_hook(*renderer, cam);
            renderer->Present();
            // DRAIN queued window captures HERE, after Present(): the frame
            // hook runs BEFORE the render, so a capture taken there would
            // read the previous frame. Draining after Present guarantees
            // capture == the pixels the just-rendered frame produced.
            if (!capture_requests_.empty()) {
                std::vector<std::string> reqs;
                reqs.swap(capture_requests_);
                for (const auto& p : reqs) {
                    if (!SaveWindowPPM(p))
                        std::fprintf(stderr, "Engine: window capture failed (%s)\n",
                                     p.c_str());
                }
            }
        }
        ++f;
        // Cap before sampling FPS so the printed number measures complete
        // present-to-present wall time, including the throttle (rather than
        // comparing an uncapped frame-ms number with a capped FPS number).
        {
            auto now = std::chrono::steady_clock::now();
            double dt_frame = std::chrono::duration<double>(now - frame_t0).count();
            // The cap is a floor on the frame PERIOD. 0 = uncapped.
            const double period = (max_fps_ > 0.0) ? (1.0 / max_fps_) : 0.0;
            if (period > 0.0 && dt_frame < period) {
                Uint32 ms = (Uint32)((period - dt_frame) * 1000.0);
                if (ms > 0 && ms < 250) SDL_Delay(ms);
            }
        }
        ++fps_frames;
        {
            auto now = std::chrono::steady_clock::now();
            const double dt_frame = std::chrono::duration<double>(now - frame_t0).count();
            const double elapsed = std::chrono::duration<double>(now - fps_t0).count();
            if (elapsed >= 0.5) {
                const float fps_inst = float(fps_frames) / float(elapsed);
                // Publish the timing so a docked UI (the workspace status
                // bar) can show the REAL rate -- otherwise it prints a
                // permanent "0 FPS", which reads as "the engine is broken".
                fps_display = int(fps_inst + 0.5f);
                frame_ms_display = float(dt_frame * 1000.0);
                std::printf("FPS: %.1f (last full frame %.1f ms)  cam=(%.2f,%.2f,%.2f)\n",
                            fps_inst, dt_frame * 1000.0,
                            cam.position.x, cam.position.y, cam.position.z);
                fps_t0 = now;
                fps_frames = 0;
            }
        }
    }
    std::printf("Engine::Run: exited after %d frames\n", f);
    // Capture the composited window AFTER the last Present(), so the
    // screenshot shows the final painted frame (viewport + panels).
    if (!shot_on_exit_.empty()) {
        if (!SaveWindowPPM(shot_on_exit_))
            std::fprintf(stderr, "Engine: window capture failed (%s)\n",
                         shot_on_exit_.c_str());
    }
    SDL_SetRelativeMouseMode(SDL_FALSE);
#else
    for (int fi = 0; fi < 60 && running; ++fi) {
        scene->FixedUpdate(fixedTimeStep);
        scene->Update(dt);
        if (renderer) {
            for (const auto& o : objs) {
                if (!o || !o->HasBody()) continue;
                renderer->RenderBody(o->Body(), cam);
            }
            renderer->Present();
        }
    }
#endif
}

void Engine::Shutdown() {
    std::printf("Engine::Shutdown\n");
    running = false;
    if (renderer) {
        renderer->Shutdown();
        renderer.reset();
    }
    currentScene.reset();
    physics.reset();
    // Hooks may capture a stack-owned Editor; never retain them across runs.
    input_hook = {};
    hud_hook = {};
    render_hook = {};
    overlay_hook = {};
    frame_hook = {};
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    SDL_Quit();
#endif
}

} // namespace DeliRium
