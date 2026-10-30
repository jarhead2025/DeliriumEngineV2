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
#include <memory>
#include <string>

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

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init failed: %s (continuing headless)\n", SDL_GetError());
    } else {
        SDL_Window* win = SDL_CreateWindow(
            title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, 0);
        SDL_Renderer* rnd = win ? SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED) : nullptr;
        if (!win || !rnd) {
            std::fprintf(stderr, "SDL window/renderer failed: %s\n", SDL_GetError());
        } else {
            std::printf("Engine: SDL window %dx%d ready\n", w, h);
            std::printf("Controls: WASD move, Space/Ctrl up/down, mouse look, Esc quit\n");
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

IReconstructor& Engine::Reconstructor() {
    if (!reconstructor)
        reconstructor = std::make_unique<FixedReconstructor>();
    return *reconstructor;
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
    int fps_frames = 0;
    float fps_smooth = 0.f;
    while (running) {
        auto frame_t0 = std::chrono::steady_clock::now();

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) running = false;
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
            if (input_hook && input_hook(&ev)) continue;
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
        if (keys[SDL_SCANCODE_W]) move(forward, sp_move);
        if (keys[SDL_SCANCODE_S]) move(forward, -sp_move);
        if (keys[SDL_SCANCODE_D]) move(right, sp_move);
        if (keys[SDL_SCANCODE_A]) move(right, -sp_move);
        if (keys[SDL_SCANCODE_SPACE]) cam.position.y += sp_move;
        if (keys[SDL_SCANCODE_LCTRL] || keys[SDL_SCANCODE_RCTRL]) cam.position.y -= sp_move;
        apply_look();

        scene->FixedUpdate(fixedTimeStep);
        scene->Update(dt);
        if (renderer) {
            for (const auto& o : objs) {
                if (!o || !o->HasBody()) continue;
                renderer->RenderBody(o->Body(), cam);
            }
            renderer->Present();
        }
        ++f;
        ++fps_frames;
        {
            auto now = std::chrono::steady_clock::now();
            double dt_frame = std::chrono::duration<double>(now - frame_t0).count();
            double elapsed = std::chrono::duration<double>(now - fps_t0).count();
            if (elapsed >= 0.5) {
                float fps_inst = float(fps_frames) / float(elapsed);
                fps_smooth = (fps_smooth > 1.f) ? (fps_smooth * 0.7f + fps_inst * 0.3f) : fps_inst;
                std::printf("FPS: %.1f (frame %.1f ms)  cam=(%.2f,%.2f,%.2f)\n",
                            fps_smooth, dt_frame * 1000.0,
                            cam.position.x, cam.position.y, cam.position.z);
                fps_t0 = now;
                fps_frames = 0;
            }
        }
        // Cap only if we're faster than ~60Hz; don't sleep when GPU path is already slow
        {
            auto now = std::chrono::steady_clock::now();
            double dt_frame = std::chrono::duration<double>(now - frame_t0).count();
            if (dt_frame < 1.0/60.0) {
                Uint32 ms = (Uint32)((1.0/60.0 - dt_frame) * 1000.0);
                if (ms > 0 && ms < 30) SDL_Delay(ms);
            }
        }
    }
    std::printf("Engine::Run: exited after %d frames\n", f);
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
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    SDL_Quit();
#endif
}

} // namespace DeliRium
