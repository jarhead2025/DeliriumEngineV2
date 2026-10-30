// FILEPATH: engine/src/core/Engine.cpp
// Product path: M1 load preserved; optional SDL (DELIRIUM_HAS_SDL); OpenCL via SplatRenderer.
#include "delirium/core/Engine.h"
#include "delirium/core/Scene.h"
#include "delirium/psf/Reconstructor.h"
#include "delirium/render/SplatRenderer.h"
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
    std::printf("Engine: headless (no DELIRIUM_HAS_SDL) -- install libsdl2-dev for window\n");
    renderer = std::make_unique<SplatRenderer>();
    if (!renderer->Initialize(w, h, nullptr, "kernels")) {
        std::fprintf(stderr, "SplatRenderer::Initialize failed (OpenCL optional headless)\n");
        renderer.reset();
    } else {
        std::printf("Engine: SplatRenderer OpenCL headless OK (%s)\n",
                    renderer->DeviceName().c_str());
    }
#endif
    return true;
}

Scene* Engine::CreateScene(const std::string& name) {
    currentScene = std::make_unique<Scene>(name.empty() ? std::string("scene") : name);
    std::printf("Engine::CreateScene(\"%s\")\n", currentScene->sceneName.c_str());
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
    if (!running) {
        std::printf("Engine::Run skipped (not initialized)\n");
        return;
    }
    Scene* scene = currentScene.get();
    if (!scene) {
        std::printf("Engine::Run: no scene\n");
        return;
    }
    const auto& objs = scene->Objects();
    std::printf("Engine::Run: %zu object(s)\n", objs.size());
    for (size_t i = 0; i < objs.size(); ++i) {
        if (!objs[i]) continue;
        std::printf("  [%zu] \"%s\" HasBody=%s\n", i, objs[i]->name.c_str(),
                    objs[i]->HasBody() ? "yes" : "no");
    }

    const int frames = 60;
    const float dt = 1.0f / 60.0f;
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    bool interactive = true;
#else
    bool interactive = false;
#endif
    for (int f = 0; f < frames && running; ++f) {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            if (input_hook && input_hook(&ev)) continue;
        }
#endif
        scene->FixedUpdate(fixedTimeStep);
        scene->Update(dt);
        if (renderer) {
            Camera& cam = scene->GetCamera();
            for (const auto& o : objs) {
                if (!o || !o->HasBody()) continue;
                renderer->RenderBody(o->Body(), cam);
            }
            renderer->Present();
        }
    }
    std::printf("Engine::Run: completed up to %d frames (interactive=%s renderer=%s)\n",
                frames, interactive ? "yes" : "no", renderer ? "yes" : "no");
    std::printf("M1 load path preserved -- E2/E3: SDL optional, OpenCL via SplatRenderer\n");
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
