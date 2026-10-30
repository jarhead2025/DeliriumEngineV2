// =============================================================================
// Engine.h - Singleton loop (aligned with Delirium v0.1: SDL2 window,
// fixed-step physics). v0.2 swaps the GL mesh path for SplatRenderer and
// owns the active IReconstructor.
// =============================================================================
#pragma once
#include <memory>
#include <string>
#include <functional>
#include "Scene.h"
#include "../render/SplatRenderer.h"
#include "../physics/PhysicsWorld.h"
#include "../psf/Reconstructor.h"

namespace DeliRium {

class Engine {
public:
    static Engine& Instance();
    bool Initialize(int w, int h, const std::string& title);  // TODO SDL + CL
    void Run();                                               // TODO loop
    void Shutdown();                                          // TODO

    Scene* CreateScene(const std::string& name);              // TODO
    void   SetReconstructor(std::unique_ptr<IReconstructor>); // TODO hot-swap fixed <-> neural
    IReconstructor& Reconstructor();

    SplatRenderer* GetRenderer()     { return renderer.get(); }
    PhysicsWorld*  GetPhysicsWorld() { return physics.get(); }
    Scene*         CurrentScene()    { return currentScene.get(); }

    // Editor hooks. The engine owns the window/loop/render; an editor
    // installs callbacks to handle input and contribute a HUD string,
    // keeping all tool logic outside the engine. The input hook receives
    // the raw SDL_Event (as void* to avoid leaking SDL into this header)
    // and returns true if it consumed the event. The HUD hook returns a
    // status string appended to the window title each second.
    using InputHook = std::function<bool(const void* sdl_event)>;
    using HudHook   = std::function<std::string()>;
    // Runs after the scene's objects are rendered (before Present). Lets a
    // tool composite extra bodies into the same frame (e.g. fracture
    // fragments) without touching the engine loop.
    using RenderHook = std::function<void(SplatRenderer&, const Camera&)>;
    // Optional SDL overlay between texture copy and the single Present call.
    // Uses the existing renderer; does not read back or clone the PSF image.
    using OverlayHook = std::function<void(void* sdl_renderer, int width, int height)>;
    void SetInputHook(InputHook h)  { input_hook = std::move(h); }
    void SetHudHook(HudHook h)      { hud_hook = std::move(h); }
    void SetRenderHook(RenderHook h){ render_hook = std::move(h); }
    void SetOverlayHook(OverlayHook h) { overlay_hook = std::move(h); }
    void DrawOverlay(void* sdl_renderer, int width, int height) {
        if (overlay_hook) overlay_hook(sdl_renderer, width, height);
    }
private:
    Engine() = default;
    std::unique_ptr<Scene>          currentScene;
    std::unique_ptr<SplatRenderer>  renderer;
    std::unique_ptr<PhysicsWorld>   physics;
    std::unique_ptr<IReconstructor> reconstructor;
    bool  running = false;
    float fixedTimeStep = 1.0f / 120.0f;
    InputHook input_hook;
    HudHook   hud_hook;
    RenderHook render_hook;
    OverlayHook overlay_hook;
};

} // namespace DeliRium
