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
    // Who owns the camera. FPS = the engine's built-in WASD + mouse-look
    // (every existing harness). External = the camera belongs to an editor
    // that wants a cursor (Godot-like workspace: orbit with the mouse over
    // the viewport, click panels, type in fields). In External mode the
    // engine never touches yaw/pitch/position and never grabs the pointer.
    enum class CameraControl { FPS, External };

    static Engine& Instance();
    bool Initialize(int w, int h, const std::string& title);  // TODO SDL + CL
    void Run();                                               // TODO loop
    void Shutdown();                                          // TODO

    Scene* CreateScene(const std::string& name);              // TODO
    void   SetReconstructor(std::unique_ptr<IReconstructor>); // TODO hot-swap fixed <-> neural
    IReconstructor& Reconstructor();

    void SetCameraControl(CameraControl m) {
        camera_control = m;
        // Re-apply immediately: the pointer grab was set during Initialize,
        // and an editor that switches to External AFTER init must release
        // it -- otherwise the cursor stays captured and invisible.
        ApplyCameraControl();
    }
    CameraControl CameraControlMode() const { return camera_control; }

    // Release any pointer grab and show the cursor.
    //
    // WHY: the pointer is grabbed (and hidden) in RELATIVE mouse mode. A
    // docked editor needs a VISIBLE cursor it can move over panels, and a
    // grabbed pointer is also a CAPTURED pointer -- SDL keeps sending
    // motion events relative to a virtual centre, so dragging "outward"
    // runs the cursor into the window edge and it stops tracking. The user
    // reported exactly this: the mouse vanished, and dragging outward
    // started clicking sliders.
    //
    // The grab is set at Initialize() time, but the workspace only decides
    // it wants an editor AFTER Initialize() returns -- so the mode must be
    // re-applied whenever it changes, not only at init.
    //
    // (The SDL calls live in the .cpp: this header stays SDL-free.)
    void ApplyCameraControl();

    // --- quitting ---------------------------------------------------------
    // Ask the loop to stop. Safe to call from any thread (the flag is
    // polled once per frame) and from inside a hook -- a workspace that
    // wants to close itself after a scripted interaction uses this.
    void RequestQuit() { running = false; }
    bool IsRunning() const { return running; }
    // Auto-close after N seconds of wall-clock run time. 0 (the default)
    // disables it. This exists so an automated check can launch the app,
    // let it settle, and have it close ITSELF instead of needing an
    // external kill -- which is what leaves a half-written frame behind.
    void SetAutoQuitSeconds(double s) { auto_quit_seconds_ = s; }
    double AutoQuitSeconds() const { return auto_quit_seconds_; }

    // Auto-close after N RENDERED FRAMES (0 = disabled).
    //
    // The DETERMINISTIC counterpart to SetAutoQuitSeconds: the render is
    // not frame-stable, so a clock-defined frame lands on a different sample
    // each run and a pixel measurement taken there is not reproducible. A
    // frame count is. See docs/FIXES_REGISTER.md entry 36 for the
    // measurements and the second cause (a shared GPU needs to drain
    // between launches).
    void SetAutoQuitFrames(int n) { auto_quit_frames_ = n; }
    int  AutoQuitFrames() const { return auto_quit_frames_; }

    // Cap the frame rate (0 = uncapped, default 60).
    //
    // WHY an editor needs this: while a LOCAL model is generating, the
    // editor is usually the heaviest thing on the machine -- it renders
    // every frame at full resolution. The model then gets whatever CPU is
    // left, and a 7B model on a shared box can stretch a 2-minute reply
    // past the point where the user thinks the button is broken. Dropping
    // to a low rate while the model works gives the CPU back and makes the
    // editor feel like it is waiting on purpose rather than hanging.
    void SetMaxFps(double fps) { max_fps_ = fps; }
    double MaxFps() const { return max_fps_; }

    // Capture the WINDOW (viewport + the editor panels drawn over it) to a
    // PPM when the loop ends. Empty = disabled.
    //
    // WHY: the GPU framebuffer readback (GrabFinalRGBA) only sees the 3D
    // image -- it cannot prove the UI painted. This reads the composited
    // window surface, so a screenshot is evidence that the panels are
    // really on screen and not just "the code ran".
    void SetScreenshotOnExit(const std::string& path) { shot_on_exit_ = path; }
    const std::string& ScreenshotOnExit() const { return shot_on_exit_; }
    // Capture the composited window to a PPM. Returns false if the read
    // failed (no SDL, no window, or a readback error).
    bool SaveWindowPPM(const std::string& path) const;

    // Window size in pixels (the SDL window, NOT the render viewport).
    int WindowWidth()  const { return window_w; }
    int WindowHeight() const { return window_h; }
    // Live frame timing, published by the render loop so a docked UI (the
    // workspace status bar) can show the real rate instead of a stub 0.
    int   Fps() const { return fps_display; }
    float LastFrameMs() const { return frame_ms_display; }

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
    // Runs once per frame AFTER the SDL event queue has been drained but
    // BEFORE the scene updates and renders. A docked workspace uses it to
    // lay out its panels, resize the viewport, and run the UI UPDATE pass
    // (hit-testing + callbacks); the PAINT pass then happens in the
    // overlay hook, which the renderer calls during Present().
    using FrameHook = std::function<void()>;
    void SetInputHook(InputHook h)  { input_hook = std::move(h); }
    void SetHudHook(HudHook h)      { hud_hook = std::move(h); }
    void SetRenderHook(RenderHook h){ render_hook = std::move(h); }
    void SetOverlayHook(OverlayHook h) { overlay_hook = std::move(h); }
    void SetFrameHook(FrameHook h)  { frame_hook = std::move(h); }
    void DrawOverlay(void* sdl_renderer, int width, int height) {
        if (overlay_hook) overlay_hook(sdl_renderer, width, height);
    }
    // The SDL_Renderer* the engine created (as void* to keep SDL out of this
    // header). The workspace needs it to paint its panels.
    void* SDLRenderer() const { return sdl_renderer; }
private:
    Engine() = default;
    std::unique_ptr<Scene>          currentScene;
    std::unique_ptr<SplatRenderer>  renderer;
    std::unique_ptr<PhysicsWorld>   physics;
    std::unique_ptr<IReconstructor> reconstructor;
    bool  running = false;
    float fixedTimeStep = 1.0f / 120.0f;
    CameraControl camera_control = CameraControl::FPS;
    int   window_w = 0, window_h = 0;
    // Wall-clock seconds after which Run() stops itself (0 = never).
    double auto_quit_seconds_ = 0.0;
    // Rendered-frame count after which Run() stops itself (0 = never).
    // The DETERMINISTIC counterpart to auto_quit_seconds_: a clock lands on
    // an arbitrary frame, and the render is not frame-stable.
    int   auto_quit_frames_ = 0;
    // Frame-rate cap. 0 = uncapped. 60 = the default interactive rate.
    double max_fps_ = 60.0;
    // If set, the window is captured to this PPM when Run() ends.
    std::string shot_on_exit_;
    // Published by the render loop; read by the workspace status bar.
    int   fps_display = 0;
    float frame_ms_display = 0.0f;
    InputHook input_hook;
    HudHook   hud_hook;
    RenderHook render_hook;
    OverlayHook overlay_hook;
    FrameHook frame_hook;
    void* sdl_renderer = nullptr;
};

} // namespace DeliRium
