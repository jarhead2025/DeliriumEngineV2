// =============================================================================
// 04_spectral_live - Real-time SDL viewer for spectral primitives (.spf).
//
// Renders a spectral primitive on the GPU every frame via SpectralRenderer
// and blits it to an SDL window. Orbit with the mouse, zoom with the wheel
// (or W/S). As you zoom IN, the continuous frequency LOD reveals texture
// detail progressively -- the cow's spots emerge from soft to crisp with no
// popping; as you zoom OUT they melt back. The title bar shows live FPS, the
// camera distance, and the geometry/color LOD orders in use, so you can
// watch the "internal changes in real time, like Blender".
//
//   ./example_spectral_live assets/spot/spot.spf
//
//   mouse drag = orbit    wheel / W,S = zoom    R = reset view    ESC = quit
// =============================================================================
#include "delirium/psf/SpectralPrimitive.h"
#include "delirium/psf/SpectralObject.h"
#include "delirium/render/SpectralRenderer.h"
#include "delirium/render/Camera.h"
#include <SDL2/SDL.h>
#include <cstdio>
#include <cmath>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* spf = argc > 1 ? argv[1] : "assets/spot/spot.spf";
    const int W = 800, H = 600;

    SpectralPrimitive prim;
    if (!prim.Load(spf)) { std::fprintf(stderr, "load failed: %s\n", spf); return 1; }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL init: %s\n", SDL_GetError()); return 1; }
    SDL_Window* win = SDL_CreateWindow("DeliRium Spectral (live)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_SHOWN);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888,
        SDL_TEXTUREACCESS_STREAMING, W, H);

    SpectralRenderer rnd;
    if (!rnd.Initialize(W, H, "kernels")) {
        std::fprintf(stderr, "renderer init failed (OpenCL device?)\n"); return 1; }
    std::printf("GPU: %s\n", rnd.DeviceName().c_str());

    Vec3 c = prim.Center();
    // orbit state
    float yaw = 0.6f, pitch = 0.4f, dist = 3.0f;
    const float base_dist = 3.0f;
    bool dragging = false; int lastx = 0, lasty = 0;

    bool running = true;
    Uint32 last = SDL_GetTicks(); int frames = 0; float fps = 0;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_KEYDOWN) {
                SDL_Keycode k = e.key.keysym.sym;
                if (k == SDLK_ESCAPE) running = false;
                else if (k == SDLK_w) dist = std::fmax(0.4f, dist - 0.15f);
                else if (k == SDLK_s) dist += 0.15f;
                else if (k == SDLK_r) { yaw = 0.6f; pitch = 0.4f; dist = base_dist; }
            }
            else if (e.type == SDL_MOUSEBUTTONDOWN) {
                dragging = true; lastx = e.button.x; lasty = e.button.y; }
            else if (e.type == SDL_MOUSEBUTTONUP) dragging = false;
            else if (e.type == SDL_MOUSEMOTION && dragging) {
                yaw   += (e.motion.x - lastx) * 0.01f;
                pitch += (e.motion.y - lasty) * 0.01f;
                pitch = std::fmax(-1.5f, std::fmin(1.5f, pitch));
                lastx = e.motion.x; lasty = e.motion.y;
            }
            else if (e.type == SDL_MOUSEWHEEL)
                dist = std::fmax(0.4f, dist - e.wheel.y * 0.2f);
        }

        // camera orbits the primitive center at `dist`
        Camera cam;
        cam.position = { c.x + dist*std::cos(pitch)*std::cos(yaw),
                         c.y + dist*std::cos(pitch)*std::sin(yaw),
                         c.z + dist*std::sin(pitch) };
        cam.target = c; cam.up = { 0, 0, 1 };

        // continuous LOD from distance. IMPORTANT: only the COLOR/texture
        // detail varies with distance -- the SHAPE stays fixed. Varying the
        // geometry order changes the silhouette (low SH orders give a
        // genuinely different outline, not just less texture), which looked
        // like the model morphing on zoom. So geometry is held at full order
        // and only the color order tracks distance (the user's real intent:
        // texture sharpens as you approach, shape unchanged).
        float t = std::fmax(0.0f, std::fmin(1.0f, (dist - 0.4f) / (6.0f - 0.4f)));
        float baked_color = (float)prim.ColorOrder();
        float color_order = baked_color * (1.0f - t) + 2.0f * t; // far=2, near=baked
        if (color_order > baked_color) color_order = baked_color;
        uint32_t geom_order = prim.GeomOrder();   // FIXED: shape never morphs

        float cw[3] = { c.x, c.y, c.z };
        rnd.BeginFrame();
        rnd.DrawPrimitive(prim, cw, 1.0f, geom_order, color_order, cam);
        rnd.EndFrame();

        SDL_UpdateTexture(tex, nullptr, rnd.Image().data(), W * 4);
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, nullptr, nullptr);
        SDL_RenderPresent(ren);

        // fps + live state in the title
        if (++frames >= 15) {
            Uint32 now = SDL_GetTicks();
            fps = frames * 1000.0f / (now - last);
            last = now; frames = 0;
            char title[256];
            std::snprintf(title, sizeof title,
                "DeliRium Spectral  |  %.0f fps  |  dist %.2f  |  "
                "geomL %u  colorL %.1f  |  %zu B",
                fps, dist, geom_order, color_order, prim.ByteSize());
            SDL_SetWindowTitle(win, title);
        }
    }

    rnd.Shutdown();
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
