// =============================================================================
// ui_font_probe - Renders the embedded 5x7 font atlas + sample strings to a
// PPM so the glyph table can be inspected by eye.
//
// This exists because SDL2_ttf is unavailable on this machine and the whole
// workspace UI depends on the embedded font being CORRECT. A wrong glyph
// table is invisible in a unit test (it still "renders") but obvious in a
// picture.
//
//   cmake --build build_delirium --target ui_font_probe
//   ./build_delirium/ui_font_probe /tmp/font.ppm
// =============================================================================
#include "delirium/ui/Font.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>

using namespace DeliRium;

static bool write_ppm(const char* path, const std::vector<unsigned char>& rgba, int w, int h) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return false;
    std::fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; ++i) {
        const unsigned char rgb[3] = { rgba[size_t(i)*4+0], rgba[size_t(i)*4+1], rgba[size_t(i)*4+2] };
        std::fwrite(rgb, 1, 3, fp);
    }
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "/tmp/font_probe.ppm";
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* win = SDL_CreateWindow("font", 0, 0, 900, 260, SDL_WINDOW_HIDDEN);
    SDL_Renderer* rnd = win ? SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE) : nullptr;
    if (!win || !rnd) {
        std::fprintf(stderr, "SDL renderer failed: %s\n", SDL_GetError());
        return 1;
    }
    BitmapFont font;
    if (!font.Init(rnd)) {
        std::fprintf(stderr, "BitmapFont::Init failed\n");
        return 1;
    }
    const int W = 900, H = 260;
    SDL_SetRenderDrawColor(rnd, 18, 18, 24, 255);
    SDL_RenderClear(rnd);
    SDL_SetRenderDrawColor(rnd, 229, 234, 242, 255);

    // Row 1: the whole printable range at scale 2.
    std::string all;
    for (int c = 32; c <= 126; ++c) all += char(c);
    font.Draw(rnd, 8, 8, all, 2);
    // Row 2: scale 1 (the density the panels use).
    font.Draw(rnd, 8, 40, all, 1);
    // Row 3: real UI strings at scale 2.
    font.Draw(rnd, 8, 70, "Scene Tree / Inspector / AI Console", 2);
    font.Draw(rnd, 8, 92, "orb  bump  wave  mound  torus", 2);
    font.Draw(rnd, 8, 114, "mass 1.00  bounce 0.40  density 1000  Young 2.0e+06", 2);
    font.Draw(rnd, 8, 136, "generate: a weathered stone torus with cracks", 2);
    font.Draw(rnd, 8, 158, "0123456789 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", 2);
    font.Draw(rnd, 8, 180, "the quick brown fox jumps over the lazy dog", 2);
    font.Draw(rnd, 8, 202, "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", 2);

    std::vector<unsigned char> px(size_t(W) * H * 4);
    SDL_RenderReadPixels(rnd, nullptr, SDL_PIXELFORMAT_RGBA32, px.data(), W * 4);
    if (!write_ppm(out, px, W, H)) {
        std::fprintf(stderr, "could not write %s\n", out);
        return 1;
    }
    // --- assertions on the GEOMETRY the UI depends on ---------------------
    int bad = 0;
    if (BitmapFont::Measure("abc", 2) != 3 * 6 * 2) { std::fprintf(stderr, "FAIL Measure\n"); ++bad; }
    if (BitmapFont::Measure("abc", 1) != 18) { std::fprintf(stderr, "FAIL Measure scale1\n"); ++bad; }
    const std::string e = BitmapFont::Ellipsize("abcdefghij", 36, 1);   // 6 chars fit
    if (e != "abc...") { std::fprintf(stderr, "FAIL Ellipsize -> '%s'\n", e.c_str()); ++bad; }
    if (BitmapFont::Ellipsize("abc", 36, 1) != "abc") { std::fprintf(stderr, "FAIL Ellipsize no-cut\n"); ++bad; }

    // Count lit pixels: a blank atlas means the glyph table is all zeros.
    long lit = 0;
    for (size_t i = 0; i < px.size(); i += 4)
        if (px[i] > 100) ++lit;
    if (lit < 3000) { std::fprintf(stderr, "FAIL too few lit px: %ld\n", lit); ++bad; }
    std::printf("ui_font_probe: wrote %s (%dx%d) lit_px=%ld %s\n",
                out, W, H, lit, bad ? "FAILURES" : "OK");
    font.Shutdown();
    SDL_DestroyRenderer(rnd);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return bad ? 1 : 0;
}
#else
int main() {
    std::fprintf(stderr, "ui_font_probe needs SDL (DELIRIUM_HAS_SDL)\n");
    return 1;
}
#endif
