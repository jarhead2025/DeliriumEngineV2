// =============================================================================
// Font.h - Embedded 5x7 bitmap font for the Delirium workspace UI.
//
// WHY EMBEDDED: SDL2_ttf is not available on this machine (checked: no
// sdl2_ttf pkg-config). The workspace needs real TEXT -- scene-tree names,
// inspector values, chat messages -- and the old Editor.cpp only had a
// digits-only 3x5 glyph set (labels had to go to the console).
//
// The glyph table below is the classic public-domain 5x7 ASCII font, one
// byte per COLUMN (5 columns x 7 rows = 5 bytes/glyph, bit 0 = top row).
// Each glyph is 5 bytes; the printable range 32..126 = 95 glyphs.
//
// NO per-frame allocation: the atlas is built once into an SDL_Texture.
// =============================================================================
#pragma once
#include <cstdint>
#include <string>

struct SDL_Texture;
struct SDL_Renderer;

namespace DeliRium {

class BitmapFont {
public:
    // Build the glyph atlas texture. Must be called with a valid renderer.
    // Returns false if the texture could not be created.
    bool Init(SDL_Renderer* renderer);
    void Shutdown();

    // Draw text at (x,y) in pixels. `scale` multiplies the 5x7 cell.
    // Returns the advance width in pixels.
    int Draw(SDL_Renderer* r, float x, float y, const std::string& text,
             int scale = 1) const;

    // Draw text clipped to a rectangle (the text is cut, not wrapped).
    int DrawClipped(SDL_Renderer* r, float x, float y, int clip_x, int clip_y,
                    int clip_w, int clip_h, const std::string& text,
                    int scale = 1) const;

    // Pixel width of `text` at `scale` (no rendering).
    static int Measure(const std::string& text, int scale = 1);

    // Truncate `text` so it fits `max_px` pixels, appending "..." if cut.
    static std::string Ellipsize(const std::string& text, int max_px, int scale = 1);

    // The atlas texture, so callers can tint the glyphs (SDL_SetTextureColorMod).
    // The atlas is built WHITE with a transparent background; tinting it is
    // how the UI draws colored text without a second atlas per color.
    SDL_Texture* Texture() const { return texture_; }

    static constexpr int kGlyphW = 5;
    static constexpr int kGlyphH = 7;
    static constexpr int kAdvance = 6;   // 5 columns + 1 spacing column
    static constexpr int kFirst = 32;     // space
    static constexpr int kLast  = 126;    // '~'
    static constexpr int kCount = kLast - kFirst + 1;

private:
    SDL_Texture* texture_ = nullptr;
};

} // namespace DeliRium
