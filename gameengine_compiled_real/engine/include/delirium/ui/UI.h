// =============================================================================
// UI.h - Immediate-mode-ish widget toolkit for the DeliRium workspace.
//
// WHY IMMEDIATE MODE: a retained widget tree for a docked editor is a lot of
// machinery (layout solvers, focus chains, invalidation) for something whose
// layout is FIXED (a scene tree on the left, an inspector on the right, a
// console at the bottom). Immediate mode makes the workspace code read like
// the layout it draws: "if the mouse is over this rect and the button was
// clicked, do the thing".
//
// WHY NO ImGui: the project has zero third-party UI dependencies and
// SDL2_ttf is unavailable on this machine (the reason the embedded 5x7
// bitmap font exists). A hand-rolled widget layer over the embedded font is
// ~600 lines and has no build-system risk.
//
// All coordinates are WINDOW pixels. The 3D viewport occupies a rectangle in
// the middle; panels are drawn around it. Widgets never allocate per frame
// except for the text buffer the caller owns.
//
// SDL types are kept out of this header (void* for the renderer) so the
// header stays dependency-free like the rest of the engine.
// =============================================================================
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "Font.h"

struct SDL_Renderer;

namespace DeliRium {
namespace UI {

// --- theme -----------------------------------------------------------------
struct Theme {
    // panel backgrounds (RGB)
    unsigned char bg_dark[3]    = { 18, 20, 26 };
    unsigned char bg_panel[3]   = { 26, 30, 38 };
    unsigned char bg_header[3]  = { 34, 40, 51 };
    unsigned char bg_field[3]   = { 14, 16, 21 };
    unsigned char bg_button[3]  = { 44, 52, 66 };
    unsigned char bg_active[3]  = { 52, 92, 120 };
    unsigned char bg_hover[3]   = { 58, 68, 84 };
    unsigned char border[3]     = { 48, 56, 70 };
    // text
    unsigned char fg[3]         = { 226, 232, 240 };
    unsigned char fg_dim[3]     = { 138, 148, 165 };
    unsigned char fg_accent[3]  = { 120, 190, 240 };
    unsigned char fg_warn[3]    = { 240, 180, 100 };
    unsigned char fg_bad[3]     = { 240, 110, 110 };
    unsigned char fg_good[3]    = { 130, 210, 150 };
};

const Theme& ThemeDefault();

// --- geometry --------------------------------------------------------------
struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
    bool Contains(int px, int py) const {
        return px >= x && px < x + w && py >= y && py < y + h;
    }
    bool Contains(const Rect& r) const {
        return r.x >= x && r.y >= y && r.x + r.w <= x + w && r.y + r.h <= y + h;
    }
    Rect Deflate(int d) const { return { x + d, y + d, w - 2*d, h - 2*d }; }
    Rect Deflate(int dx, int dy) const { return { x + dx, y + dy, w - 2*dx, h - 2*dy }; }
    int Right()  const { return x + w; }
    int Bottom() const { return y + h; }
};

// --- the UI context --------------------------------------------------------
// One per frame. The caller pushes mouse/keyboard state in, then calls the
// widget functions, then Draw() renders the whole frame.
//
// The split between "Update" (hit-testing, running callbacks) and "Draw"
// (painting) is deliberate: a widget callback can mutate the scene, and the
// paint pass must reflect the state AFTER all callbacks ran -- otherwise a
// button that adds an object would not appear until the next frame.
class Context {
public:
    // --- per-frame input state ---
    int  mouse_x = 0, mouse_y = 0;
    bool mouse_down = false;         // left button held
    bool mouse_clicked = false;      // left button pressed this frame
    bool mouse_released = false;     // left button released this frame
    bool mouse_right_down = false;
    bool mouse_right_clicked = false;
    int  wheel = 0;
    bool shift = false, ctrl = false, alt = false;

    // --- drag capture -----------------------------------------------------
    //
    // WHY: a drag must belong to the widget that STARTED it. Without
    // capture, dragging a number field and moving the cursor off it makes
    // the field stop tracking, and the next widget under the cursor starts
    // scrubbing -- so one drag edits two unrelated values, and a drag that
    // wanders over a slider yanks the slider too. The user reported exactly
    // this: "when I drag around more outwardly I end up clicking and
    // dragging sliding bars, overlapping them by simply moving it around".
    //
    // The capture owner is a widget id. While it is set, ONLY that widget
    // responds to the drag, everywhere in the window. It is released when
    // the button comes up.
    std::string drag_capture;
    // True while the mouse button that started the capture is held.
    bool drag_active = false;
    // The mouse position where the capture started -- a scrub widget uses
    // the DELTA from here, so the value does not jump when the cursor
    // crosses the widget's boundary.
    int  drag_origin_x = 0, drag_origin_y = 0;
    // The value a scrub widget had when the drag started (for undo).
    float drag_origin_value = 0.f;
    // Set when the drag travelled far enough to count as a DRAG rather than
    // a CLICK. A click applies a tool; a drag moves the camera.
    bool drag_moved = false;

    void CaptureDrag(const std::string& id, float value = 0.f);
    void ReleaseDrag() { drag_capture.clear(); drag_active = false; drag_moved = false; }
    // True when this widget owns the drag.
    bool OwnsDrag(const std::string& id) const { return drag_active && drag_capture == id; }
    // True when the drag has moved far enough to be a drag, not a click.
    bool DragMoved() const { return drag_moved; }

    // --- keyboard ---
    // Text input is delivered as UTF-8 bytes; the widget layer stores the
    // raw bytes and the workspace decodes for display. SDL delivers UTF-8
    // text input events, so this is already UTF-8.
    std::string text_input;           // characters typed this frame
    int  key_backspace = 0;           // presses this frame
    int  key_delete = 0;
    int  key_left = 0, key_right = 0, key_up = 0, key_down = 0;
    int  key_home = 0, key_end = 0;
    int  key_enter = 0, key_escape = 0, key_tab = 0;
    int  key_backspace_held = 0;      // repeats

    // --- layout / state ---
    SDL_Renderer* renderer = nullptr;
    const BitmapFont* font = nullptr;
    const Theme* theme = &ThemeDefault();
    int  width = 0, height = 0;

    // The focused text field. Exactly one may hold focus at a time; a
    // widget claims focus when clicked and gives it up when it loses the
    // pointer. Empty = no field focused.
    std::string focused_field;

    void BeginFrame(SDL_Renderer* r, const BitmapFont* f, int w, int h);
    void EndFrame();                 // paints the recorded draw list

    // --- primitives ---
    void FillRect(const Rect& r, const unsigned char rgb[3], unsigned char alpha = 255);
    void DrawRect(const Rect& r, const unsigned char rgb[3]);
    void Text(const Rect& r, const std::string& s, const unsigned char rgb[3],
              int scale = 1, bool centered = false);
    void TextClipped(const Rect& r, const std::string& s, const unsigned char rgb[3],
                     int scale = 1, bool centered = false);
    // A panel with a title bar. Returns the CONTENT rect (below the bar).
    Rect Panel(const Rect& r, const std::string& title);
    void HSeparator(const Rect& r);
    void VSeparator(const Rect& r);

    // --- widgets (call during update; they also queue their paint) ---
    bool Button(const Rect& r, const std::string& label);
    bool ToggleButton(const Rect& r, const std::string& label, bool* state);
    bool IconButton(const Rect& r, const char glyph);
    // A single-line editable field. Returns true if the text changed.
    bool TextField(const Rect& r, const std::string& id, std::string* text,
                   const std::string& placeholder = "");
    // A labelled numeric drag field: click+drag horizontally to change the
    // value. Returns true if the value changed.
    bool NumberField(const Rect& r, const std::string& id, float* value,
                     float step, float min_v, float max_v, int decimals = 3);
    bool Slider(const Rect& r, const std::string& id, float* value,
                float min_v, float max_v);
    bool CheckBox(const Rect& r, const std::string& id, bool* state,
                  const std::string& label);
    // A drop-down. `items` is the full list; `selected` the current index.
    // Returns true if the selection changed.
    bool ComboBox(const Rect& r, const std::string& id,
                  const std::vector<std::string>& items, int* selected);
    // A tab strip. Returns the newly selected index (-1 if unchanged).
    int TabBar(const Rect& r, const std::string& id,
               const std::vector<std::string>& labels, int current);
    // A scrollable list viewport. `scroll` is in pixels and is clamped.
    // Returns the (possibly adjusted) scroll offset.
    int ScrollArea(const Rect& r, const std::string& id, int* scroll, int content_h);
    // A read-only multi-line log view with a scrollbar.
    void LogView(const Rect& r, const std::string& id,
                 const std::vector<std::string>& lines, int* scroll);

    // --- helpers ---
    bool MouseOver(const Rect& r) const { return r.Contains(mouse_x, mouse_y); }
    bool MouseClicked(const Rect& r) const { return mouse_clicked && MouseOver(r); }
    bool KeyHit(const std::string& id, int key) const;

private:
    struct DrawCmd {
        enum Kind { Fill, Border, Text, TextClip } kind;
        Rect r;
        unsigned char rgb[3] = {0,0,0};
        unsigned char alpha = 255;
        int scale = 1;
        bool centered = false;
        std::string text;
    };
    std::vector<DrawCmd> draw_list_;
    // Combo boxes need their popup drawn ABOVE other widgets, and the popup
    // is only known during update. It is painted as a special last entry.
    struct Popup {
        bool active = false;
        Rect r;
        std::vector<std::string> items;
        int selected = 0;
        int hover = -1;
    };
    Popup popup_;
    std::vector<std::string> popup_owner_;

    void Queue(DrawCmd c) { draw_list_.push_back(std::move(c)); }
    // A stable per-widget id -> geometry map so hit-testing works for
    // widgets that are drawn in a different order than they update.
    struct WidgetState { Rect r; bool hovered = false; bool focused = false; };
    std::vector<std::pair<std::string, WidgetState>> states_;

public:
    // Internal: used by the widget functions.
    void NoteWidget(const std::string& id, const Rect& r, bool hovered, bool focused);
    const WidgetState* FindWidget(const std::string& id) const;
};

// --- formatting helpers (no allocation surprises) --------------------------
std::string FormatFloat(float v, int decimals);
std::string FormatInt(int v);
std::string FormatBytes(size_t n);

// A word-wrapped label for a rect (returns the number of lines drawn).
int WrapText(const BitmapFont& font, const Rect& r, const std::string& s,
             int scale, int max_lines, std::vector<std::string>* out_lines);

} // namespace UI
} // namespace DeliRium
