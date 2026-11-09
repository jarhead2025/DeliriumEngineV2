// FILEPATH: engine/src/ui/UI.cpp
// Widget toolkit implementation. Draw commands are queued during the update
// pass and painted in EndFrame(), so a callback that mutates the scene is
// reflected in the same frame's paint.
#include "delirium/ui/UI.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

namespace DeliRium {
namespace UI {

const Theme& ThemeDefault() {
    static const Theme t;
    return t;
}

// --- formatting ------------------------------------------------------------
std::string FormatFloat(float v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

std::string FormatInt(int v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d", v);
    return buf;
}

std::string FormatBytes(size_t n) {
    char buf[48];
    if (n < 1024) std::snprintf(buf, sizeof(buf), "%zu B", n);
    else if (n < 1024*1024) std::snprintf(buf, sizeof(buf), "%.1f KB", n/1024.0);
    else if (n < 1024ull*1024*1024) std::snprintf(buf, sizeof(buf), "%.1f MB", n/(1024.0*1024.0));
    else std::snprintf(buf, sizeof(buf), "%.2f GB", n/(1024.0*1024.0*1024.0));
    return buf;
}

int WrapText(const BitmapFont& font, const Rect& r, const std::string& s,
             int scale, int max_lines, std::vector<std::string>* out_lines) {
    if (out_lines) out_lines->clear();
    const int max_px = r.w;
    const int line_h = BitmapFont::kGlyphH * scale + 2;
    const int max_rows = r.h / line_h;
    int rows = max_lines > 0 ? std::min(max_rows, max_lines) : max_rows;
    if (rows <= 0) return 0;

    // Split on explicit newlines first, then word-wrap each piece.
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size() && (int)out.size() < rows) {
        size_t nl = s.find('\n', start);
        std::string line = s.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
        // word wrap
        std::string cur;
        size_t i = 0;
        while (i < line.size() && (int)out.size() < rows) {
            size_t sp = line.find(' ', i);
            std::string word = line.substr(i, sp == std::string::npos ? std::string::npos : sp - i);
            std::string cand = cur.empty() ? word : cur + " " + word;
            if (BitmapFont::Measure(cand, scale) > max_px && !cur.empty()) {
                out.push_back(cur);
                cur = word;
            } else {
                cur = cand;
            }
            if (sp == std::string::npos) { i = line.size(); break; }
            i = sp + 1;
        }
        if (!cur.empty() && (int)out.size() < rows) out.push_back(cur);
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    if (out_lines) *out_lines = out;
    return (int)out.size();
}

// --- context ---------------------------------------------------------------
void Context::CaptureDrag(const std::string& id, float value) {
    drag_capture = id;
    drag_active = true;
    drag_origin_x = mouse_x;
    drag_origin_y = mouse_y;
    drag_origin_value = value;
    drag_moved = false;
}

void Context::BeginFrame(SDL_Renderer* r, const BitmapFont* f, int w, int h) {
    renderer = r;
    font = f;
    width = w; height = h;
    draw_list_.clear();
    states_.clear();
    popup_.active = false;
    popup_owner_.clear();
    // A drag ends when the button comes up. While it is held, the capture
    // owner keeps the drag even if the cursor leaves its rect.
    if (mouse_released) ReleaseDrag();
    // A drag that has travelled far enough is a DRAG, not a CLICK. The
    // threshold is small (3 px) so a deliberate scrub registers, but a
    // slightly shaky click does not.
    if (drag_active) {
        const int dx = mouse_x - drag_origin_x;
        const int dy = mouse_y - drag_origin_y;
        if (dx * dx + dy * dy > 9) drag_moved = true;
    }
}

void Context::EndFrame() {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (!renderer) return;
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    for (const DrawCmd& c : draw_list_) {
        switch (c.kind) {
            case DrawCmd::Fill: {
                SDL_SetRenderDrawColor(renderer, c.rgb[0], c.rgb[1], c.rgb[2], c.alpha);
                SDL_Rect r{ c.r.x, c.r.y, c.r.w, c.r.h };
                SDL_RenderFillRect(renderer, &r);
                break;
            }
            case DrawCmd::Border: {
                SDL_SetRenderDrawColor(renderer, c.rgb[0], c.rgb[1], c.rgb[2], c.alpha);
                SDL_Rect r{ c.r.x, c.r.y, c.r.w, c.r.h };
                SDL_RenderDrawRect(renderer, &r);
                break;
            }
            case DrawCmd::Text: {
                if (!font) break;
                // The font atlas is white; tint it by drawing a colored
                // rect masked by the glyph alpha would need a temp texture.
                // Instead the atlas is built white and we draw it with
                // SDL_SetTextureColorMod for the foreground color.
                SDL_Texture* tex = (SDL_Texture*)font->Texture();
                if (!tex) break;
                SDL_SetTextureColorMod(tex, c.rgb[0], c.rgb[1], c.rgb[2]);
                font->Draw(renderer, (float)c.r.x, (float)c.r.y, c.text, c.scale);
                break;
            }
            case DrawCmd::TextClip: {
                if (!font) break;
                SDL_Texture* tex = (SDL_Texture*)font->Texture();
                if (!tex) break;
                SDL_SetTextureColorMod(tex, c.rgb[0], c.rgb[1], c.rgb[2]);
                font->DrawClipped(renderer, (float)c.r.x, (float)c.r.y,
                                  c.r.x, c.r.y, c.r.w, c.r.h, c.text, c.scale);
                break;
            }
        }
    }
    // The combo popup is painted LAST so it sits above every other widget.
    if (popup_.active && !popup_.items.empty()) {
        SDL_SetRenderDrawColor(renderer, theme->bg_panel[0], theme->bg_panel[1], theme->bg_panel[2], 250);
        SDL_Rect pr{ popup_.r.x, popup_.r.y, popup_.r.w, popup_.r.h };
        SDL_RenderFillRect(renderer, &pr);
        SDL_SetRenderDrawColor(renderer, theme->border[0], theme->border[1], theme->border[2], 255);
        SDL_RenderDrawRect(renderer, &pr);
        const int ih = 14;
        for (size_t i = 0; i < popup_.items.size(); ++i) {
            const Rect ir{ popup_.r.x, popup_.r.y + int(i)*ih, popup_.r.w, ih };
            if (int(i) == popup_.hover) {
                SDL_SetRenderDrawColor(renderer, theme->bg_hover[0], theme->bg_hover[1], theme->bg_hover[2], 255);
                SDL_Rect hr{ ir.x+1, ir.y, ir.w-2, ir.h };
                SDL_RenderFillRect(renderer, &hr);
            }
            if (font) {
                SDL_Texture* tex = (SDL_Texture*)font->Texture();
                if (tex) {
                    const bool sel = int(i) == popup_.selected;
                    const unsigned char* col = sel ? theme->fg_accent : theme->fg;
                    SDL_SetTextureColorMod(tex, col[0], col[1], col[2]);
                    font->Draw(renderer, (float)(ir.x+4), (float)(ir.y+3), popup_.items[i], 1);
                }
            }
        }
    }
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);
    SDL_SetTextureColorMod(nullptr, 255,255,255);
#endif
}

void Context::FillRect(const Rect& r, const unsigned char rgb[3], unsigned char alpha) {
    DrawCmd c; c.kind = DrawCmd::Fill; c.r = r;
    std::memcpy(c.rgb, rgb, 3); c.alpha = alpha;
    Queue(std::move(c));
}

void Context::DrawRect(const Rect& r, const unsigned char rgb[3]) {
    DrawCmd c; c.kind = DrawCmd::Border; c.r = r;
    std::memcpy(c.rgb, rgb, 3); c.alpha = 255;
    Queue(std::move(c));
}

void Context::Text(const Rect& r, const std::string& s, const unsigned char rgb[3],
                   int scale, bool centered) {
    DrawCmd c; c.kind = DrawCmd::Text; c.r = r;
    std::memcpy(c.rgb, rgb, 3); c.scale = scale; c.centered = centered; c.text = s;
    if (centered && font) {
        const int w = BitmapFont::Measure(s, scale);
        c.r.x = r.x + (r.w - w)/2;
    }
    Queue(std::move(c));
}

void Context::TextClipped(const Rect& r, const std::string& s, const unsigned char rgb[3],
                          int scale, bool centered) {
    DrawCmd c; c.kind = DrawCmd::TextClip; c.r = r;
    std::memcpy(c.rgb, rgb, 3); c.scale = scale; c.centered = centered;
    c.text = centered && font ? BitmapFont::Ellipsize(s, r.w, scale) : s;
    Queue(std::move(c));
}

Rect Context::Panel(const Rect& r, const std::string& title) {
    FillRect(r, theme->bg_panel);
    DrawRect(r, theme->border);
    const int bar_h = 16;
    FillRect(Rect{ r.x, r.y, r.w, bar_h }, theme->bg_header);
    Text(Rect{ r.x + 6, r.y + 4, r.w - 12, bar_h }, title, theme->fg_accent, 1);
    return Rect{ r.x + 1, r.y + bar_h + 1, r.w - 2, r.h - bar_h - 2 };
}

void Context::HSeparator(const Rect& r) {
    FillRect(Rect{ r.x, r.y + r.h/2, r.w, 1 }, theme->border);
}

void Context::VSeparator(const Rect& r) {
    FillRect(Rect{ r.x + r.w/2, r.y, 1, r.h }, theme->border);
}

void Context::NoteWidget(const std::string& id, const Rect& r, bool hovered, bool focused) {
    states_.push_back({ id, WidgetState{ r, hovered, focused } });
}

const Context::WidgetState* Context::FindWidget(const std::string& id) const {
    for (const auto& kv : states_)
        if (kv.first == id) return &kv.second;
    return nullptr;
}

bool Context::KeyHit(const std::string& id, int key) const {
    // Reserved for widgets that want to know whether a specific key was
    // pressed while they owned focus.
    (void)id; (void)key;
    return false;
}

// --- widgets ---------------------------------------------------------------
bool Context::Button(const Rect& r, const std::string& label) {
    const bool hover = MouseOver(r);
    const bool clicked = mouse_clicked && hover;
    const unsigned char* bg = hover ? theme->bg_hover : theme->bg_button;
    FillRect(r, bg);
    DrawRect(r, theme->border);
    Text(Rect{ r.x, r.y + (r.h - BitmapFont::kGlyphH)/2, r.w, BitmapFont::kGlyphH },
         label, theme->fg, 1, true);
    return clicked;
}

bool Context::ToggleButton(const Rect& r, const std::string& label, bool* state) {
    const bool hover = MouseOver(r);
    const bool clicked = mouse_clicked && hover;
    FillRect(r, *state ? theme->bg_active : (hover ? theme->bg_hover : theme->bg_button));
    DrawRect(r, theme->border);
    Text(Rect{ r.x, r.y + (r.h - BitmapFont::kGlyphH)/2, r.w, BitmapFont::kGlyphH },
         label, *state ? theme->fg : theme->fg_dim, 1, true);
    if (clicked) *state = !*state;
    return clicked;
}

bool Context::IconButton(const Rect& r, const char glyph) {
    const bool hover = MouseOver(r);
    const bool clicked = mouse_clicked && hover;
    FillRect(r, hover ? theme->bg_hover : theme->bg_button);
    DrawRect(r, theme->border);
    Text(Rect{ r.x, r.y + (r.h - BitmapFont::kGlyphH)/2, r.w, BitmapFont::kGlyphH },
         std::string(1, glyph), theme->fg, 1, true);
    return clicked;
}

bool Context::TextField(const Rect& r, const std::string& id, std::string* text,
                        const std::string& placeholder) {
    if (!text) return false;
    const bool hover = MouseOver(r);
    const bool focused = (focused_field == id);
    bool changed = false;

    if (mouse_clicked && hover) {
        // Clicking another field moves focus; clicking this one focuses it.
        focused_field = id;
    } else if (mouse_clicked && !hover && focused) {
        focused_field.clear();
    }

    // While focused, the field consumes the keyboard.
    if (focused) {
        if (!text_input.empty()) { *text += text_input; changed = true; }
        if (key_backspace > 0 && !text->empty()) {
            // Erase one UTF-8 code point (not one byte).
            size_t n = text->size();
            while (n > 0 && (static_cast<unsigned char>((*text)[n-1]) & 0xC0) == 0x80) --n;
            if (n > 0) --n;
            text->resize(n);
            changed = true;
        }
        if (key_delete > 0 && !text->empty()) {
            size_t n = text->size();
            while (n > 0 && (static_cast<unsigned char>((*text)[n-1]) & 0xC0) == 0x80) --n;
            if (n > 0) --n;
            text->erase(n);
            changed = true;
        }
        if (key_left > 0 || key_right > 0) {
            // Cursor movement is not yet modelled -- the caret stays at the
            // end. Left/right are consumed so they do not leak to the
            // viewport camera.
        }
        if (key_escape > 0) focused_field.clear();
        if (key_enter > 0) focused_field.clear();
    }

    FillRect(r, theme->bg_field);
    DrawRect(r, focused ? theme->fg_accent : theme->border);
    if (text->empty() && !focused && !placeholder.empty())
        TextClipped(r.Deflate(4, 0), placeholder, theme->fg_dim, 1);
    else
        TextClipped(r.Deflate(4, 0), *text, theme->fg, 1);
    if (focused) {
        // A caret block at the end of the text.
        const int w = font ? BitmapFont::Measure(*text, 1) : 0;
        const int cx = r.x + 4 + w;
        if ((mouse_x / 250) % 2 == 0)
            FillRect(Rect{ cx, r.y + 3, 5, BitmapFont::kGlyphH }, theme->fg_accent);
    }
    NoteWidget(id, r, hover, focused);
    return changed;
}

bool Context::NumberField(const Rect& r, const std::string& id, float* value,
                          float step, float min_v, float max_v, int decimals) {
    if (!value) return false;
    bool changed = false;

    // DRAG CAPTURE: this widget owns the drag only if it started it.
    //
    // The old code scrubbed whenever `mouse_down && hover`, so a drag that
    // wandered off the field stopped scrubbing it and the NEXT widget under
    // the cursor started scrubbing instead -- one gesture edited two
    // unrelated fields, and sliders got yanked by a mere mouse move. With
    // capture, the field that was pressed keeps the drag until the button
    // is released, wherever the cursor goes.
    if (mouse_clicked && r.Contains(mouse_x, mouse_y)) {
        CaptureDrag(id, *value);
        focused_field = id;
    }

    if (OwnsDrag(id)) {
        // Scrub by the DELTA from where the drag started. Using the
        // absolute cursor position would make the value jump the moment the
        // cursor crossed the field's edge.
        const int dx = mouse_x - drag_origin_x;
        if (dx != 0) {
            *value = drag_origin_value + step * float(dx) * 0.25f;
            changed = true;
        }
    }

    if (focused_field == id) {
        if (key_up > 0)   { *value += step; changed = true; }
        if (key_down > 0) { *value -= step; changed = true; }
    }
    if (*value < min_v) *value = min_v;
    if (*value > max_v) *value = max_v;

    const bool hover = r.Contains(mouse_x, mouse_y);
    FillRect(r, theme->bg_field);
    DrawRect(r, hover ? theme->fg_accent : theme->border);
    Text(Rect{ r.x + 4, r.y + (r.h - BitmapFont::kGlyphH)/2, r.w - 8, BitmapFont::kGlyphH },
         FormatFloat(*value, decimals), theme->fg, 1);
    NoteWidget(id, r, hover, focused_field == id);
    return changed;
}

bool Context::Slider(const Rect& r, const std::string& id, float* value,
                     float min_v, float max_v) {
    if (!value || max_v <= min_v) return false;
    bool changed = false;
    const Rect track{ r.x, r.y + r.h/2 - 2, r.w, 4 };

    // Same capture contract as NumberField: the slider that was pressed
    // keeps the drag. Without it, sliding a value and drifting the cursor
    // onto another slider teleports that one too.
    if (mouse_clicked && r.Contains(mouse_x, mouse_y)) {
        CaptureDrag(id, *value);
    }
    if (OwnsDrag(id)) {
        const float nt = float(mouse_x - track.x) / float(std::max(1, track.w));
        const float nv = min_v + (max_v - min_v) * (nt < 0.f ? 0.f : (nt > 1.f ? 1.f : nt));
        if (nv != *value) { *value = nv; changed = true; }
    }

    const bool hover = r.Contains(mouse_x, mouse_y);
    FillRect(track, theme->bg_field);
    const float t = (*value - min_v) / (max_v - min_v);
    const int fw = int(float(r.w) * (t < 0.f ? 0.f : (t > 1.f ? 1.f : t)));
    FillRect(Rect{ track.x, track.y, fw, track.h }, theme->fg_accent);
    const int knob_x = track.x + fw - 4;
    FillRect(Rect{ knob_x, r.y + 2, 8, r.h - 4 }, hover ? theme->fg_accent : theme->fg_dim);
    NoteWidget(id, r, hover, false);
    return changed;
}

bool Context::CheckBox(const Rect& r, const std::string& id, bool* state,
                       const std::string& label) {
    if (!state) return false;
    const bool hover = MouseOver(r);
    const bool clicked = mouse_clicked && hover;
    const int box = 11;
    const Rect box_r{ r.x, r.y + (r.h - box)/2, box, box };
    FillRect(box_r, theme->bg_field);
    DrawRect(box_r, hover ? theme->fg_accent : theme->border);
    if (*state) {
        FillRect(Rect{ box_r.x + 3, box_r.y + 3, box - 6, box - 6 }, theme->fg_accent);
    }
    Text(Rect{ r.x + box + 5, r.y + (r.h - BitmapFont::kGlyphH)/2, r.w - box - 5, BitmapFont::kGlyphH },
         label, theme->fg, 1);
    if (clicked) *state = !*state;
    NoteWidget(id, r, hover, false);
    return clicked;
}

bool Context::ComboBox(const Rect& r, const std::string& id,
                       const std::vector<std::string>& items, int* selected) {
    if (!selected || items.empty()) return false;
    const bool hover = MouseOver(r);
    bool changed = false;

    // The popup is owned by exactly one combo. A click inside the open
    // popup selects the item under the mouse.
    if (popup_.active) {
        if (mouse_clicked) {
            const int ih = 14;
            const int idx = (mouse_y - popup_.r.y) / ih;
            if (mouse_x >= popup_.r.x && mouse_x < popup_.r.Right() &&
                idx >= 0 && idx < int(popup_.items.size())) {
                if (*selected != idx) { *selected = idx; changed = true; }
            }
            popup_.active = false;
            popup_owner_.clear();
        } else {
            const int ih = 14;
            const int idx = (mouse_y - popup_.r.y) / ih;
            popup_.hover = (mouse_x >= popup_.r.x && mouse_x < popup_.r.Right() &&
                            idx >= 0 && idx < int(popup_.items.size())) ? idx : -1;
        }
    } else if (mouse_clicked && hover) {
        popup_.active = true;
        popup_.r = Rect{ r.x, r.Bottom(), r.w, int(items.size()) * 14 };
        popup_.items = items;
        popup_.selected = *selected;
        popup_.hover = -1;
        popup_owner_.push_back(id);
    }

    FillRect(r, theme->bg_field);
    DrawRect(r, hover || (popup_.active && !popup_owner_.empty() &&
                          popup_owner_[0] == id) ? theme->fg_accent : theme->border);
    const int idx = *selected >= 0 && *selected < int(items.size()) ? *selected : 0;
    TextClipped(Rect{ r.x + 4, r.y + (r.h - BitmapFont::kGlyphH)/2, r.w - 14, BitmapFont::kGlyphH },
                items[idx], theme->fg, 1);
    // A small down-arrow.
    Text(Rect{ r.Right() - 10, r.y + (r.h - BitmapFont::kGlyphH)/2, 8, BitmapFont::kGlyphH },
         "v", theme->fg_dim, 1, true);
    NoteWidget(id, r, hover, false);
    return changed;
}

int Context::TabBar(const Rect& r, const std::string& id,
                    const std::vector<std::string>& labels, int current) {
    if (labels.empty()) return -1;
    const int tw = std::max(40, r.w / int(labels.size()));
    int result = -1;
    for (size_t i = 0; i < labels.size(); ++i) {
        const Rect tr{ r.x + int(i)*tw, r.y, tw, r.h };
        const bool hover = MouseOver(tr);
        const bool sel = int(i) == current;
        FillRect(tr, sel ? theme->bg_active : (hover ? theme->bg_hover : theme->bg_header));
        if (sel) FillRect(Rect{ tr.x, tr.Bottom()-2, tr.w, 2 }, theme->fg_accent);
        Text(Rect{ tr.x, tr.y + (tr.h - BitmapFont::kGlyphH)/2, tr.w, BitmapFont::kGlyphH },
             labels[i], sel ? theme->fg : theme->fg_dim, 1, true);
        if (mouse_clicked && hover) result = int(i);
    }
    NoteWidget(id, r, false, false);
    return result;
}

int Context::ScrollArea(const Rect& r, const std::string& id, int* scroll, int content_h) {
    if (!scroll) return 0;
    const int max_scroll = std::max(0, content_h - r.h);
    int s = *scroll;
    if (wheel && MouseOver(r)) {
        s -= wheel * 18;
    }
    if (s < 0) s = 0;
    if (s > max_scroll) s = max_scroll;
    *scroll = s;
    NoteWidget(id, r, MouseOver(r), false);
    return s;
}

void Context::LogView(const Rect& r, const std::string& id,
                      const std::vector<std::string>& lines, int* scroll) {
    FillRect(r, theme->bg_field);
    DrawRect(r, theme->border);
    const int lh = BitmapFont::kGlyphH + 2;
    const int visible = std::max(1, (r.h - 4) / lh);
    const int max_scroll = std::max(0, int(lines.size()) - visible);
    int s = scroll ? *scroll : 0;
    if (wheel && MouseOver(r)) s -= wheel * 18;
    if (s < 0) s = 0;
    if (s > max_scroll) s = max_scroll;
    if (scroll) *scroll = s;

    // Newest line at the bottom (like a terminal) when the view is pinned.
    const int count = int(lines.size());
    for (int i = 0; i < visible; ++i) {
        const int li = s + i;
        if (li >= count) break;
        TextClipped(Rect{ r.x + 3, r.y + 2 + i*lh, r.w - 16, lh },
                    lines[size_t(li)], theme->fg_dim, 1);
    }
    // Scrollbar.
    if (max_scroll > 0) {
        const Rect sb{ r.Right() - 10, r.y + 2, 7, r.h - 4 };
        FillRect(sb, theme->bg_panel);
        const int bar_h = std::max(16, int(float(sb.h) * float(visible) / float(count)));
        const int bar_y = sb.y + int(float(sb.h - bar_h) * float(s) / float(max_scroll));
        FillRect(Rect{ sb.x, bar_y, sb.w, bar_h }, theme->border);
    }
    NoteWidget(id, r, MouseOver(r), false);
}

} // namespace UI
} // namespace DeliRium
