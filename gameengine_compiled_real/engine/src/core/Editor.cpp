// FILEPATH: engine/src/core/Editor.cpp
// Lightweight SDL editor model: scene selection and live physics inspection.
#include "delirium/core/Editor.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdio>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

namespace DeliRium {

bool Editor::SelectObject(size_t index) {
    const Scene* scene = engine.CurrentScene();
    if (!scene || index >= scene->Objects().size() || !scene->Objects()[index])
        return false;
    selected_index = index;
    target = scene->Objects()[index].get();
    return true;
}

size_t Editor::SelectedIndex() const {
    const Scene* scene = engine.CurrentScene();
    const GameObject* selected = SelectedObject();
    if (scene && selected) {
        const auto& objects = scene->Objects();
        for (size_t i = 0; i < objects.size(); ++i)
            if (objects[i].get() == selected) return i;
    }
    return static_cast<size_t>(-1);
}

const GameObject* Editor::SelectedObject() const {
    const Scene* scene = engine.CurrentScene();
    if (!scene) return nullptr;
    const auto& objects = scene->Objects();
    if (target && selected_index < objects.size() && objects[selected_index].get() == target)
        return target;
    // An externally assigned target is only valid while it belongs to this
    // scene. Never dereference a stale pointer from a replaced scene.
    for (const auto& o : objects)
        if (o.get() == target) return o.get();
    return nullptr;
}

std::string Editor::SceneTreePanel() const {
    const Scene* scene = engine.CurrentScene();
    if (!scene) return "Scene: (none)";
    std::ostringstream out;
    out << "Scene: " << scene->sceneName << '\n';
    const GameObject* selected = SelectedObject();
    const auto& objects = scene->Objects();
    for (size_t i = 0; i < objects.size(); ++i) {
        if (objects[i]) out << (objects[i].get() == selected ? "> " : "  ")
                            << i << ": " << objects[i]->name << '\n';
    }
    return out.str();
}

std::string Editor::InspectorPanel() const {
    const GameObject* selected = SelectedObject();
    if (!selected) return "Inspector: select a scene object";
    const RigidState& rigid = selected->Rigid();
    const Material& mat = selected->Mat();
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "Inspector: " << selected->name << '\n'
        << "mass " << rigid.mass << "  restitution " << rigid.restitution
        << "  friction " << rigid.friction << '\n'
        << "density " << mat.density << "  Young " << mat.youngs_modulus
        << "  Poisson " << mat.poisson << "  damping " << mat.damping << '\n'
        << "patches " << (selected->HasBody() ? selected->Body().Patches().size() : 0);
    return out.str();
}

bool Editor::SetPhysicsProperty(const std::string& key, float value) {
    const GameObject* selected = SelectedObject();
    if (!selected || !std::isfinite(value)) return false;
    GameObject* object = const_cast<GameObject*>(selected);
    RigidState& rigid = object->Rigid();
    Material& mat = object->Mat();
    if (key == "mass" && value >= 0.f) {
        rigid.mass = value;
        rigid.inv_mass = value > 0.f ? 1.f / value : 0.f;
    } else if (key == "restitution" && value >= 0.f && value <= 1.f) {
        rigid.restitution = value;
        mat.restitution = value;
    } else if (key == "friction" && value >= 0.f) {
        rigid.friction = value;
        mat.friction = value;
    } else if (key == "density" && value > 0.f) mat.density = value;
    else if (key == "youngs_modulus" && value > 0.f) mat.youngs_modulus = value;
    else if (key == "poisson" && value > -1.f && value < 0.5f) mat.poisson = value;
    else if (key == "damping" && value >= 0.f) mat.damping = value;
    else return false;
    return true;
}

namespace {
struct PropertyRow { const char* label; const char* key; float step; };
constexpr std::array<PropertyRow, 7> properties{{
    {"Mass", "mass", 0.1f}, {"Bounce", "restitution", 0.05f},
    {"Friction", "friction", 0.05f}, {"Density", "density", 10.f},
    {"Young", "youngs_modulus", 10000.f},
    {"Poisson", "poisson", 0.01f}, {"Damping", "damping", 0.01f}
}};
float property_value(const GameObject& o, int row) {
    switch (row) {
        case 0: return o.Rigid().mass;
        case 1: return o.Rigid().restitution;
        case 2: return o.Rigid().friction;
        case 3: return o.Mat().density;
        case 4: return o.Mat().youngs_modulus;
        case 5: return o.Mat().poisson;
        default: return o.Mat().damping;
    }
}
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
void box(SDL_Renderer* r, SDL_Rect rect, Uint8 red, Uint8 green, Uint8 blue) {
    SDL_SetRenderDrawColor(r, red, green, blue, 235);
    SDL_RenderFillRect(r, &rect);
}
// Tiny built-in numerical font: no TTF dependency or per-frame glyph textures.
// Numeric values and indices are exact; object labels remain in console/title.
void glyph(SDL_Renderer* r, int x, int y, char c, int scale = 2) {
    static constexpr unsigned char digits[10][5] = {
        {7,5,5,5,7},{2,6,2,2,7},{7,1,7,4,7},{7,1,7,1,7},{5,5,7,1,1},
        {7,4,7,1,7},{7,4,7,5,7},{7,1,1,1,1},{7,5,7,5,7},{7,5,7,1,7}
    };
    static constexpr unsigned char minus[5] = {0,0,7,0,0};
    static constexpr unsigned char dot[5] = {0,0,0,0,2};
    const unsigned char* rows = c >= '0' && c <= '9' ? digits[c-'0']
                              : c == '-' ? minus : c == '.' ? dot : nullptr;
    if (!rows) return;
    SDL_SetRenderDrawColor(r, 229, 234, 242, 255);
    for (int v = 0; v < 5; ++v)
        for (int u = 0; u < 3; ++u)
            if (rows[v] & (1u << (2-u))) {
                SDL_Rect p{x+u*scale, y+v*scale, scale, scale};
                SDL_RenderFillRect(r, &p);
            }
}
void number(SDL_Renderer* r, int x, int y, const char* text) {
    for (const char* p = text; *p; ++p, x += 8) glyph(r, x, y, *p);
}
#endif
} // namespace

void Editor::DrawOverlay(void* opaque, int width, int height) const {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (!visible || !opaque || width < 440 || height < 220) return;
    auto* r = static_cast<SDL_Renderer*>(opaque);
    // Keep the center unobstructed; no CPU PSF or GPU framebuffer copies.
    const int panel = std::min(210, width/5), top = 36;
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    box(r, SDL_Rect{0, top, panel, height-top}, 15, 22, 35);
    box(r, SDL_Rect{width-panel, top, panel, height-top}, 19, 27, 39);
    box(r, SDL_Rect{0, 0, width, top}, 28, 42, 58);
    const Scene* scene = engine.CurrentScene();
    const size_t count = scene ? scene->Objects().size() : 0;
    char text[48];
    std::snprintf(text, sizeof(text), "%zu", count);
    number(r, 15, 12, text);
    if (scene) for (size_t i = 0; i < count && i < size_t((height-top-16)/28); ++i) {
        if (!scene->Objects()[i]) continue;
        if (i == SelectedIndex()) box(r, SDL_Rect{4, top+8+int(i)*28, panel-8, 23}, 42, 83, 108);
        std::snprintf(text, sizeof(text), "%zu", i);
        number(r, 15, top+14+int(i)*28, text);
    }
    const GameObject* o = SelectedObject();
    if (o) for (size_t i = 0; i < properties.size() && top+12+int(i)*28+22 < height; ++i) {
        const int x = width-panel+8, y = top+12+int(i)*28;
        if (int(i) == editing_property) box(r, SDL_Rect{x, y-3, panel-16, 23}, 48, 81, 104);
        // Row numbers correspond to Mass/Bounce/Friction/Density/Young/Poisson/Damping
        // as printed in the console on selection. +/- adjust the highlighted row.
        std::snprintf(text, sizeof(text), "%zu", i+1);
        number(r, x+4, y+2, text);
        const float value = property_value(*o, int(i));
        std::snprintf(text, sizeof(text), value >= 1000.f ? "%.0f" : "%.2f", value);
        number(r, x+45, y+2, text);
    }
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
    SDL_SetRenderDrawColor(r, 255, 255, 255, 255);
#else
    (void)opaque; (void)width; (void)height;
#endif
}

bool Editor::OnInput(const void* ev) {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (!ev) return false;
    const SDL_Event& event = *static_cast<const SDL_Event*>(ev);
    if (event.type == SDL_KEYDOWN && !event.key.repeat && event.key.keysym.sym == SDLK_F1) {
        visible = !visible;
        editing_property = -1;
        SDL_SetRelativeMouseMode(visible ? SDL_FALSE : SDL_TRUE);
        if (visible) std::printf("Editor: left=scene tree, right=inspector; F2=mouse look, +/-=edit value, Tab=next, F1=close\n"
                                 "Inspector rows: 1 mass, 2 bounce, 3 friction, 4 density, 5 Young, 6 Poisson, 7 damping\n");
        return true;
    }
    if (!visible) return false;
    if (event.type == SDL_KEYDOWN && !event.key.repeat &&
        event.key.keysym.sym == SDLK_F2) {
        SDL_SetRelativeMouseMode(SDL_GetRelativeMouseMode() == SDL_TRUE ? SDL_FALSE : SDL_TRUE);
        editing_property = -1;
        return true;
    }
    if (SDL_GetRelativeMouseMode() == SDL_TRUE && event.type != SDL_KEYDOWN)
        return false;
    if (SDL_GetRelativeMouseMode() == SDL_TRUE && event.type == SDL_KEYDOWN &&
        event.key.keysym.sym != SDLK_F2) return false;
    if (event.type == SDL_MOUSEMOTION && SDL_GetRelativeMouseMode() == SDL_FALSE)
        return true; // pointer mode: never rotate the viewport behind the UI
    if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
        int w = 0, h = 0;
        if (SDL_Window* win = SDL_GetMouseFocus()) SDL_GetWindowSize(win, &w, &h);
        if (w < 440 || h < 220 || event.button.y < 36) return true;
        const int panel = std::min(210, w/5);
        if (event.button.x < panel) {
            const int row = (event.button.y - 36 - 8)/28;
            if (event.button.y >= 44 && row >= 0 && SelectObject(size_t(row))) {
                std::printf("%s%s\n", SceneTreePanel().c_str(), InspectorPanel().c_str());
                return true;
            }
            return true; // prevent viewport picking through the tree
        }
        if (event.button.x >= w-panel) {
            const int row = (event.button.y - 36 - 12)/28;
            editing_property = event.button.y >= 48 && row >= 0 && row < int(properties.size())
                             ? row : -1;
            return true;
        }
        return true; // pointer mode selects UI only; F2 restores viewport control
    }
    if (event.type == SDL_MOUSEWHEEL || event.type == SDL_MOUSEBUTTONDOWN ||
        event.type == SDL_MOUSEBUTTONUP)
        return true; // editing focus never sends zoom or dent to the viewport
    if (event.type != SDL_KEYDOWN || event.key.repeat) return false;
    if (event.key.keysym.sym == SDLK_ESCAPE) {
        if (editing_property >= 0) editing_property = -1;
        else { visible = false; SDL_SetRelativeMouseMode(SDL_TRUE); }
        return true;
    }
    if (editing_property >= 0 && event.key.keysym.sym == SDLK_TAB) {
        editing_property = (editing_property + 1) % int(properties.size());
        return true;
    }
    if (editing_property >= 0 && SelectedObject() &&
        (event.key.keysym.sym == SDLK_EQUALS || event.key.keysym.sym == SDLK_KP_PLUS ||
         event.key.keysym.sym == SDLK_MINUS || event.key.keysym.sym == SDLK_KP_MINUS)) {
        const float sign = event.key.keysym.sym == SDLK_MINUS || event.key.keysym.sym == SDLK_KP_MINUS ? -1.f : 1.f;
        const auto& row = properties[size_t(editing_property)];
        const float updated = property_value(*SelectedObject(), editing_property) + sign*row.step;
        if (SetPhysicsProperty(row.key, updated)) std::printf("%s\n", InspectorPanel().c_str());
        return true;
    }
    // Keep the editor modal in pointer mode; F2 explicitly returns camera
    // keys to the viewport, while F1 closes it without quitting the app.
    if (event.key.keysym.sym == SDLK_TAB) {
        const Scene* scene = engine.CurrentScene();
        if (!scene || scene->Objects().empty()) return false;
        const auto& objects = scene->Objects();
        size_t current = objects.size();
        const GameObject* selected = SelectedObject();
        for (size_t i = 0; i < objects.size(); ++i)
            if (objects[i].get() == selected && selected) { current = i; break; }
        for (size_t step = 1; step <= objects.size(); ++step) {
            const size_t next = (current == objects.size() ? step - 1
                                  : (current + step) % objects.size());
            if (SelectObject(next)) {
                std::printf("%s%s\n", SceneTreePanel().c_str(), InspectorPanel().c_str());
                return true;
            }
        }
    }
    if (SDL_GetRelativeMouseMode() == SDL_FALSE) return true;
#else
    (void)ev;
#endif
    return false;
}

std::string Editor::Hud() {
    const GameObject* selected = SelectedObject();
    if (!visible) return "F1: editor";
    return selected ? "Editor: " + selected->name + " | Tab: next | " + InspectorPanel()
                    : "Editor: Tab selects object";
}

float Editor::Diag() const {
    const GameObject* selected = SelectedObject();
    if (!selected || !selected->HasBody()) return 0.f;
    const PSFBody& body = selected->Body();
    const Vec3 span = body.BBoxSpan();
    return std::sqrt(span.x*span.x + span.y*span.y + span.z*span.z);
}

} // namespace DeliRium
