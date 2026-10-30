// =============================================================================
// Editor.h - SDL-only editor model built on existing Engine input/HUD hooks.
// First slice: scene-object selection, live inspector and validated physics
// property edits. A docked, drawable interface and authored scene persistence
// are later stages; this model does not claim to implement them.
// =============================================================================
#pragma once
#include "delirium/core/Engine.h"
#include "delirium/core/GameObject.h"
#include <string>
#include <cstddef>

namespace DeliRium {

class Editor {
public:
    enum class Tool { Dent, Poke, Refine, Coarsen, Fracture, Morph, Weld };

    explicit Editor(Engine& eng) : engine(eng) {}

    void Attach() {
        engine.SetInputHook([this](const void* e){ return OnInput(e); });
        engine.SetHudHook([this](){ return Hud(); });
        engine.SetOverlayHook([this](void* renderer, int w, int h){ DrawOverlay(renderer, w, h); });
    }

    // Future dock model: no copies of 64-byte PSF records. Selection is an
    // index into Scene::Objects, resolved on demand; bounds are rechecked
    // whenever the scene changes. Properties edit the live GameObject.
    void SetTarget(GameObject* o) { target = o; selected_index = static_cast<size_t>(-1); }
    bool SelectObject(size_t index);
    size_t SelectedIndex() const;
    const GameObject* SelectedObject() const;
    std::string SceneTreePanel() const;
    std::string InspectorPanel() const;
    bool SetPhysicsProperty(const std::string& key, float value);
    // Editor overlay only; the viewport image remains owned by SplatRenderer.
    void DrawOverlay(void* sdl_renderer, int width, int height) const;

private:
    Engine&      engine;
    GameObject*  target = nullptr;
    size_t       selected_index = static_cast<size_t>(-1);
    Tool         tool = Tool::Dent;
    int          mat_idx = 1;                 // Elastic
    float        morph_k = 0.0f;
    char         last_action[64] = "ready";
    int          editing_property = -1; // selected inspector row, -1 = none
    bool         visible = false; // F1 opens editor and releases relative mouse

    static const char* ToolName(Tool t) {
        switch (t) {
            case Tool::Dent:    return "DENT";
            case Tool::Poke:    return "POKE";
            case Tool::Refine:  return "REFINE";
            case Tool::Coarsen: return "COARSEN";
            case Tool::Fracture:return "FRACTURE";
            case Tool::Morph:   return "MORPH";
            case Tool::Weld:    return "WELD";
        }
        return "?";
    }
    static const char* MatName(int i) {
        static const char* n[] = {"Rigid","Elastic","Soft","Fluid","Brittle"};
        return n[i % 5];
    }

    bool OnInput(const void* ev);             // defined in Editor.cpp
    std::string Hud();
    float Diag() const;
};

} // namespace DeliRium
