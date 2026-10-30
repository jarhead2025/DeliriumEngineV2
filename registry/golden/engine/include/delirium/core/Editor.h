// =============================================================================
// Editor.h - Interactive editor built ON the engine via its input/HUD hooks.
// Connects every validated operation -- dent, poke, split/refine, merge/
// coarsen, fracture, morph, harmonic weld, material switch -- to keys and
// mouse, and reports the live internal state (patch count, seam gap, file
// size estimate, active tool/material) in the window HUD. This is the
// "see internal changes in real time, like Blender" goal: a single tool
// that exercises the whole pipeline interactively on the GPU.
// =============================================================================
#pragma once
#include "delirium/core/Engine.h"
#include "delirium/core/GameObject.h"
#include "delirium/psf/Seam.h"
#include "delirium/physics/Material.h"
#include <string>
#include <cstdio>

namespace DeliRium {

class Editor {
public:
    enum class Tool { Dent, Poke, Refine, Coarsen, Fracture, Morph, Weld };

    explicit Editor(Engine& eng) : engine(eng) {}

    void Attach() {
        engine.SetInputHook([this](const void* e){ return OnInput(e); });
        engine.SetHudHook([this](){ return Hud(); });
    }

    void SetTarget(GameObject* o) { target = o; }

private:
    Engine&      engine;
    GameObject*  target = nullptr;
    Tool         tool = Tool::Dent;
    int          mat_idx = 1;                 // Elastic
    float        morph_k = 0.0f;
    char         last_action[64] = "ready";

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
