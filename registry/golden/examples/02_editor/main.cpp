// =============================================================================
// 02_editor - Interactive DeliRium editor.
// Load a model, then sculpt it live on the GPU with the whole pipeline:
//   1 dent  2 poke  3 refine  4 coarsen  5 fracture  6 morph  7 weld
//   M cycle material   W weld now   R reset   ESC quit
//   left/right mouse = apply active tool at the picked surface point
// The window title shows live internal state: patch count, file size,
// seam gap, active tool/material -- "see the internal changes in real
// time, like Blender".
// =============================================================================
#include "delirium/core/Engine.h"
#include "delirium/core/Scene.h"
#include "delirium/core/Editor.h"
#include "delirium/psf/Reconstructor.h"
#include <cstdio>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1] : "assets/roughball.psf";

    Engine& eng = Engine::Instance();
    if (!eng.Initialize(800, 600, "DeliRium Editor")) {
        std::fprintf(stderr, "engine init failed\n"); return -1;
    }
    eng.SetReconstructor(std::make_unique<FixedReconstructor>());

    Scene* scene = eng.CreateScene("edit");
    ReconstructionRequest req; req.semantic = std::string("file:") + model;
    GameObject* obj = scene->Create("model", req);
    if (!obj->HasBody()) {
        std::fprintf(stderr,
            "ERROR: could not load %s\n"
            "Run from the PROJECT ROOT (folder containing 'assets/').\n", model);
        return -1;
    }

    Editor editor(eng);
    editor.SetTarget(obj);
    editor.Attach();

    std::printf("DeliRium Editor: 1-7 tools, M material, W weld, R reset, "
                "mouse to apply. Watch the title bar.\n");
    eng.Run();
    eng.Shutdown();
    return 0;
}
