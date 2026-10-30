// =============================================================================
// 01_torus_deform - first frames on a real GPU.
// Loads assets/torus.psf + assets/torus.modes; SPACE pokes the torus
// (modal impulse), ESC exits. Everything on screen is equations.
// =============================================================================
#include "delirium/core/Engine.h"
#include <cstdio>
using namespace DeliRium;

int main() {
    Engine& engine = Engine::Instance();
    if (!engine.Initialize(800, 600, "DeliRium v0.2 - torus deform")) return -1;

    Scene* scene = engine.CreateScene("demo");
    scene->GetCamera().position = {2.6f, -2.2f, 1.6f};
    scene->GetCamera().target   = {0, 0, 0};
    scene->GetCamera().up       = {0, 0, 1};

    ReconstructionRequest req;
    req.semantic = "file:assets/torus.psf";
    req.material.youngs_modulus = 5e5f;
    GameObject* torus = scene->Create("torus", req);
    if (!torus->HasBody()) {
        std::printf("ERROR: could not load assets/torus.psf\n"
                    "Run from the PROJECT ROOT (the folder containing "
                    "'assets/'), e.g.:\n  ./build/example_torus\n");
        return -1;
    }
    if (!torus->LoadModes("assets/torus.modes"))
        std::printf("(no modes file: object will render rigid)\n");

    engine.Run();
    engine.Shutdown();
    return 0;
}
