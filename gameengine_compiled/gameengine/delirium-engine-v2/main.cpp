#include "delirium/core/engine.hpp"
#include <iostream>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    delirium::Engine engine;

    if (!engine.init(1280, 720)) {
        std::cerr << "Failed to initialize DeliRium Engine v2\n";
        return 1;
    }

    std::cout << "=== DeliRium Engine v2 ===\n";
    std::cout << "Layer 0: Math core ✓\n";
    std::cout << "Layer 1: Geometry (patch) ✓\n";
    std::cout << "Layer 2: Render (surfel + OpenGL) ✓\n";
    std::cout << "Layer 3: Physics (stub) → TODO\n";
    std::cout << "Layer 4: Generation → TODO\n";
    std::cout << "==========================\n";

    engine.run();
    engine.shutdown();

    return 0;
}