#include "demo_scene.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string obj_path;
    std::string texture_path;

    if (argc >= 2) {
        obj_path = argv[1];
    }
    if (argc >= 3) {
        texture_path = argv[2];
    }

    if (argc >= 2) {
        std::cout << "[Engine] Loading OBJ: " << obj_path;
        if (!texture_path.empty()) {
            std::cout << " with texture: " << texture_path;
        }
        std::cout << "\n";
    } else {
        std::cout << "[Engine] Running built-in 8x8 grid demo\n";
        std::cout << "[Usage] " << (argc > 0 ? argv[0] : "./delirium_engine")
                  << " [model.obj] [texture.raw]\n";
    }

    delirium::DemoScene scene;

    if (!scene.initialize(obj_path, texture_path)) {
        std::cerr << "Failed to initialize engine\n";
        return 1;
    }

    scene.run();
    scene.shutdown();

    std::cout << "Engine shut down cleanly\n";
    return 0;
}