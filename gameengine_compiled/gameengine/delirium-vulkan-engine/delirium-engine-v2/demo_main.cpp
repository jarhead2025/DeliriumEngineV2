#include "demo_scene.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::cout << "DeliRium Engine v2 - Mathematical Creative Workshop\n";
    delirium::DemoScene scene;
    if (!scene.initialize()) {
        std::cerr << "Failed to initialize engine\n";
        return 1;
    }
    scene.run();
    scene.shutdown();
    std::cout << "Engine shut down cleanly\n";
    return 0;
}