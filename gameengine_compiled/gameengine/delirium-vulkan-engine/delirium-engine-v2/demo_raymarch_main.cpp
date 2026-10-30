#include "demo_scene.hpp"

int main() {
    delirium::DemoScene scene;
    if (!scene.initialize()) return 1;
    scene.run();
    scene.shutdown();
    return 0;
}