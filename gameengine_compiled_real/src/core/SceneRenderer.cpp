// FILEPATH: src/core/SceneRenderer.cpp
#include "SceneRenderer.h"
#include <iostream>

void SceneRenderer::render(const Scene& scene) {
    for (const auto& object : scene.objects) {
        std::cout << "Rendering object: " << object.name << std::endl;
    }

    for (const auto& primitive : scene.primitives) {
        std::cout << "Rendering primitive of type: " << primitive.type << std::endl;
    }
}
