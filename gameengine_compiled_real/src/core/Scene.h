// FILEPATH: src/core/Scene.h
#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "SpectralObject.h"
#include "SpectralPrimitive.h"

class Scene {
public:
    std::vector<SpectralObject> objects;
    std::vector<SpectralPrimitive> primitives;

    void addObject(const SpectralObject& object);
    void addPrimitive(const SpectralPrimitive& primitive);
};

#endif // SCENE_H
