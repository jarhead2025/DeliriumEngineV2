// FILEPATH: src/core/main.cpp

#include "GameObject.h"
#include "AnchorPhysics.h"
#include "ShaderArray.h"
#include "ShaderProgram.h"
#include "GPUManager.h"

int main() {
    GPUManager gpuManager;
    gpuManager.initializeOpenCL();
    gpuManager.initializeOpenGL();

    // Initialize game object and anchor physics
    GameObject obj;
    AnchorPhysics physics;
    initializeGameObject(&obj, "config.txt");
    initializeAnchorPhysics(&physics, 9.81f);

    // Update and render game object
    updateGameObject(&obj, 0.016f);
    ShaderProgram shader;
    renderGameObject(&obj, &shader);

    // Update anchor physics and render shader array
    updateAnchorPhysics(&physics, 0.016f);
    solveConstraints(&physics);
    ShaderArray array;
    initializeShaderArray(&array, 1024);
    updateShaderArray(&array, physics.data);
    renderShaderArray(&array, &shader);

    gpuManager.cleanupOpenCL();

    return 0;
}
