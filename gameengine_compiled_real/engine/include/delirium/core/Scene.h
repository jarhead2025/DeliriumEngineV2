#pragma once
#include <memory>
#include <string>
#include <vector>
#include "GameObject.h"
#include "../render/Camera.h"

namespace DeliRium {

class Scene {
public:
    explicit Scene(std::string n) : sceneName(std::move(n)) {}
    GameObject* Create(const std::string& name,
                       const ReconstructionRequest& req);  // TODO uses engine reconstructor
    // Create from a body built OUTSIDE the engine (the AI path: an element
    // tree parsed from a model response, baked by ElementSystem::BakeBody).
    // The body is moved in and the scene owns it.
    GameObject* Create(const std::string& name, std::unique_ptr<PSFBody> body);
    void Update(float dt);                                 // TODO
    void FixedUpdate(float fdt);                           // TODO
    Camera& GetCamera() { return camera; }
    const std::vector<std::unique_ptr<GameObject>>& Objects() const { return objects; }
    const std::string sceneName;
private:
    std::vector<std::unique_ptr<GameObject>> objects;
    Camera camera;
};

} // namespace DeliRium
