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
