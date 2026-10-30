// FILEPATH: engine/src/core/Scene.cpp
#include "delirium/core/Scene.h"
#include "delirium/core/GameObject.h"
#include "delirium/psf/Reconstructor.h"
#include <memory>
#include <string>

namespace DeliRium {

GameObject* Scene::Create(const std::string& name,
                          const ReconstructionRequest& req) {
    static FixedReconstructor maker;
    auto go = std::make_unique<GameObject>(name, maker, req);
    GameObject* raw = go.get();
    objects.push_back(std::move(go));
    return raw;
}

void Scene::Update(float dt) {
    for (auto& o : objects) {
        if (o) o->Update(dt);
    }
}

void Scene::FixedUpdate(float fdt) {
    for (auto& o : objects) {
        if (o) o->FixedUpdate(fdt);
    }
}

} // namespace DeliRium
