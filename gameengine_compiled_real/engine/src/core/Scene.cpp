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

GameObject* Scene::Create(const std::string& name, std::unique_ptr<PSFBody> body) {
    // The AI path: the body was built OUTSIDE the engine (an element tree
    // parsed from a model response, baked by ElementSystem::BakeBody; or a
    // .psf loaded from disk). The GameObject takes it directly -- NO
    // reconstruction is attempted, so there is no spurious
    // "reconstruction failed ()" on stderr for a body that is perfectly
    // valid.
    if (!body) return nullptr;
    auto go = std::make_unique<GameObject>(name, std::move(body));
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
