// FILEPATH: engine/src/core/GameObject.cpp
#include "delirium/core/GameObject.h"
#include "delirium/psf/Reconstructor.h"
#include <fstream>
#include <string>
#include <vector>

namespace DeliRium {

GameObject::GameObject(std::string name, IReconstructor& maker,
                       const ReconstructionRequest& req)
    : name(std::move(name)) {
    body = maker.Reconstruct(req);
}

bool GameObject::LoadModes(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    // Accept presence of file; full modal decode can be filled later.
    return true;
}

void GameObject::Poke(const Vec3& /*world_point*/, const Vec3& /*impulse*/) {}
void GameObject::PokeSurface(uint32_t, float, float, float) {}
void GameObject::DentLocally(const Vec3&, float, float) {}
void GameObject::Fracture(const Vec3&, float) {}
void GameObject::Morph(uint32_t, float) {}
void GameObject::RecalculateStructure() {}
void GameObject::Update(float) {}
void GameObject::FixedUpdate(float) {}

void NPC::SetPose(const std::vector<BoneRecord>& pose) { skeleton = pose; }
void NPC::Update(float dt) { GameObject::Update(dt); }

} // namespace DeliRium
