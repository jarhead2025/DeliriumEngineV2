// FILEPATH: engine/src/core/GameObject.cpp
// Auto-generated M1 link body — matches GameObject.h (no out-of-line dtor: header does not declare ~GameObject).
// Virtual Update/FixedUpdate are the vtable key functions.
#include "delirium/core/GameObject.h"
#include "delirium/psf/Reconstructor.h"
#include <string>
#include <vector>

namespace DeliRium {

GameObject::GameObject(std::string name, IReconstructor& maker, const ReconstructionRequest& req)
    : name(std::move(name)) {
    (void)maker;
    (void)req;
    // body = maker.Reconstruct(req) when full engine path is wired
}

bool GameObject::LoadModes(const std::string& /*path*/) {
    return true;
}

void GameObject::Poke(const Vec3& /*world_point*/, const Vec3& /*impulse*/) { (void)0; }
void GameObject::PokeSurface(uint32_t /*s*/, float /*u*/, float /*v*/, float /*j*/) { (void)0; }
void GameObject::DentLocally(const Vec3& /*p*/, float /*radius*/, float /*depth*/) { (void)0; }
void GameObject::Fracture(const Vec3& /*p*/, float /*energy*/) { (void)0; }
void GameObject::Morph(uint32_t /*pattern_id*/, float /*k*/) { (void)0; }
void GameObject::RecalculateStructure() { (void)0; }

// Virtual key functions — anchors vtable (do NOT define ~GameObject unless header declares it)
void GameObject::Update(float /*dt*/) { (void)0; }
void GameObject::FixedUpdate(float /*fdt*/) { (void)0; }

void NPC::SetPose(const std::vector<BoneRecord>& pose) {
    skeleton = pose;
}
void NPC::Update(float dt) {
    GameObject::Update(dt);
}

} // namespace DeliRium
