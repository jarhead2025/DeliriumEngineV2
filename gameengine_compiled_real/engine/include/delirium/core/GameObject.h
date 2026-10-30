// =============================================================================
// GameObject.h - Pure OOP reconstruction: an object IS its equations.
// No mesh member exists anywhere in this class. Gameplay code talks to
// shape and physics through the same handles the file format stores:
// patch coefficients, K amplitudes, seam curves, anchors.
// =============================================================================
#pragma once
#include <memory>
#include <string>
#include "Types.h"
#include "../psf/PSFBody.h"
#include "../psf/Reconstructor.h"
#include "../physics/ModalDynamics.h"
#include "../physics/Material.h"

namespace DeliRium {

struct Transform {
    Vec3 position{}; Quat rotation{0,0,0,1}; Vec3 scale{1,1,1};
};

// rigid-body state integrated by PhysicsWorld (v1: uniform mass,
// solid-sphere inertia approximation from the bbox -- documented)
struct RigidState {
    Vec3  velocity{}, omega{};
    float mass = 1.0f, inv_mass = 1.0f;
    float inertia = 0.4f, inv_inertia = 2.5f;   // scalar (sphere approx)
    float restitution = 0.4f, friction = 0.5f;
};

class GameObject {
public:
    GameObject(std::string name, IReconstructor& maker,
               const ReconstructionRequest& req);                // TODO body = maker.Reconstruct

    // --- high-level live manipulation (gameplay API) ----------------------
    void Poke(const Vec3& world_point, const Vec3& impulse);     // TODO needs ClosestPoint
    void PokeSurface(uint32_t s, float u, float v, float j);     // modal impulse, direct
    bool LoadModes(const std::string& path);                     // .modes from psf_modes.py
    void DentLocally(const Vec3& p, float radius, float depth);  // TODO RefineWhere + RefitLocal
    void Fracture(const Vec3& p, float energy,
                  const Vec3& cam_forward = {0,0,0});           // TODO split seams -> new bodies
    void Morph(uint32_t pattern_id, float k);                    // TODO direct K edit
    void RecalculateStructure();                                 // TODO global refit pass
                                                                 //      (the future AI entry point)
    virtual void Update(float dt);                               // TODO
    virtual void FixedUpdate(float fdt);                         // TODO modal step

    PSFBody&   Body()      { return *body; }
    const PSFBody& Body() const { return *body; }
    bool       HasBody() const { return body != nullptr; }
    PSFBody*   Fragment()  { return fragments.get(); }
    Transform& GetTransform() { return transform; }
    RigidState& Rigid()    { return rigid; }
    const RigidState& Rigid() const { return rigid; }
    ModalDynamics& Dynamics() { return dynamics; }
    Material&       Mat()      { return material; }
    const Material& Mat() const { return material; }
    const std::string name;

protected:
    Transform                      transform;
    RigidState                     rigid;
    std::unique_ptr<PSFBody>       body;
    std::unique_ptr<PSFBody>       fragments; // Fracture output
    ModalDynamics                  dynamics;
    Material                       material;
};

// NPC = GameObject + skeleton: bones displace anchors (delta[3] +
// anim_weight already in the record), equations ride on top untouched.
class NPC : public GameObject {
public:
    using GameObject::GameObject;
    void SetPose(const std::vector<BoneRecord>& pose);           // TODO anchors follow bones
    void Update(float dt) override;                              // TODO anim + secondary modal motion
private:
    std::vector<BoneRecord> skeleton;
};

} // namespace DeliRium
