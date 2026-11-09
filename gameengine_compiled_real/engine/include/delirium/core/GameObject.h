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
#include "Transform.h"
#include "../psf/PSFBody.h"
#include "../psf/Reconstructor.h"
#include "../physics/ModalDynamics.h"
#include "../physics/Material.h"

namespace DeliRium {

// The rigid world transform lives in core/Transform.h (Mat4 + ToMatrix +
// InversePoint). It is deliberately OUTSIDE the frozen 64-byte records so
// moving an object never re-quantizes anchors -- the renderer applies it as
// a GPU model matrix.

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

    // Construct from a body that was BUILT OUTSIDE the engine (the AI
    // path: an element tree parsed from a model response, baked by
    // ElementSystem::BakeBody; or a .psf loaded from disk).
    //
    // WHY A SEPARATE CONSTRUCTOR: the reconstructing constructor always calls
    // maker.Reconstruct(req), and for these bodies the request is empty --
    // so the reconstructor fails, the constructor prints a spurious
    // "reconstruction failed ()" on stderr, and the body is then REPLACED
    // by AdoptBody. The user saw two bogus errors per object on every
    // launch. This path performs NO reconstruction at all.
    GameObject(std::string name, std::unique_ptr<PSFBody> body,
               Material material = Material{});

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
    // Adopt a body built outside the engine (the AI path: an element tree
    // parsed from a model response, baked by ElementSystem::BakeBody).
    // The GameObject constructor always reconstructs through an
    // IReconstructor, so this is how an externally-built body enters.
    void AdoptBody(std::unique_ptr<PSFBody> b) { body = std::move(b); }
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
