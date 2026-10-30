// =============================================================================
// PhysicsWorld.h - The two-scale fixed step:
//   1. rigid integration (gravity, velocities, quaternion rotation)
//   2. analytic contacts vs static planes
//   3. impulse resolution (restitution + Coulomb friction) with
//      positional projection, and IMPACT -> MODAL coupling: each normal
//      impulse also excites the deformation modes at the contact (u,v)
//   4. modal step + writeback happens in GameObject::FixedUpdate
// =============================================================================
#pragma once
#include <vector>
#include "CollisionAnalytic.h"
#include "LocalFEM.h"
#include <memory>

namespace DeliRium {

class PhysicsWorld {
public:
    void Register(GameObject* obj)            { objects.push_back(obj);
                                                relax_debt.push_back(0); }
    void AddPlane(const Vec3& n, float d)     { planes.push_back({n, d}); }
    void SetGravity(const Vec3& g)            { gravity = g; }
    void Step(float fixed_dt);
    // Promote a region to a high-fidelity local FEM relaxation (the
    // expensive scale). One-shot: binds, assembles, relaxes `iters` steps,
    // writes back. Used by gameplay for precise local deformation.
    void PromoteToFEM(GameObject* obj, const Vec3& center, float radius,
                      int iters = 30);

    // Signal that an object was deformed/fractured and should relax toward
    // its continuous form over the next frames. The weld is then applied
    // PROGRESSIVELY in Step() -- a few relaxation passes per frame, paced
    // by the material category -- so a big deformation settles smoothly
    // without stalling any single frame. `amount` (>=0) adds relaxation
    // debt; larger means more total welding.
    void RequestWeld(GameObject* obj, float amount = 1.0f);

    // Tunable: how many weld passes to spend per frame (work budget).
    void SetWeldBudgetPerFrame(int passes)    { weld_budget = passes; }

    // Inspect remaining relaxation debt for an object (0 == settled).
    float WeldDebt(const GameObject* obj) const {
        for (size_t i = 0; i < objects.size(); ++i)
            if (objects[i] == obj) return relax_debt[i];
        return 0.0f;
    }

private:
    struct Plane { Vec3 n; float d; };
    std::vector<GameObject*> objects;
    std::vector<float>       relax_debt;   // per-object weld debt (passes)
    std::vector<Plane>       planes;
    Vec3 gravity{0, 0, -9.81f};
    int  weld_budget = 4;                  // weld passes per frame per object
};

} // namespace DeliRium
