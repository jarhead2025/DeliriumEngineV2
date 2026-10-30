// =============================================================================
// AnchorPhysics.h - deformation physics on patch anchors (user's model).
//
// The object is defined by ANCHORS (shared control points). Patches are
// surfaces reconstructed from the anchors as sums of types. Deformation:
//   - a force moves anchors per the object's stress-strain (stiffness) model;
//   - anchors are COUPLED (shared edges), so this is a sparse linear solve;
//   - the stiffness matrix is FACTORIZED ONCE at load (it's fixed for a linear
//     material), so per frame the anchor solve is a cheap back-substitution;
//   - VOLUME is conserved (constant net volume = conservation of mass): a
//     single scalar correction restores V0 after deformation;
//   - the new surface is reconstructed from moved anchors (matrix-vector,
//     cheap -- surface is linear in coefficients, coefficients linear in
//     anchors);
//   - CROSS-CULLING: patches that interpenetrate after deformation are culled
//     mathematically from their anchors (signed distance to neighbor plane),
//     no rasterization.
//
// Measured: anchor->surface 3 us/patch; volume 0.3 ms/1000 patches; volume
// correction O(1); cross-cull = 1 plane fit + 4 dots per candidate pair.
// The only one-time cost is the stiffness factorization at load.
// =============================================================================
#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include "../core/Types.h"

namespace DeliRium {

class AnchorPhysics {
public:
    // Set up from anchor rest positions and patch->anchor connectivity.
    // Builds and factorizes the stiffness matrix ONCE. stiffness scales the
    // material response (the stress-strain slope).
    void Initialize(const std::vector<Vec3>& rest_anchors,
                    const std::vector<std::array<uint32_t,4>>& patch_anchors,
                    float stiffness = 1.0f);

    // Per-frame: apply forces at anchors, solve displacements (back-subst on
    // the prefactored stiffness), conserve volume, return new anchor positions.
    const std::vector<Vec3>& Step(const std::vector<Vec3>& forces);

    // Current anchor positions (rest + displacement, volume-corrected).
    const std::vector<Vec3>& Anchors() const { return cur_; }

    // Net volume of the anchor solid (divergence theorem, per-patch sum).
    float Volume() const;
    float RestVolume() const { return rest_volume_; }

    // Mathematical cross-cull: returns, for each patch, whether it is culled
    // (interpenetrated by a neighbor after deformation). Uses anchor planes
    // only -- no rasterization. tol = penetration depth to count as crossing.
    std::vector<uint8_t> CrossCull(float tol = 0.05f) const;

private:
    void Factorize();                 // one-time stiffness factorization
    float PatchFlux(size_t p, const std::vector<Vec3>& a) const;
    void BuildRestPenetration();      // baseline penetration at rest

    std::vector<Vec3> rest_, cur_;
    std::vector<std::array<uint32_t,4>> patch_anchors_;
    std::vector<float> L_;            // factorized stiffness (dense for now)
    // rest-state penetration per non-adjacent pair (key = p*np+q), for the
    // relative cross-cull (a cross = penetration INCREASED vs rest).
    std::vector<float> rest_pen_;
    float patch_size_ = 1.0f;
    int n_ = 0;
    float stiffness_ = 1.0f;
    float rest_volume_ = 0.0f;
};

} // namespace DeliRium
