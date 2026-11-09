// =============================================================================
// Inertia.h - mass distribution, centre of mass, and the inertia tensor of a
// skinned body.
//
// WHY THIS EXISTS (Josias, 2026-11-08): "object reconstruction is encoded
// based on line segment (bones) or points in space (articulations) ... so
// we can achieve faithful physical reparametrization live, as well as
// calculating rotational and directional inertias, using conservation of
// mass and energy".
//
// The skeleton makes this possible. A surface-only body says nothing about
// where the mass IS; a body with bones says "this much mass rides on this
// rigid segment", and the anchors say WHERE that segment's skin sits. So
// the mass distribution is: each anchor carries a share of its bone's
// mass, and the inertia tensor is the mass-weighted sum of r x (r x .).
//
// THE INVARIANTS THIS MODULE GUARANTEES:
//
//   * TRANSLATION INVARIANCE. The inertia tensor is about the CENTRE OF
//     MASS, so translating the body does not change it. (The tensor about
//     the origin does change, by the parallel-axis theorem -- that is
//     physics, not a bug.)
//   * ROTATION COVARIANCE. Rotating the body rotates the tensor with it.
//   * CONSERVATION OF MASS. The total mass is exactly the sum of the
//     bones' masses; nothing is created or lost by the distribution.
//   * DIRECTIONAL INERTIA. For any axis u, the inertia about that axis
//     is u . I . u -- the "directional" form the user asked for, and the
//     rotation-about-an-axis question is answered directly from it.
//
// WHY A BONE-PALETTE MASS, NOT A MESH MASS: the body's surface is a
// deformed graph, and its area depends on the LOD. The BONES do not: a
// bone's mass is a property of the mechanism, not of how finely it is
// tessellated. So the mass distribution is per-bone (authored), spread
// over that bone's anchors (geometric). Coarsening the surface changes
// how finely the mass is spread, but not how much there is -- which is
// exactly what "complete in representation, coarse for speed" requires.
// =============================================================================
#pragma once
#include <string>
#include <vector>
#include "delirium/core/Types.h"
#include "delirium/psf/PSFBody.h"

namespace DeliRium {

// The mass properties of a body, about its centre of mass.
struct MassProperties {
    float total_mass = 0.f;        // the body's mass (sum of the bones')
    Vec3  centre_of_mass{0.f,0.f,0.f};

    // The inertia TENSOR about the centre of mass. A symmetric 3x3 matrix
    // stored as [Ixx Iyy Izz Ixy Ixz Iyz] -- the off-diagonals appear
    // twice in the matrix, so they are stored once.
    float Ixx = 0.f, Iyy = 0.f, Izz = 0.f;
    float Ixy = 0.f, Ixz = 0.f, Iyz = 0.f;

    // The directional inertia about an arbitrary unit axis: u . I . u.
    // This is the "rotational inertia about this axis" question, answered
    // directly from the tensor.
    float InertiaAbout(const Vec3& unit_axis) const;

    // The principal moments -- the eigenvalues of the tensor -- and the
    // axes they belong to. These are the numbers that characterise how
    // "top-heavy" or "long and thin" a body is: the largest moment is the
    // axis the body is LONGEST along.
    void PrincipalMoments(Vec3 moments[3], Vec3 axes[3]) const;

    // A human-readable summary for the console.
    std::string Describe() const;
};

// Compute the mass properties of a body whose anchors are bound to bones.
//
// WHY THE BONES ARE A SEPARATE ARGUMENT: the bones live in the
// RECONSTRUCTION MODEL (ModelOutput::bones), not in the PSFBody. A PSFBody
// carries the surface -- anchors, patches, seams -- and nothing else. The
// skeleton is the part of the reconstruction that says WHAT MOVES, and the
// body is the part that says WHAT IT LOOKS LIKE. Inertia needs both: the
// mass (from the bones) and the mass's LOCATION (from the anchors).
//
// `bone_mass` maps a bone index to that bone's mass. A bone with no entry
// contributes NO mass (the caller decides the density policy). Anchors
// with no bone (bone_index == 0xFFFFFFFF) also contribute no mass.
//
// The mass of a bone is spread EQUALLY over that bone's anchors, so the
// result does not depend on the tessellation density.
//
// `fallback_density`: if `bone_mass` is empty, every bound anchor gets
// this mass instead (a uniform-density body). This lets a caller ask for
// "the inertia of this shape" without authoring a skeleton.
//
// Returns false with `why` when the body has no mass at all (no bones, no
// bound anchors, or every mass is zero) -- an inertia tensor of zero is
// not a useful answer, it is a missing one.
bool ComputeMassProperties(const PSFBody& body,
                           const std::vector<float>& bone_mass,
                           float fallback_density,
                           MassProperties& out, std::string& why);

// The same, but taking the skeleton itself. When `bone_mass` is EMPTY, the
// mass is read from each BoneRecord's reserved bytes (the baker stored the
// authored, normalised mass there) -- so a caller that has a reconstruction
// model needs no separate mass table. A non-empty `bone_mass` still wins,
// because an explicit table is more specific than a quantised record.
bool ComputeMassProperties(const PSFBody& body,
                           const std::vector<BoneRecord>& bones,
                           const std::vector<float>& bone_mass,
                           float fallback_density,
                           MassProperties& out, std::string& why);

} // namespace DeliRium
