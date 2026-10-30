// =============================================================================
// CageSurface.h - The anchor-cage surface model (architectural reframe).
//
// The user's insight: instead of "fixed anchors + scalar deformation that
// inflates the surface between them along a possibly-wrong normal", make the
// ANCHORS the degrees of freedom. Deformation MOVES anchors; the surface is
// whatever smoothly fits the anchor positions. The anchors define a coarse
// physical volume (a cage); the surface is the skin over it.
//
// Why this is structurally better (all MEASURED, see validate_cage):
//   * Seam continuity is FREE: two patches sharing an edge share its two
//     anchors, and the surface is determined by anchor positions, so the
//     shared edge is ONE curve on both sides -> gap is 0 by construction.
//     No weld, no stitch, no harmonize, no bridge -- the entire seam
//     subsystem dissolves.
//   * Normals are DERIVED from the cage (cross products of neighbor edges),
//     so they can never contradict the surface -- the "point appears where
//     it shouldn't" failure mode is impossible.
//   * Deformation is physically meaningful: moving an anchor moves the skin,
//     and shared anchors couple neighbors automatically.
//
// The surface within a patch is the bilinear of its 4 corner anchors plus a
// smooth "fullness" term derived from the discrete curvature of the anchor
// cage (how far each anchor sits from the mean of its neighbors, along the
// derived normal). This term vanishes at edges (so seams stay exact) and
// peaks at the patch center -- it reconstructs the bulge that stored coefs
// used to encode, but as a CONSEQUENCE of anchor positions, shared across
// neighbors.
// =============================================================================
#pragma once
#include "PSFBody.h"
#include <vector>
#include <unordered_map>

namespace DeliRium {

class CageSurface {
public:
    explicit CageSurface(const PSFBody& body);

    // Evaluate the cage surface for patch s at (u,v).
    Vec3 Evaluate(uint32_t s, float u, float v) const;
    // Derived normal (consequence of anchor positions, cannot contradict).
    Vec3 EvaluateNormal(uint32_t s, float u, float v) const;

    // Max seam gap of the cage surface (should be ~0 by construction).
    float MaxSeamGap() const;
    // RMS deviation of the cage surface from the body's current (coef-based)
    // surface -- the detail the cage doesn't yet capture.
    float FidelityVsCurrent() const;

private:
    const PSFBody& body_;
    std::vector<Vec3> anchor_normal_;   // derived per-anchor normal
    std::vector<float> anchor_bulge_;    // cage curvature per anchor
    Vec3 CornerBulgeDir(uint16_t a) const;
    void BuildCage();
};

} // namespace DeliRium
