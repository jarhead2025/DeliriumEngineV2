#pragma once
#include "../math/vec3.h"
#include "../math/coefficient.h"

namespace delirium {

/// A patch is a quadrilateral surface region with shared anchors at boundaries.
/// Anchors are shared between adjacent patches → gap 0.0000% by construction.
/// Each patch stores its own coefficients but shares anchor positions.
struct Patch {
    // 4 anchor indices into the shared AnchorPool
    int anchor_ids[4];

    // Coefficient register for this patch's surface
    CoefficientRegister coeffs;

    // Patch ID for debugging
    int id = -1;

    // Evaluate surface at (u,v) in [0,1] parametric space
    vec3 evaluate(float u, float v) const;

    /// Evaluate normal at (u,v) via finite differences
    vec3 normal_at(float u, float v) const;

    /// Bounding box in world space
    void bounding_box(vec3& min, vec3& max) const;
};

/// Shared anchor pool — ensures patches merge seamlessly.
/// Two patches sharing an edge share the same 2 anchor indices → one curve.
struct AnchorPool {
    static constexpr int MAX_ANCHORS = 8192;

    vec3 positions[MAX_ANCHORS];
    int count = 0;

    /// Add anchor, return index
    int add_anchor(const vec3& pos) {
        if(count < MAX_ANCHORS) {
            positions[count] = pos;
            return count++;
        }
        return -1;
    }

    /// Get anchor position by index
    const vec3& get(int idx) const { return positions[idx]; }

    /// Transform all anchors by matrix (for body-level transforms)
    void transform(const float mat[4][4]);
};

/// Seam between two patches — enforces boundary continuity.
/// Gap is 0 by construction since patches share anchor indices.
struct Seam {
    int patch_a;    // First patch
    int patch_b;    // Second patch
    int edge_a;     // Edge index in patch_a (0-3)
    int edge_b;     // Edge index in patch_b (0-3)

    /// Verify seam continuity — should always be 0
    float gap(const Patch& pa, const Patch& pb, const AnchorPool& anchors) const;
};

} // namespace delirium