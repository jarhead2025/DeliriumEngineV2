#pragma once
#include "patch.h"
#include "../math/basis.h"

namespace delirium {

/// A body is a collection of patches + topology + material regions.
/// This is the fundamental scene object — a creature, robot, or prop.
/// Physics operates on body coefficients; render reads body coefficients.
struct Body {
    // Patch collection
    Patch patches[256];
    int patch_count = 0;

    // Shared anchor pool
    AnchorPool anchors;

    // Seam definitions
    Seam seams[512];
    int seam_count = 0;

    // Combined coefficient register (all patches merged for rendering)
    CoefficientRegister render_coeffs;

    // Body transform (world space)
    mat4 transform;

    // Bone/anchor transforms for animation
    mat4 bone_transforms[64];
    int bone_count = 0;

    // Body ID and name
    int id = -1;
    char name[64];

    Body() {
        transform = mat4::identity();
        memset(name, 0, sizeof(name));
    }

    /// Add a patch to this body
    int add_patch(const Patch& p);

    /// Add a seam between two patches
    void add_seam(const Seam& s);

    /// Update render coefficient register from all patches
    /// Called when patches are modified — merges into single register for GPU
    void update_render_register();

    /// Transform body by matrix
    void apply_transform(const mat4& t) { transform = t * transform; }

    /// Animate bone — transforms all coefficients bound to that bone
    void animate_bone(int bone_id, const mat4& bone_transform);

    /// Evaluate surface at world position
    float surface_at(float px, float py, float pz) const {
        return BasisFunction::evaluate_sum(render_coeffs, px, py, pz);
    }

    /// Query surface at world position (for ray marching)
    bool query_surface(float px, float py, float pz,
                       float& depth, vec3& normal, vec3& color, float& material) const {
        return BasisFunction::query_surface(render_coeffs, px, py, pz, depth, normal, color, material);
    }

    /// Bounding box in world space
    void world_bounding_box(vec3& min, vec3& max) const;

    /// Verify all seams are continuous (gap < threshold)
    float max_seam_gap() const;
};

} // namespace delirium