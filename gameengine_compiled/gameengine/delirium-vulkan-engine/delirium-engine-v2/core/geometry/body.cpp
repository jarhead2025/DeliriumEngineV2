#include "body.h"
#include "../math/basis.h"
#include <cstring>
#include <cmath>

namespace delirium {

// ---------------------------------------------------------------------------
// Body implementation
// ---------------------------------------------------------------------------

int Body::add_patch(const Patch& p) {
    if(patch_count >= 256) return -1;
    patches[patch_count] = p;
    patches[patch_count].id = patch_count;
    return patch_count++;
}

void Body::add_seam(const Seam& s) {
    if(seam_count >= 512) return;
    seams[seam_count++] = s;
}

void Body::update_render_register() {
    render_coeffs.reset();
    for(int i = 0; i < patch_count; i++) {
        const auto& patch = patches[i];
        for(int j = 0; j < patch.coeffs.count; j++) {
            Coefficient c = patch.coeffs.coeffs[j];
            // Transform coefficient center by body transform
            c.cx = transform.transformPoint(vec3(c.cx, c.cy, c.cz)).x;
            c.cy = transform.transformPoint(vec3(c.cx, c.cy, c.cz)).y;
            c.cz = transform.transformPoint(vec3(c.cx, c.cy, c.cz)).z;
            render_coeffs.add_coefficient(c);
        }
    }
    render_coeffs.mark_all_dirty();
}

void Body::animate_bone(int bone_id, const mat4& bone_transform) {
    if(bone_id < 0 || bone_id >= bone_count) return;
    bone_transforms[bone_id] = bone_transform;

    // Transform all coefficients bound to this bone
    for(int i = 0; i < render_coeffs.count; i++) {
        auto& c = render_coeffs.coeffs[i];
        if(c.bone_id == bone_id) {
            c.transform(bone_transform.m);
            render_coeffs.mark_dirty(i);
        }
    }
}

void Body::world_bounding_box(vec3& min_out, vec3& max_out) const {
    min_out = vec3(1e30f); max_out = vec3(-1e30f);
    for(int i = 0; i < render_coeffs.count; i++) {
        const auto& c = render_coeffs.coeffs[i];
        vec3 corner = vec3(c.cx - c.sx, c.cy - c.sy, c.cz - c.sz);
        vec3 corner2 = vec3(c.cx + c.sx, c.cy + c.sy, c.cz + c.sz);
        min_out = vec3(fminf(min_out.x, corner.x), fminf(min_out.y, corner.y), fminf(min_out.z, corner.z));
        max_out = vec3(fmaxf(max_out.x, corner2.x), fmaxf(max_out.y, corner2.y), fmaxf(max_out.z, corner2.z));
    }
}

float Body::max_seam_gap() const {
    float max_gap = 0;
    for(int i = 0; i < seam_count; i++) {
        if(seams[i].patch_a >= 0 && seams[i].patch_a < patch_count &&
           seams[i].patch_b >= 0 && seams[i].patch_b < patch_count) {
            float g = seams[i].gap(patches[seams[i].patch_a], patches[seams[i].patch_b], anchors);
            max_gap = fmaxf(max_gap, g);
        }
    }
    return max_gap;
}

} // namespace delirium