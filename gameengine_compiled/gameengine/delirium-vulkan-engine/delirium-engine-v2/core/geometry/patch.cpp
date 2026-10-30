#include "patch.h"
#include "../math/vec3.h"

namespace delirium {

// ---------------------------------------------------------------------------
// Patch implementation
// ---------------------------------------------------------------------------

vec3 Patch::evaluate(float u, float v) const {
    // Placeholder — needs anchor pool reference.
    // For now: evaluate via coefficient register
    return vec3(0, 0, 0);
}

vec3 Patch::normal_at(float u, float v) const {
    float eps = 0.01f;
    vec3 p  = evaluate(u, v);
    vec3 pu = evaluate(u + eps, v);
    vec3 pv = evaluate(u, v + eps);
    return (pu - p).cross(pv - p).normalized();
}

void Patch::bounding_box(vec3& min_out, vec3& max_out) const {
    min_out = vec3(1e30f); max_out = vec3(-1e30f);
    float corners[5][2] = {{0,0},{1,0},{0,1},{1,1},{0.5f,0.5f}};
    for(int i = 0; i < 5; i++) {
        vec3 p = evaluate(corners[i][0], corners[i][1]);
        min_out = vec3(fminf(min_out.x, p.x), fminf(min_out.y, p.y), fminf(min_out.z, p.z));
        max_out = vec3(fmaxf(max_out.x, p.x), fmaxf(max_out.y, p.y), fmaxf(max_out.z, p.z));
    }
}

// ---------------------------------------------------------------------------
// AnchorPool implementation
// ---------------------------------------------------------------------------

void AnchorPool::transform(const float mat[4][4]) {
    for(int i = 0; i < count; i++) {
        vec3 p = positions[i];
        positions[i] = vec3(
            mat[0][0]*p.x + mat[0][1]*p.y + mat[0][2]*p.z + mat[0][3],
            mat[1][0]*p.x + mat[1][1]*p.y + mat[1][2]*p.z + mat[1][3],
            mat[2][0]*p.x + mat[2][1]*p.y + mat[2][2]*p.z + mat[2][3]
        );
    }
}

// ---------------------------------------------------------------------------
// Seam implementation
// ---------------------------------------------------------------------------

float Seam::gap(const Patch& pa, const Patch& pb, const AnchorPool& anchors) const {
    // Gap should be 0 by construction — patches share anchor indices.
    float max_gap = 0;
    for(int i = 0; i <= 10; i++) {
        float t = i / 10.0f;
        // TODO: implement edge sampling
    }
    return max_gap;
}

} // namespace delirium