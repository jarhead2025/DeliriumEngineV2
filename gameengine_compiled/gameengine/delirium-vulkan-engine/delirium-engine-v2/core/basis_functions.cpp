#include "basis_functions.h"
#include <cmath>

namespace delirium {

/* ── Corner-vanishing basis (8 terms) ──────────────────────────────
 *
 * b[0] = u(1-u)(1-v)    b[1] = u(1-u)v
 * b[2] = v(1-v)(1-u)    b[3] = v(1-v)u
 * b[4] = u(1-u)(2u-1)(1-v)
 * b[5] = u(1-u)(2u-1)v
 * b[6] = v(1-v)(2v-1)(1-u)
 * b[7] = v(1-v)(2v-1)u
 *
 * All terms vanish at every corner, so Evaluate(corner) == anchor_pos.
 * Shared anchors => crack-free seams by construction.
 * ────────────────────────────────────────────────────────────────── */
void BasisEvaluator::deform_basis(float u, float v, float b[8]) {
    const float bu = u * (1.0f - u),  bv = v * (1.0f - v);
    const float au = 2.0f * u - 1.0f, av = 2.0f * v - 1.0f;
    b[0] = bu * (1.0f - v);
    b[1] = bu * v;
    b[2] = bv * (1.0f - u);
    b[3] = bv * u;
    b[4] = bu * au * (1.0f - v);
    b[5] = bu * au * v;
    b[6] = bv * av * (1.0f - u);
    b[7] = bv * av * u;
}

/* ── evaluate() ──────────────────────────────────────────────────── */
SurfacePoint BasisEvaluator::evaluate(const BodyState& state,
                                       const BodyTopology& topology,
                                       PatchId patch_id,
                                       float u, float v) const {
    const auto& patch = topology.patch(patch_id);
    const auto& state_patch = state.patches[patch_id];
    const auto& coefs = state_patch.coefficients;

    // Bilinear blend of the four anchor positions
    const auto& a00 = state.anchors[patch.anchors[0]]; // (u=0,v=0)
    const auto& a10 = state.anchors[patch.anchors[1]]; // (u=1,v=0)
    const auto& a01 = state.anchors[patch.anchors[2]]; // (u=0,v=1)
    const auto& a11 = state.anchors[patch.anchors[3]]; // (u=1,v=1)

    Vec3 P = a00.position * (1.0f - u) * (1.0f - v) +
             a10.position * u * (1.0f - v) +
             a01.position * (1.0f - u) * v +
             a11.position * u * v;

    // Bilinear blend of anchor normals
    Vec3 Nd = normalize_vec3(
        a00.normal * (1.0f - u) * (1.0f - v) +
        a10.normal * u * (1.0f - v) +
        a01.normal * (1.0f - u) * v +
        a11.normal * u * v);

    // Corner-vanishing deformation
    float b[8];
    deform_basis(u, v, b);
    float h = 0.0f;
    for (int i = 0; i < 8; ++i)
        h += coefs[i] * b[i];

    P = P + Nd * h;

    // Surface normal via finite-difference gradient
    const float eps = 1e-4f;
    float bu[8], bv[8];
    deform_basis(u + eps, v, bu);
    deform_basis(u, v + eps, bv);

    float hu = 0.0f, hv = 0.0f;
    for (int i = 0; i < 8; ++i) {
        hu += coefs[i] * bu[i];
        hv += coefs[i] * bv[i];
    }

    // Analytical deformation gradient: (h(u+eps) - h(u)) / eps
    // h already computed above
    float dhdu = 0.0f, dhdv = 0.0f;
    for (int i = 0; i < 8; ++i) {
        dhdu += coefs[i] * (bu[i] - b[i]);
        dhdv += coefs[i] * (bv[i] - b[i]);
    }
    dhdu /= eps;
    dhdv /= eps;

    // Bilinear tangent directions
    Vec3 du_pos = (a10.position - a00.position) * (1.0f - v) +
                  (a11.position - a01.position) * v;
    Vec3 dv_pos = (a01.position - a00.position) * (1.0f - u) +
                  (a11.position - a10.position) * u;

    Vec3 T = du_pos + Nd * dhdu;
    Vec3 B = dv_pos + Nd * dhdv;
    Vec3 N = normalize_vec3(cross(T, B));

    return {P, N, patch.material, patch_id, u, v};
}

/* ── compute_seam_error() ──────────────────────────────────────────
 * Find shared edge by matching anchors, then sample along it.
 * ────────────────────────────────────────────────────────────────── */
float BasisEvaluator::compute_seam_error(const BodyState& state,
                                          const BodyTopology& topology,
                                          PatchId patch_a, PatchId patch_b) const {
    const auto& pa = topology.patch(patch_a);
    const auto& pb = topology.patch(patch_b);

    // Find shared anchors and their parametric positions
    struct Shared { int ai; float ui, vi; int bi; float uj, vj; };
    std::vector<Shared> shared;
    for (int ai = 0; ai < 4; ++ai) {
        float ui = (ai & 1) ? 1.0f : 0.0f;
        float vi = (ai & 2) ? 1.0f : 0.0f;
        for (int bi = 0; bi < 4; ++bi) {
            if (pa.anchors[ai] != pb.anchors[bi]) continue;
            float uj = (bi & 1) ? 1.0f : 0.0f;
            float vj = (bi & 2) ? 1.0f : 0.0f;
            shared.push_back({ai, ui, vi, bi, uj, vj});
        }
    }
    if (shared.size() < 2) return 1e30f; // no shared edge

    // Interpolate along shared edge
    float max_gap = 0;
    for (int s = 0; s <= 10; ++s) {
        float t = s / 10.0f;
        float u = shared[0].ui + t * (shared[1].ui - shared[0].ui);
        float v = shared[0].vi + t * (shared[1].vi - shared[0].vi);
        float uj = shared[0].uj + t * (shared[1].uj - shared[0].uj);
        float vj = shared[0].vj + t * (shared[1].vj - shared[0].vj);
        SurfacePoint sa = evaluate(state, topology, patch_a, u, v);
        SurfacePoint sb = evaluate(state, topology, patch_b, uj, vj);
        float gap = length(sa.position - sb.position);
        if (gap > max_gap) max_gap = gap;
    }
    return max_gap;
}

} // namespace delirium