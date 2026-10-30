#version 430 core

/* ── psf_splat.comp.glsl ──────────────────────────────────────────
 *
 * GPU compute shader: generates surfels from PSF patches.
 * Each work item evaluates one (patch, u, v) sample → writes Surfel to SSBO.
 *
 * SurfelVertex layout (44 bytes):
 *   pos(3) + normal(3) + color(4) + radius(1)
 *
 * ~115 FLOPs/sample, no image atomics, no depth buffer.
 * Hardware depth test handles z-ordering in the render pass.
 * ────────────────────────────────────────────────────────────────── */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

/* ── SSBO inputs ─────────────────────────────────────────────────── */
// CRITICAL: std430 pads vec3 to 16 bytes. CPU uploads float[na*3] (12B/anchor).
// Use flat float[] to match CPU layout exactly — no padding.
layout(std430, binding = 0) buffer AnchorsBuf {
    float anchors[];  // na * 3 floats (NO vec3 padding)
};

layout(std430, binding = 1) buffer AnchorsNormBuf {
    float anchor_normals[];  // na * 3 floats
};

layout(std430, binding = 2) buffer PatchesBuf {
    int patch_anchors[];  // np * 4 ints
};

layout(std430, binding = 3) buffer CoefBuf {
    float coefficients[];  // np * 8 floats
};

layout(std430, binding = 4) buffer MatBuf {
    float materials[];  // np * 4 floats (r,g,b,a)
};

layout(std430, binding = 5) buffer MetaBuf {
    uint meta[];  // [0]=anchor_count, [1]=patch_count, [2]=samples, [3]=total_surfels
};

/* ── SSBO output ─────────────────────────────────────────────────── */
layout(std430, binding = 6) buffer SurfelBuf {
    // SurfelVertex: pos(3) + normal(3) + color(4) + radius(1) = 11 floats
    float surfels[];  // total_surfels * 11 floats
};

layout(std430, binding = 7) buffer CounterBuf {
    uint surfel_count;  // written surfel count (set by CPU each frame)
};

/* ── Uniforms ────────────────────────────────────────────────────── */
uniform mat4 u_mvp;
uniform vec3 u_cam_pos;
uniform vec3 u_light_dir;
uniform uint u_width;
uniform uint u_height;
uniform float u_bbox_diag;  // bounding box diagonal for radius scaling

/* ── 8-term corner-vanishing basis (PROVEN from psf_splat.cl) ────── */
vec4 basis_ab(float u, float v) {
    float bu = u * (1.0 - u);
    float bv = v * (1.0 - v);
    return vec4(bu * (1.0 - v), bu * v, bv * (1.0 - u), bv * u);
}

vec4 basis_cd(float u, float v) {
    float bu = u * (1.0 - u);
    float bv = v * (1.0 - v);
    float au = 2.0 * u - 1.0;
    float av = 2.0 * v - 1.0;
    return vec4(bu * au * (1.0 - v), bu * au * v, bv * av * (1.0 - u), bv * av * u);
}

/* ── Evaluate patch surface at (u, v) ──────────────────────────────
 *
 * STANDARD BILINEAR (matches CPU basis_functions.cpp exactly):
 *   u along A0→A1 (segment/theta direction)
 *   v along A0→A2 (ring/phi direction)
 *
 * Corners: (0,0)→A0, (1,0)→A1, (0,1)→A2, (1,1)→A3
 * This MUST match the CPU evaluator — mismatch warps geometry.
 * ────────────────────────────────────────────────────────────────── */
// Flat array accessors — CPU uploads float[na*3], no std430 padding
vec3 anchor_pos(int i) { return vec3(anchors[i*3], anchors[i*3+1], anchors[i*3+2]); }
vec3 anchor_nrm(int i) { return vec3(anchor_normals[i*3], anchor_normals[i*3+1], anchor_normals[i*3+2]); }

vec3 eval_patch(int pid, float u, float v, out vec3 N) {
    int i0 = patch_anchors[pid * 4 + 0];
    int i1 = patch_anchors[pid * 4 + 1];
    int i2 = patch_anchors[pid * 4 + 2];
    int i3 = patch_anchors[pid * 4 + 3];

    vec3 A0 = anchor_pos(i0), A1 = anchor_pos(i1);
    vec3 A2 = anchor_pos(i2), A3 = anchor_pos(i3);

    // Bilinear: u along A0→A1 (segment), v along A0→A2 (ring)
    // Matches v2 CPU basis_functions.cpp:46-49
    vec3 pl = mix(A0, A1, u), pr = mix(A2, A3, u);
    vec3 P  = mix(pl, pr, v);

    // Bilinear normal blend — same u→A1, v→A2 axes
    vec3 nl = mix(anchor_nrm(i0), anchor_nrm(i1), u);
    vec3 nr = mix(anchor_nrm(i2), anchor_nrm(i3), u);
    vec3 Nd = normalize(mix(nl, nr, v));

    // 8-term corner-vanishing deformation (FMA chain, order matters)
    int ci = pid * 8;
    vec4 coef_ab = vec4(coefficients[ci+0], coefficients[ci+1], coefficients[ci+2], coefficients[ci+3]);
    vec4 coef_cd = vec4(coefficients[ci+4], coefficients[ci+5], coefficients[ci+6], coefficients[ci+7]);
    vec4 ab = basis_ab(u, v);
    vec4 cd = basis_cd(u, v);
    float h = dot(coef_ab, ab) + dot(coef_cd, cd);
    P += Nd * h;

    // Analytical normal via deformation gradient (FINITE DIFFERENCE)
    float eps = 1e-4;
    float h_u  = dot(coef_ab, basis_ab(u+eps, v)) + dot(coef_cd, basis_cd(u+eps, v));
    float h_v  = dot(coef_ab, basis_ab(u, v+eps)) + dot(coef_cd, basis_cd(u, v+eps));
    float dhdu = (h_u - h) / eps;  // derivative along u
    float dhdv = (h_v - h) / eps;  // derivative along v

    // Tangent vectors: du along u-axis (A1-A0 at v=0, A3-A2 at v=1)
    //                  dv along v-axis (A2-A0 at u=0, A3-A1 at u=1)
    vec3 du = mix(A1 - A0, A3 - A2, v);
    vec3 dv = mix(A2 - A0, A3 - A1, u);

    // Deformation gradient: add normal displacement to tangents
    vec3 du_d = du + Nd * dhdu;
    vec3 dv_d = dv + Nd * dhdv;

    // Cross product: cross(dv_d, du_d) → outward for standard quad winding
    N = normalize(cross(dv_d, du_d));

    // Guard: degenerate patches (pole collapse) → fall back to blended normal
    if (length(N) < 1e-6) N = Nd;
    return P;
}

/* ── Main ────────────────────────────────────────────────────────── */
void main() {
    uint total = gl_GlobalInvocationID.x;
    uint np = meta[1];
    uint ss = meta[2];
    uint total_surfels = meta[3];

    if (total >= total_surfels) return;

    // Sample grid: hit corners (u=0 and u=1) — matches original / (ss-1)
    uint ss2 = ss * ss;
    uint idx = total % (np * ss2);
    int pid = int(idx / ss2);
    float u = (float(idx % ss)) / float(ss - 1);
    float v = (float((idx / ss) % ss)) / float(ss - 1);

    vec3 N;
    vec3 P = eval_patch(pid, u, v, N);

    // Project to clip space
    vec4 clip = u_mvp * vec4(P, 1.0);
    vec4 ndc = clip / max(clip.w, 1e-6);

    // Culled surfels: write with radius=0 (invisible), no early return (avoids gaps)
    if (clip.w <= 0.0 || abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {
        uint si = total * 11;
        for (int i = 0; i < 11; ++i) surfels[si + i] = 0.0;
        return;
    }

    // NO back-face cull — matches proven psf_splat.cl (GPU depth test handles it)

    // Compute surfel radius in WORLD space
    float radius = u_bbox_diag / float(ss * 2);

    // Lambert lighting
    float lam = clamp(dot(N, -u_light_dir), 0.0, 1.0) * 0.85 + 0.15;
    vec3 albedo = vec3(materials[pid*4], materials[pid*4+1], materials[pid*4+2]);
    float fog = exp(-length(P - u_cam_pos) * 0.03);
    vec3 color = mix(vec3(0.08, 0.12, 0.18), albedo * lam, fog);

    // Write at invocation index — each work item writes exactly one surfel
    uint si = total * 11;
    surfels[si+0] = P.x;
    surfels[si+1] = P.y;
    surfels[si+2] = P.z;
    surfels[si+3] = N.x;
    surfels[si+4] = N.y;
    surfels[si+5] = N.z;
    surfels[si+6] = color.r;
    surfels[si+7] = color.g;
    surfels[si+8] = color.b;
    surfels[si+9] = 1.0;
    surfels[si+10] = radius;
}