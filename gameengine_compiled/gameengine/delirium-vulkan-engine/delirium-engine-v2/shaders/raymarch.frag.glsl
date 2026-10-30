#version 430 core

precision highp float;

uniform vec3  u_camera_pos;
uniform mat4  u_view_inv;
uniform mat4  u_proj_inv;
uniform vec2  u_resolution;

uniform vec3  u_anchors[256];
uniform int   u_anchor_count;

uniform ivec4 u_patches[128];
uniform int   u_patch_count;

// 8 coefficients per patch = 2×vec4
uniform vec4  u_patch_coef[256];

uniform vec3  u_patch_colors[128];

out vec4 frag_color;

const int   MAX_STEPS  = 16;
const float STEP_SIZE  = 0.25;
const float EPSILON    = 0.15;
const float MAX_DIST   = 8.0;
const int   SAMPLES    = 1;  // 1×1 = 1 sample/patch — perf

// ─── 8-term corner-vanishing basis ────────────────────────────────
// Returns 8 floats packed in 2 vec4s
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

// ─── Bilinear patch evaluation (PROVEN kernel ordering) ───────────
// u along A0→A3, v along A0→A1 (matches psf_splat.cl:194-198)
vec3 bilinear(vec3 A, vec3 B, vec3 C, vec3 D, float u, float v) {
    vec3 pl = mix(A, D, u), pr = mix(B, C, u);
    return mix(pl, pr, v);
}

vec3 bilinear_normal(vec3 A, vec3 B, vec3 C, vec3 D, float u, float v) {
    float e = 0.005;
    return normalize(cross(
        bilinear(A,B,C,D,u+e,v) - bilinear(A,B,C,D,u-e,v),
        bilinear(A,B,C,D,u,v+e) - bilinear(A,B,C,D,u,v-e)));
}

// Full evaluation: bilinear + 8-term deformation
vec3 eval_point(vec3 A, vec3 B, vec3 C, vec3 D, vec4 c0, vec4 c1, float u, float v) {
    vec3 P = bilinear(A, B, C, D, u, v);
    vec3 Nd = bilinear_normal(A, B, C, D, u, v);
    vec4 b0 = basis_ab(u, v);
    vec4 b1 = basis_cd(u, v);
    float h = dot(c0, b0) + dot(c1, b1);
    return P + Nd * h;
}

// ─── Patch distance (SAMPLES×SAMPLES grid) ────────────────────────
float patch_dist(vec3 p, vec3 A, vec3 B, vec3 C, vec3 D, vec4 c0, vec4 c1,
                 out vec3 closest, out vec2 cuv) {
    float md = 1e10;
    closest = vec3(0); cuv = vec2(0.5);
    for (int i = 0; i < SAMPLES; ++i)
        for (int j = 0; j < SAMPLES; ++j) {
            float u = (i + 0.5) / SAMPLES;
            float v = (j + 0.5) / SAMPLES;
            vec3 sp = eval_point(A, B, C, D, c0, c1, u, v);
            float d = length(p - sp);
            if (d < md) { md = d; closest = sp; cuv = vec2(u, v); }
        }
    return md;
}

// ─── Bounding sphere per patch ────────────────────────────────────
float patch_bbox_dist(vec3 p, vec3 A, vec3 B, vec3 C, vec3 D) {
    vec3 c = (A + B + C + D) * 0.25;
    float r = max(max(length(B-A), length(D-C)), max(length(C-A), length(D-B))) * 0.5;
    return length(p - c) - r;
}

// ─── Scene SDF with bbox culling ──────────────────────────────────
float scene_sdf(vec3 p, out vec3 hit, out int hpatch, out vec2 huv) {
    float md = 1e10;
    hit = vec3(0); hpatch = -1; huv = vec2(0);

    for (int pi = 0; pi < u_patch_count; ++pi) {
        ivec4 idx = u_patches[pi];
        vec3 A = u_anchors[idx.x];
        vec3 B = u_anchors[idx.y];
        vec3 C = u_anchors[idx.z];
        vec3 D = u_anchors[idx.w];

        if (patch_bbox_dist(p, A, B, C, D) > md + 0.5) continue;

        int ci = pi * 2;
        vec4 c0 = u_patch_coef[ci];
        vec4 c1 = u_patch_coef[ci + 1];

        vec3 cl; vec2 cu;
        float d = patch_dist(p, A, B, C, D, c0, c1, cl, cu);
        if (d < md) { md = d; hit = cl; hpatch = pi; huv = cu; }
    }
    return md;
}

// ─── Normal ───────────────────────────────────────────────────────
vec3 compute_normal(vec2 uv, int pid) {
    if (pid < 0 || pid >= u_patch_count) return vec3(0, 1, 0);
    ivec4 idx = u_patches[pid];
    return bilinear_normal(u_anchors[idx.x], u_anchors[idx.y],
                           u_anchors[idx.z], u_anchors[idx.w], uv.x, uv.y);
}

// ─── Ray generation ───────────────────────────────────────────────
vec3 generate_ray(vec2 pixel) {
    vec2 ndc = (pixel / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    vec4 clip = vec4(ndc, -1.0, 1.0);  // standard OpenGL: -1 → ray from origin
    vec3 dir = normalize((u_proj_inv * clip).xyz);
    mat3 vr = mat3(u_view_inv);
    return vr * dir;
}

// ─── Lighting ─────────────────────────────────────────────────────
vec3 shade(vec3 p, vec3 n, int pid) {
    vec3 ld = normalize(vec3(1.0, 2.0, 3.0));
    vec3 vd = normalize(u_camera_pos - p);
    vec3 hd = normalize(ld + vd);
    vec3 albedo = (pid >= 0 && pid < u_patch_count) ? u_patch_colors[pid] : vec3(0.8);
    return albedo * 0.2 + albedo * max(dot(n, ld), 0.0) * 0.7 +
           vec3(1.0) * pow(max(dot(n, hd), 0.0), 32.0) * 0.3;
}

// ─── Main ─────────────────────────────────────────────────────────
void main() {
    vec2 pixel = gl_FragCoord.xy;
    vec3 rd = generate_ray(pixel);
    vec3 rp = u_camera_pos;

    float td = 0.0;
    vec3  chit = vec3(0);
    int   cpatch = -1;
    vec2  cuv = vec2(0);
    float best = 1e10;

    for (int s = 0; s < MAX_STEPS; ++s) {
        vec3 hp; int hp2; vec2 hu;
        float d = scene_sdf(rp, hp, hp2, hu);

        if (d < best) { best = d; chit = hp; cpatch = hp2; cuv = hu; }
        if (d < EPSILON) break;
        if (td > MAX_DIST) break;

        rp += rd * STEP_SIZE;
        td += STEP_SIZE;
    }

    if (best < EPSILON * 2.0) {
        vec3 n = compute_normal(cuv, cpatch);
        vec3 col = shade(chit, n, cpatch);
        float fog = exp(-td * 0.03);
        col = mix(vec3(0.08, 0.12, 0.18), col, fog);
        frag_color = vec4(col, 1.0);
    } else {
        float t = rd.y * 0.5 + 0.5;
        vec3 sky = mix(vec3(0.05, 0.08, 0.15), vec3(0.2, 0.3, 0.5), t);
        frag_color = vec4(sky, 1.0);
    }
}