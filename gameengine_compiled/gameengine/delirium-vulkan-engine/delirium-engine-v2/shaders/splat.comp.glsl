#version 430 core

layout(local_size_x = 256) in;

// ── Separate SSBOs (avoids unsized array limitation) ──────────────
layout(std430, binding = 0) buffer AnchorsBuf { vec3 anchors[]; };
layout(std430, binding = 1) buffer PatchesBuf { int patches[]; };
layout(std430, binding = 2) buffer CoefsBuf { float coefs[]; };
layout(std430, binding = 3) buffer MatsBuf { vec4 mats[]; };
layout(std430, binding = 4) buffer MetaBuf { uint body_meta[4]; };

// ── Output image (binding = 0) ────────────────────────────────────
layout(rgba32f, binding = 0) uniform image2D output_img;

// ── Uniforms ──────────────────────────────────────────────────────
uniform mat4  u_view;
uniform mat4  u_proj;
uniform vec2  u_res;
uniform vec3  u_cam;
uniform vec3  u_light;

// ── 8-term corner-vanishing basis ─────────────────────────────────
vec4 basis_ab(float u, float v) {
    float bu = u*(1.0-u), bv = v*(1.0-v);
    return vec4(bu*(1.0-v), bu*v, bv*(1.0-u), bv*u);
}
vec4 basis_cd(float u, float v) {
    float bu = u*(1.0-u), bv = v*(1.0-v);
    return vec4(bu*(2.0*u-1.0)*(1.0-v), bu*(2.0*u-1.0)*v,
                bv*(2.0*v-1.0)*(1.0-u), bv*(2.0*v-1.0)*u);
}

// ── Main ──────────────────────────────────────────────────────────
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint np  = body_meta[1];
    uint gs  = body_meta[2];
    uint total = np * gs * gs;
    if (gid >= total) return;

    // Correct indexing: gid → (patch, si, sj) → (u, v)
    uint sample_idx = gid % (gs * gs);
    uint pi = gid / (gs * gs);
    uint si = sample_idx % gs;
    uint sj = sample_idx / gs;
    float u = (float(si) + 0.5) / float(gs);
    float v = (float(sj) + 0.5) / float(gs);

    // Corners
    int idx = int(pi) * 4;
    vec3 A = anchors[patches[idx]];
    vec3 B = anchors[patches[idx+1]];
    vec3 C = anchors[patches[idx+2]];
    vec3 D = anchors[patches[idx+3]];

    // Bilinear + 8-term deformation
    vec3 P = (1.0-u)*(1.0-v)*A + u*(1.0-v)*B + (1.0-u)*v*C + u*v*D;
    vec3 Nd = normalize(cross((B-A)*(1.0-v)+(D-C)*v, (C-A)*(1.0-u)+(D-B)*u));

    int ci = int(pi)*8;
    float h = dot(vec4(coefs[ci],coefs[ci+1],coefs[ci+2],coefs[ci+3]), basis_ab(u,v)) +
              dot(vec4(coefs[ci+4],coefs[ci+5],coefs[ci+6],coefs[ci+7]), basis_cd(u,v));
    P += Nd * h;

    // Deformed normal (analytical tangent + deformation gradient)
    float eps = 0.005;
    float hu = dot(vec4(coefs[ci],coefs[ci+1],coefs[ci+2],coefs[ci+3]), basis_ab(u+eps,v)) +
               dot(vec4(coefs[ci+4],coefs[ci+5],coefs[ci+6],coefs[ci+7]), basis_cd(u+eps,v)) - h;
    float hv = dot(vec4(coefs[ci],coefs[ci+1],coefs[ci+2],coefs[ci+3]), basis_ab(u,v+eps)) +
               dot(vec4(coefs[ci+4],coefs[ci+5],coefs[ci+6],coefs[ci+7]), basis_cd(u,v+eps)) - h;
    vec3 du = (B-A)*(1.0-v)+(D-C)*v;
    vec3 dv = (C-A)*(1.0-u)+(D-B)*u;
    vec3 N  = normalize(cross(du + Nd*hu, dv + Nd*hv));

    // Project
    vec4 clip = u_proj * u_view * vec4(P, 1.0);
    if (clip.w <= 0.0) return;
    vec3 ndc = clip.xyz / clip.w;
    if (ndc.z < -1.0 || ndc.z > 1.0) return;

    ivec2 px = ivec2((ndc.xy*0.5+0.5)*u_res);
    if (px.x<0||px.x>=int(u_res.x)||px.y<0||px.y>=int(u_res.y)) return;

    // Reverse Z depth: near=1.0, far=0.0
    float depth = 1.0-(ndc.z*0.5+0.5);

    // Depth test
    vec4 cur = imageLoad(output_img, px);
    if (depth <= cur.r) return;

    // Lighting (Blinn-Phong)
    vec3 V = normalize(u_cam - P);
    vec3 L = normalize(u_light);
    vec3 H = normalize(L + V);
    vec3 albedo = mats[pi].rgb;
    vec3 col = albedo*0.2 + albedo*max(dot(N,L),0.0)*0.7 +
               vec3(1.0)*pow(max(dot(N,H),0.0),32.0)*0.3;

    imageStore(output_img, px, vec4(depth, col));
}