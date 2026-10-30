#version 430 core

// Ray marching fragment shader — full implicit surface rendering.
// Evaluates 8 basis surface types per patch, interpolates UVs for
// texture sampling, and reconstructs the original model's appearance
// through coefficient-driven implicit surfaces.
// No triangles, no surfels, no geometry buffers — pure math on ALUs.

precision highp float;

// ─── Camera & View ───────────────────────────────────────────────
uniform vec3  u_camera_pos;
uniform mat4  u_view_inv;       // view → world
uniform mat4  u_proj_inv;       // clip → view
uniform vec2  u_resolution;

// ─── Scene Geometry (implicit) ───────────────────────────────────
// Anchor positions: up to 512 anchors
uniform vec3  u_anchors[512];
uniform int   u_anchor_count;

// Anchor UV coordinates (from OBJ texture mapping)
uniform vec2  u_anchor_uvs[512];

// Patch connectivity: up to 392 patches (14x14 grid)
uniform ivec4 u_patches[392];
uniform int   u_patch_count;

// Patch basis type (0=Bilinear, 1=Plano, 2=Cupula, 3=Silla,
//                 4=ValleU, 5=ValleV, 6=Eggbox, 7=Twist)
uniform int   u_patch_basis[392];

// Patch shape amplitude (scalar coefficient for basis deviation)
uniform float u_patch_amplitude[392];

// Material colors per patch (fallback when no texture)
uniform vec3  u_patch_colors[392];

// ─── Texture ─────────────────────────────────────────────────────
uniform sampler2D u_texture;
uniform int       u_has_texture;  // 0 = use patch colors, 1 = sample texture

// ─── Output ──────────────────────────────────────────────────────
out vec4 frag_color;

// ─── Constants ───────────────────────────────────────────────────
const int   MAX_STEPS    = 128;     // fixed-step iterations
const float STEP_SIZE    = 0.03;    // step size (small enough to not miss surface)
const float SURFEL_EPS   = 0.015;   // convergence threshold
const float MAX_DIST     = 50.0;    // ray max travel
const int   SAMPLES      = 4;       // bilinear grid samples per patch (4x4 = 16)

// ─── Basis shape functions ───────────────────────────────────────
// Each basis adds a shape function on top of bilinear interpolation.
// The shape function is scaled by the patch's amplitude coefficient
// and applied along the surface normal.

float cupula_shape(float u, float v) {
    return 4.0 * u * (1.0 - u) * v * (1.0 - v);  // dome, max=1 at center
}

float silla_shape(float u, float v) {
    return (u - 0.5) * (v - 0.5) * 4.0;  // hyperbolic saddle
}

float valle_u_shape(float u, float v) {
    return -4.0 * u * (1.0 - u);  // valley along U
}

float valle_v_shape(float u, float v) {
    return -4.0 * v * (1.0 - v);  // valley along V
}

float eggbox_shape(float u, float v) {
    return 16.0 * (u - 0.5) * (u - 0.5) * (v - 0.5) * (v - 0.5) - 0.25;
}

float twist_shape(float u, float v) {
    return (u - 0.5) * (v - 0.5) * 8.0 * (u + v - 1.0);
}

float get_basis_shape(int basis_type, float u, float v) {
    if (basis_type == 0) return 0.0;  // Bilinear
    if (basis_type == 1) return 0.0;  // Plano (flat)
    if (basis_type == 2) return cupula_shape(u, v);
    if (basis_type == 3) return silla_shape(u, v);
    if (basis_type == 4) return valle_u_shape(u, v);
    if (basis_type == 5) return valle_v_shape(u, v);
    if (basis_type == 6) return eggbox_shape(u, v);
    if (basis_type == 7) return twist_shape(u, v);
    return 0.0;
}

// ─── Bilinear patch evaluation ───────────────────────────────────
// P(u,v) = (1-u)(1-v)*A + u(1-v)*B + (1-u)v*C + uv*D
vec3 bilinear_point(vec3 A, vec3 B, vec3 C, vec3 D, float u, float v) {
    return (1.0 - u) * (1.0 - v) * A
         + u       * (1.0 - v) * B
         + (1.0 - u) * v       * C
         + u       * v         * D;
}

// ─── Bilinear normal ─────────────────────────────────────────────
vec3 bilinear_normal(vec3 A, vec3 B, vec3 C, vec3 D, float u, float v) {
    float eps = 0.001;
    vec3 P  = bilinear_point(A, B, C, D, u, v);
    vec3 Pu = bilinear_point(A, B, C, D, u + eps, v);
    vec3 Pv = bilinear_point(A, B, C, D, u, v + eps);
    return normalize(cross(Pu - P, Pv - P));
}

// ─── Full basis-evaluated point ──────────────────────────────────
// P(u,v) = bilinear + amplitude * shape(u,v) * normal + residual_blend
vec3 basis_point(vec3 A, vec3 B, vec3 C, vec3 D,
                 int basis_type, float amplitude,
                 float u, float v) {
    vec3 p = bilinear_point(A, B, C, D, u, v);

    // Add basis shape contribution along the surface normal
    float shape = get_basis_shape(basis_type, u, v);
    if (shape != 0.0 && amplitude != 0.0) {
        vec3 n = bilinear_normal(A, B, C, D, u, v);
        p += n * shape * amplitude;
    }

    return p;
}

// ─── UV interpolation ────────────────────────────────────────────
// Bilinear interpolation of UV coordinates at patch-local (u,v)
vec2 bilinear_uv(vec2 uA, vec2 uB, vec2 uC, vec2 uD, float u, float v) {
    return (1.0 - u) * (1.0 - v) * uA
         + u       * (1.0 - v) * uB
         + (1.0 - u) * v       * uC
         + u       * v         * uD;
}

// ─── Distance to a single patch (with basis evaluation) ──────────
// Evaluates the patch at a SAMPLES×SAMPLES grid, finds closest point.
// Also returns the (u,v) parameter of the closest sample for UV lookup.
float patch_distance(vec3 p, vec3 A, vec3 B, vec3 C, vec3 D,
                     int basis_type, float amplitude,
                     out vec3 closest, out vec2 closest_uv) {
    float min_d = 1e10;
    closest = vec3(0.0);
    closest_uv = vec2(0.5);

    for (int i = 0; i < SAMPLES; ++i) {
        for (int j = 0; j < SAMPLES; ++j) {
            float u = (float(i) + 0.5) / float(SAMPLES);
            float v = (float(j) + 0.5) / float(SAMPLES);
            vec3 sp = basis_point(A, B, C, D, basis_type, amplitude, u, v);
            float d = length(p - sp);
            if (d < min_d) {
                min_d = d;
                closest = sp;
                closest_uv = vec2(u, v);
            }
        }
    }

    // Gradient refinement: one step of gradient descent toward closest sample
    if (min_d > SURFEL_EPS) {
        float step = 0.05;
        vec3 best = closest;
        vec2 best_uv = closest_uv;
        for (int i = 0; i < SAMPLES; ++i) {
            for (int j = 0; j < SAMPLES; ++j) {
                float u = clamp((float(i) + 0.5) / float(SAMPLES) + step, 0.0, 1.0);
                float v = clamp((float(j) + 0.5) / float(SAMPLES) + step, 0.0, 1.0);
                vec3 sp = basis_point(A, B, C, D, basis_type, amplitude, u, v);
                float d = length(p - sp);
                if (d < min_d) {
                    min_d = d;
                    best = sp;
                    best_uv = vec2(u, v);
                }
            }
        }
        closest = best;
        closest_uv = best_uv;
    }

    return min_d;
}

// ─── Scene SDF (Signed Distance Function) ────────────────────────
// Minimum distance to any patch. Returns 0 if on surface.
// Also outputs the hit UV for texture sampling.
float scene_sdf(vec3 p, out vec3 hit_point, out int hit_patch, out vec2 hit_uv) {
    float min_dist = 1e10;
    hit_point = vec3(0.0);
    hit_patch = -1;
    hit_uv = vec2(0.0);

    for (int pi = 0; pi < u_patch_count; ++pi) {
        ivec4 pidx = u_patches[pi];
        vec3 A = u_anchors[pidx.x];
        vec3 B = u_anchors[pidx.y];
        vec3 C = u_anchors[pidx.z];
        vec3 D = u_anchors[pidx.w];

        int basis = u_patch_basis[pi];
        float amp = u_patch_amplitude[pi];

        vec3 closest;
        vec2 cuv;
        float d = patch_distance(p, A, B, C, D, basis, amp, closest, cuv);

        if (d < min_dist) {
            min_dist = d;
            hit_point = closest;
            hit_patch = pi;
            hit_uv = cuv;
        }
    }

    return min_dist;
}

// ─── SDF-only version (for gradient computation) ─────────────────
float scene_sdf_dist(vec3 p) {
    float min_dist = 1e10;

    for (int pi = 0; pi < u_patch_count; ++pi) {
        ivec4 pidx = u_patches[pi];
        vec3 A = u_anchors[pidx.x];
        vec3 B = u_anchors[pidx.y];
        vec3 C = u_anchors[pidx.z];
        vec3 D = u_anchors[pidx.w];

        int basis = u_patch_basis[pi];
        float amp = u_patch_amplitude[pi];

        vec3 closest;
        vec2 cuv;
        float d = patch_distance(p, A, B, C, D, basis, amp, closest, cuv);
        if (d < min_dist) min_dist = d;
    }

    return min_dist;
}

// ─── Analytical normal from gradient ─────────────────────────────
vec3 compute_normal(vec3 p) {
    float eps = 0.005;
    float d  = scene_sdf_dist(p);
    float dx = scene_sdf_dist(p + vec3(eps, 0.0, 0.0)) -
               scene_sdf_dist(p - vec3(eps, 0.0, 0.0));
    float dy = scene_sdf_dist(p + vec3(0.0, eps, 0.0)) -
               scene_sdf_dist(p - vec3(0.0, eps, 0.0));
    float dz = scene_sdf_dist(p + vec3(0.0, 0.0, eps)) -
               scene_sdf_dist(p - vec3(0.0, 0.0, eps));

    return normalize(vec3(dx, dy, dz) * (0.5 / eps));
}

// ─── Compute hit UV from patch + local (u,v) ─────────────────────
vec2 compute_hit_uv(int patch_id, vec2 local_uv) {
    if (patch_id < 0 || patch_id >= u_patch_count) return vec2(0.0);

    ivec4 pidx = u_patches[patch_id];
    vec2 uA = u_anchor_uvs[pidx.x];
    vec2 uB = u_anchor_uvs[pidx.y];
    vec2 uC = u_anchor_uvs[pidx.z];
    vec2 uD = u_anchor_uvs[pidx.w];

    return bilinear_uv(uA, uB, uC, uD, local_uv.x, local_uv.y);
}

// ─── Ray generation from pixel ───────────────────────────────────
vec3 generate_ray(vec2 pixel) {
    vec2 ndc = (pixel / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Unproject through inverse projection matrix
    vec4 clip = vec4(ndc, -1.0, 1.0);  // near plane
    vec3 dir = normalize((u_proj_inv * clip).xyz);

    // Transform to world space
    mat3 view_rot = mat3(u_view_inv);
    return view_rot * dir;
}

// ─── Lighting ────────────────────────────────────────────────────
vec3 shade(vec3 p, vec3 normal, int patch_id, vec2 tex_uv) {
    vec3 light_dir = normalize(vec3(1.0, 2.0, 3.0));
    vec3 view_dir  = normalize(u_camera_pos - p);
    vec3 half_dir  = normalize(light_dir + view_dir);

    // Albedo: texture sample or flat patch color
    vec3 albedo;
    if (u_has_texture == 1) {
        albedo = texture(u_texture, tex_uv).rgb;
    } else {
        albedo = (patch_id >= 0 && patch_id < u_patch_count)
                 ? u_patch_colors[patch_id] : vec3(0.8);
    }

    // Ambient
    vec3 ambient = albedo * 0.15;

    // Diffuse (Lambertian)
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = albedo * diff * 0.7;

    // Specular (Blinn-Phong)
    float spec = pow(max(dot(normal, half_dir), 0.0), 32.0);
    vec3 specular = vec3(1.0) * spec * 0.3;

    return ambient + diffuse + specular;
}

// ─── Main ────────────────────────────────────────────────────────
void main() {
    vec2 pixel = gl_FragCoord.xy;
    vec3 ray_dir = generate_ray(pixel);
    vec3 ray_pos = u_camera_pos;

    // ─── Fixed-step ray marching ─────────────────────────────────
    // Track the closest point found along the entire ray.
    float total_dist = 0.0;
    vec3  closest_hit = vec3(0.0);
    int   closest_patch = -1;
    vec2  closest_uv = vec2(0.0);
    float best_dist = 1e10;

    for (int step = 0; step < MAX_STEPS; ++step) {
        vec3 hit_point;
        int   hit_patch;
        vec2  hit_uv;
        float d = scene_sdf(ray_pos, hit_point, hit_patch, hit_uv);

        if (d < best_dist) {
            best_dist     = d;
            closest_hit   = hit_point;
            closest_patch = hit_patch;
            closest_uv    = hit_uv;
        }

        if (d < SURFEL_EPS) {
            closest_hit   = hit_point;
            closest_patch = hit_patch;
            closest_uv    = hit_uv;
            best_dist     = d;
            break;
        }

        if (total_dist > MAX_DIST)
            break;

        ray_pos    += ray_dir * STEP_SIZE;
        total_dist += STEP_SIZE;
    }

    // ─── Shade or background ─────────────────────────────────────
    if (best_dist < SURFEL_EPS * 3.0) {
        vec3 normal = compute_normal(closest_hit);

        // Get texture UV from the hit patch's local (u,v)
        vec2 tex_uv = compute_hit_uv(closest_patch, closest_uv);

        vec3 color = shade(closest_hit, normal, closest_patch, tex_uv);

        // Fog for depth cueing
        float fog = exp(-total_dist * 0.02);
        vec3  fog_color = vec3(0.1, 0.15, 0.2);
        color = mix(fog_color, color, fog);

        frag_color = vec4(color, 1.0);
    } else {
        // Background gradient (sky)
        float t = ray_dir.y * 0.5 + 0.5;
        vec3 sky = mix(vec3(0.05, 0.08, 0.15), vec3(0.2, 0.3, 0.5), t);
        frag_color = vec4(sky, 1.0);
    }
}