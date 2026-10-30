#include "ray_marcher.h"
#include <cmath>
#include <algorithm>

namespace delirium {

// ---------------------------------------------------------------------------
// RayMarcher::march — the core rendering operation
// Far→near marching. Closer hits override farther ones (z-min splatting).
// Fixed compute: N_steps per ray. No scene complexity penalty.
// ---------------------------------------------------------------------------

vec3 RayMarcher::march(const Ray& ray, const CoefficientRegister& reg, const vec3& light_pos) {
    vec3 result(0.05f, 0.05f, 0.1f); // Background color (dark blue)
    float hit_depth = ray.t_max;
    vec3 hit_normal(0, 0, 1);
    vec3 hit_color(1, 1, 1);
    float hit_material = 0.5f;
    bool hit = false;

    // March from far to near — closer hits override
    for(int i = steps; i >= 0; i--) {
        float t = ray.t_min + (ray.t_max - ray.t_min) * i / steps;
        vec3 p = ray.point_at(t);

        float depth;
        vec3 normal, color;
        float material;

        if(BasisFunction::query_surface(reg, p.x, p.y, p.z, depth, normal, color, material)) {
            if(t < hit_depth) {
                hit_depth = t;
                hit_normal = normal;
                hit_color = color;
                hit_material = material;
                hit = true;
            }
        }
    }

    if(!hit) return result;

    // Lighting: simple diffuse + specular via vector additions
    vec3 to_light = (light_pos - vec3(0,0,0) + hit_normal * hit_depth).normalized();
    float diffuse = fmaxf(0.0f, hit_normal.dot(to_light));

    vec3 view_dir = -ray.direction;
    vec3 reflect_dir = hit_normal * (2.0f * hit_normal.dot(to_light)) - to_light;
    float specular = powf(fmaxf(0.0f, view_dir.dot(reflect_dir)), 32.0f) * (1.0f - hit_material);

    // Final color: ambient + diffuse + specular
    result = hit_color * (0.2f + 0.7f * diffuse) + vec3(specular * 0.5f);

    return result;
}

// ---------------------------------------------------------------------------
// march_with_bounce — lateral ray bounce for reflections
// All lighting is simple vector additions toward POV.
// ---------------------------------------------------------------------------

vec3 RayMarcher::march_with_bounce(const Ray& ray, const CoefficientRegister& reg,
                                    const vec3& light_pos, int bounce_depth) {
    if(bounce_depth < 0) return vec3(0);

    vec3 primary = march(ray, reg, light_pos);

    // If we hit something reflective, trace a bounce ray
    if(bounce_depth > 0 && max_bounces > 0) {
        // Find hit point for bounce direction
        float hit_depth = ray.t_max;
        vec3 hit_normal(0, 0, 1);
        vec3 hit_pos;
        float hit_material = 0.5f;

        for(int i = steps; i >= 0; i--) {
            float t = ray.t_min + (ray.t_max - ray.t_min) * i / steps;
            vec3 p = ray.point_at(t);
            float depth; vec3 n, c; float m;
            if(BasisFunction::query_surface(reg, p.x, p.y, p.z, depth, n, c, m)) {
                if(t < hit_depth) {
                    hit_depth = t;
                    hit_pos = p;
                    hit_normal = n;
                    hit_material = m;
                }
            }
        }

        if(hit_depth < ray.t_max && hit_material < 0.7f) {
            // Reflect ray
            vec3 reflect_dir = hit_normal * (2.0f * hit_normal.dot(-ray.direction)) + ray.direction;
            Ray bounce_ray(hit_pos + hit_normal * 0.01f, reflect_dir.normalized(), 0.01f, ray.t_max);

            vec3 bounce_color = march_with_bounce(bounce_ray, reg, light_pos, bounce_depth - 1);
            float reflectivity = 1.0f - hit_material;
            primary = primary * (1.0f - reflectivity * 0.3f) + bounce_color * reflectivity * 0.3f;
        }
    }

    return primary;
}

// ---------------------------------------------------------------------------
// render_frame — full frame render
// W × H × N_steps operations. Fixed compute, no scene complexity penalty.
// ---------------------------------------------------------------------------

void RayMarcher::render_frame(const Camera& cam, const CoefficientRegister& reg,
                               const vec3& light_pos, float* out_rgba) {
    for(int y = 0; y < cam.height; y++) {
        for(int x = 0; x < cam.width; x++) {
            Ray ray = cam.ray_for_pixel(x, y);
            vec3 color = march_with_bounce(ray, reg, light_pos, max_bounces);

            // Tone mapping (simple Reinhard)
            color = color / (color + vec3(1.0f));

            // Gamma correction
            color = vec3(powf(fmaxf(0.0f, color.x), 1.0f/2.2f),
                         powf(fmaxf(0.0f, color.y), 1.0f/2.2f),
                         powf(fmaxf(0.0f, color.z), 1.0f/2.2f));

            int idx = (y * cam.width + x) * 4;
            out_rgba[idx]   = color.x;
            out_rgba[idx+1] = color.y;
            out_rgba[idx+2] = color.z;
            out_rgba[idx+3] = 1.0f;
        }
    }
}

} // namespace delirium