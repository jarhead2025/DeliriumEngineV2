#include "core/engine.h"
#include "core/scene.h"

#include <algorithm>
#include <cmath>

namespace delirium {

Engine::Engine() = default;

void Engine::set_scene(Scene* scene) { scene_ = scene; }

void Engine::set_external_forces(ExternalForce f) {
    if (scene_) {
        for (auto* b : scene_->bodies_mut()) {
            if (b) b->set_external_forces(f);
        }
    }
}

void Engine::set_camera(const Vec3& pos, const Vec3& target, const Vec3& up) {
    camera_pos_ = pos;
    camera_target_ = target;
    camera_up_ = up;
}

/* ── update() — fixed-timestep physics accumulator ───────────────── */
bool Engine::update(float dt, std::string* error) {
    if (!scene_) return true;

    accumulated_dt_ += dt;

    // Step physics at fixed rate
    while (accumulated_dt_ >= physics_dt_) {
        if (!scene_->step(physics_dt_, error))
            return false;
        accumulated_dt_ -= physics_dt_;
    }

    return true;
}

/* ── prepare_render() ─────────────────────────────────────────────── */
bool Engine::prepare_render(SurfelCache& cache, std::string* error) {
    if (!scene_) return true;

    float cam_dist = length(camera_pos_ - scene_->bounding_center());
    return scene_->rebuild_cache(cache, cam_dist, error);
}

/* ── render_cpu() — simple splat renderer (depth-tested, CPU-side) ─ */
void Engine::render_cpu(const SurfelCache& cache,
                         std::vector<std::uint32_t>& out_pixels) const {
    int w = viewport_w_, h = viewport_h_;
    out_pixels.assign(w * h, 0); // black

    // Simple perspective projection
    float fov = kPi / 3.0f; // 60 degrees
    float aspect = static_cast<float>(w) / static_cast<float>(h);
    float focal = 1.0f / std::tan(fov / 2.0f);

    // Camera basis
    Vec3 forward = normalize_vec3(camera_target_ - camera_pos_);
    Vec3 right = normalize_vec3(cross(forward, camera_up_));
    Vec3 up = cross(right, forward);

    // Surfels are already sorted far-to-near by the cache
    for (const auto& s : cache.surfels()) {
        // Transform to camera space
        Vec3 p = s.position - camera_pos_;
        float cx = dot(p, right);
        float cy = dot(p, up);
        float cz = dot(p, forward);

        if (cz <= 0.1f) continue; // behind camera

        // Project
        float sx = (cx * focal / cz) / aspect + 0.5f;
        float sy = -(cy * focal / cz) + 0.5f;

        // Convert to pixel coords
        int px = static_cast<int>(sx * w);
        int py = static_cast<int>(sy * h);

        // Splat radius in pixels
        float radius_px = s.radius * focal / cz * std::min(w, h) * 0.5f;
        int r = static_cast<int>(std::ceil(radius_px));

        // Simple circle splat with depth test
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int x = px + dx;
                int y = py + dy;
                if (x < 0 || x >= w || y < 0 || y >= h) continue;

                // Distance from center (normalised)
                float dist = std::sqrt(dx * dx + dy * dy);
                if (dist > r) continue;

                // Depth test: store depth in upper 8 bits of alpha
                int idx = y * w + x;
                float depth_bits = std::max(0.0f, 1.0f - cz / 100.0f);
                std::uint8_t stored_depth = static_cast<std::uint8_t>(depth_bits * 255.0f);

                // Only write if closer (smaller cz = larger depth_bits)
                std::uint8_t existing_depth = static_cast<std::uint8_t>((out_pixels[idx] >> 24) & 0xFF);
                if (stored_depth <= existing_depth) continue;

                // Falloff
                float falloff = 1.0f - (dist / (r + 0.5f));
                falloff = falloff * falloff;

                // Simple diffuse lighting
                Vec3 light_dir = normalize_vec3({1, 1, 1});
                float diff = std::max(0.0f, dot(s.normal, light_dir));
                diff = 0.3f + 0.7f * diff; // ambient + diffuse

                std::uint8_t rr = static_cast<std::uint8_t>(s.r * diff * falloff);
                std::uint8_t gg = static_cast<std::uint8_t>(s.g * diff * falloff);
                std::uint8_t bb = static_cast<std::uint8_t>(s.b * diff * falloff);

                out_pixels[idx] = (stored_depth << 24) | (bb << 16) | (gg << 8) | rr;
            }
        }
    }

    // Strip depth from alpha channel — set full alpha
    for (auto& px : out_pixels) {
        px = 0xFF000000u | (px & 0x00FFFFFFu);
    }
}

} // namespace delirium