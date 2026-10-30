#pragma once

#include "scene.h"
#include "surfel_cache.h"
#include "types.h"
#include <string>
#include <vector>

namespace delirium {

/* ── Engine: frame loop with decoupled physics/render ──────────────
 *
 * Physics runs at a fixed timestep (default 120 Hz).  Render runs
 * at whatever the display provides.  A double-buffered BodyState
 * ensures render always reads a consistent frame.
 * ────────────────────────────────────────────────────────────────── */
class Engine {
public:
    Engine();

    /* Add a scene to the engine. */
    void set_scene(Scene* scene);

    /* Physics configuration. */
    void set_physics_dt(float dt) { physics_dt_ = dt; }
    float physics_dt() const { return physics_dt_; }

    /* External forces (applied to all bodies in the scene). */
    void set_external_forces(ExternalForce f);

    /* Render configuration. */
    void set_viewport(int w, int h) { viewport_w_ = w; viewport_h_ = h; }
    int viewport_width() const { return viewport_w_; }
    int viewport_height() const { return viewport_h_; }

    /* Camera. */
    void set_camera(const Vec3& pos, const Vec3& target, const Vec3& up);
    Vec3 camera_pos() const { return camera_pos_; }

    /* ── Frame loop ─────────────────────────────────────────────── */
    /* Accumulate delta time and step physics at fixed rate. */
    bool update(float dt, std::string* error = nullptr);

    /* Rebuild surfel cache for current frame. */
    bool prepare_render(SurfelCache& cache, std::string* error = nullptr);

    /* Render a frame to a pixel buffer (CPU-side splat). */
    void render_cpu(const SurfelCache& cache, std::vector<std::uint32_t>& out_pixels) const;

    /* Get elapsed simulation time. */
    float simulation_time() const { return accumulated_dt_; }

private:
    Scene* scene_ = nullptr;
    float physics_dt_ = 1.0f / 120.0f;
    float accumulated_dt_ = 0.0f;

    int viewport_w_ = 800, viewport_h_ = 600;

    Vec3 camera_pos_{0, 0, 5};
    Vec3 camera_target_{0, 0, 0};
    Vec3 camera_up_{0, 1, 0};
};

} // namespace delirium