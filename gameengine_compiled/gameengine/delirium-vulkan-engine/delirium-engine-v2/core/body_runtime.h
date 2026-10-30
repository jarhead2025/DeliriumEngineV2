#pragma once

#include "body_topology.h"
#include "body_state.h"
#include "surfel_cache.h"
#include "basis_functions.h"
#include "material_registry.h"
#include "modal_system.h"

#include <array>
#include <string>
#include <vector>

namespace delirium {

/* ── BodyRuntime: central orchestrator (M8 pattern) ────────────────
 *
 * Combines topology, double-buffered state, physics modules, and
 * surfel cache into one object.  Read/write access is split:
 *   read_state()  — published, immutable frame (front buffer)
 *   write_state() — mutable, in-progress frame (back buffer)
 *
 * step() advances physics and flips the double buffer.
 * ────────────────────────────────────────────────────────────────── */
class BodyRuntime {
public:
    explicit BodyRuntime(BodyTopology topology, BodyId id = 0);

    /* Initialize anchor positions from rest positions. */
    bool initialize(const std::vector<Vec3>& rest_positions, std::string* error = nullptr);

    /* Double-buffered state access. */
    const BodyState& read_state() const { return states_[front_]; }
    BodyState& write_state() { return states_[1 - front_]; }
    Generation published_generation() const { return read_state().generation; }
    TopologyGeneration published_topology_generation() const { return read_state().topology_generation; }

    /* Topology & identity. */
    const BodyTopology& topology() const { return topology_; }
    BodyId body_id() const { return body_id_; }

    /* Materials. */
    void set_materials(MaterialRegistry materials);
    const MaterialRegistry& materials() const { return materials_; }

    /* Modal physics. */
    void set_modal_system(ModalSystem modal_system);

    /* External forces. */
    void set_external_forces(ExternalForce f);
    ExternalForce& external_forces_mut() { return external_forces_; }

    /* ── Frame step ─────────────────────────────────────────────── */
    bool step(float fixed_dt, std::string* error = nullptr);

    /* ── Surface evaluation ─────────────────────────────────────── */
    SurfacePoint evaluate_surface(PatchId patch_id, float u, float v) const;
    float compute_seam_error(PatchId patch_a, PatchId patch_b) const;

    /* ── Surfel cache ───────────────────────────────────────────── */
    bool rebuild_cache(SurfelCache& cache, std::string* error = nullptr);
    bool rebuild_cache_with_lod(SurfelCache& cache, float camera_distance,
                                std::string* error = nullptr);

    /* ── Bounding box (for culling) ─────────────────────────────── */
    Vec3 bounding_center() const;
    float bounding_radius() const;

    /* ── Time ───────────────────────────────────────────────────── */
    float current_time() const { return time_; }
    void set_time(float t) { time_ = t; }

private:
    void mark_incident_patches(BodyState& state, AnchorId anchor);
    void flip_buffer();

    BodyTopology topology_;
    BodyId body_id_ = 0;
    std::array<BodyState, 2> states_;
    std::size_t front_ = 0;
    bool initialized_ = false;
    float time_ = 0.0f;

    /* Subsystems */
    MaterialRegistry materials_;
    ModalSystem modal_system_;
    std::vector<float> modal_amplitudes_;
    std::vector<float> modal_velocities_;
    BasisEvaluator evaluator_;
    CacheUpdater cache_updater_;
    ExternalForce external_forces_;
};

} // namespace delirium