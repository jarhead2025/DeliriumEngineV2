#include "body_runtime.h"
#include <cmath>
#include <algorithm>

namespace delirium {

/* ── Construction ─────────────────────────────────────────────────── */
BodyRuntime::BodyRuntime(BodyTopology topology, BodyId id)
    : topology_(std::move(topology)), body_id_(id) {}

/* ── Initialize ───────────────────────────────────────────────────── */
bool BodyRuntime::initialize(const std::vector<Vec3>& rest_positions,
                              std::string* error) {
    if (!topology_.finalized()) {
        if (error) *error = "topology not finalized";
        return false;
    }

    std::size_t na = topology_.anchor_count();
    std::size_t np = topology_.patch_count();

    for (auto& st : states_) {
        st.resize(na, np);
        st.topology_generation = topology_.topology_generation();
    }

    // Copy rest positions to both buffers
    for (std::size_t i = 0; i < na && i < rest_positions.size(); ++i) {
        Vec3 pos = rest_positions[i];
        for (auto& st : states_) {
            st.anchors[i].rest_position = pos;
            st.anchors[i].position = pos;
        }
    }

    // Copy patch coefficients from topology to both buffers
    for (std::size_t i = 0; i < np; ++i) {
        for (auto& st : states_) {
            st.patches[i].coefficients = topology_.patch(i).coefficients.scalars;
        }
    }

    // Compute anchor normals from incident patch geometry
    for (auto& st : states_) {
        for (AnchorId aid = 0; aid < na; ++aid) {
            Vec3 n{};
            const auto& pids = topology_.patches_for_anchor(aid);
            for (PatchId pid : pids) {
                const auto& pt = topology_.patch(pid);
                auto idx = [aid, &pt]() {
                    for (int c = 0; c < 4; ++c)
                        if (pt.anchors[c] == aid) return c;
                    return -1;
                }();
                if (idx < 0) continue;
                // Adjacent corners: idx^1 and idx^2 (edge neighbors in quad)
                int c1 = idx ^ 1;
                int c2 = idx ^ 2;
                Vec3 t1 = st.anchors[pt.anchors[c1]].position - st.anchors[aid].position;
                Vec3 t2 = st.anchors[pt.anchors[c2]].position - st.anchors[aid].position;
                n = n + normalize_vec3(cross(t1, t2));
            }
            st.anchors[aid].normal = normalize_vec3(n);
        }
    }

    initialized_ = true;
    return true;
}

/* ── Materials ────────────────────────────────────────────────────── */
void BodyRuntime::set_materials(MaterialRegistry materials) {
    materials_ = std::move(materials);
}

/* ── Modal system ─────────────────────────────────────────────────── */
void BodyRuntime::set_modal_system(ModalSystem modal_system) {
    modal_system_ = std::move(modal_system);
}

/* ── External forces ──────────────────────────────────────────────── */
void BodyRuntime::set_external_forces(ExternalForce f) {
    external_forces_ = std::move(f);
}

/* ── step() ─────────────────────────────────────────────────────────
 *
 * Physics pipeline:
 *   1. Apply external forces (gravity, drag)
 *   2. Integrate modal dynamics
 *   3. Semi-implicit Euler on anchor velocities
 *   4. Flip double buffer
 * ─────────────────────────────────────────────────────────────────── */
bool BodyRuntime::step(float fixed_dt, std::string* error) {
    if (!initialized_) {
        if (error) *error = "not initialized";
        return false;
    }

    BodyState& state = write_state();

    // 1. External forces
    for (auto& anchor : state.anchors) {
        // Gravity
        anchor.velocity += external_forces_.gravity * fixed_dt;
        // Drag
        if (external_forces_.drag > 0.0f) {
            anchor.velocity *= std::exp(-external_forces_.drag * fixed_dt);
        }
    }

    // 2. Modal integration (init from mode.amplitude, persist across frames)
    if (modal_system_.mode_count() > 0) {
        if (modal_amplitudes_.size() != modal_system_.mode_count()) {
            modal_amplitudes_.resize(modal_system_.mode_count());
            modal_velocities_.resize(modal_system_.mode_count());
            for (std::size_t i = 0; i < modal_system_.mode_count(); ++i) {
                modal_amplitudes_[i] = modal_system_.modes()[i].amplitude;
                modal_velocities_[i] = 0.0f;
            }
        }
        modal_system_.integrate(modal_amplitudes_, modal_velocities_, fixed_dt);
        modal_system_.write_back(modal_amplitudes_, state, topology_);
    }

    // 3. Semi-implicit Euler on anchors
    for (auto& anchor : state.anchors) {
        anchor.position += anchor.velocity * fixed_dt;
    }

    // 4. Flip
    flip_buffer();
    time_ += fixed_dt;

    return true;
}

/* ── flip_buffer() ────────────────────────────────────────────────── */
void BodyRuntime::flip_buffer() {
    const BodyState& old_front = states_[front_];
    BodyState& back = states_[1 - front_];

    back.generation = old_front.generation + 1;
    back.topology_generation = topology_.topology_generation();
    back.clear_dirty();

    front_ = 1 - front_;
}

/* ── mark_incident_patches() ──────────────────────────────────────── */
void BodyRuntime::mark_incident_patches(BodyState& state, AnchorId anchor) {
    const auto& pids = topology_.patches_for_anchor(anchor);
    for (PatchId pid : pids) {
        state.dirty_patches.mark(pid);
    }
}

/* ── evaluate_surface() ───────────────────────────────────────────── */
SurfacePoint BodyRuntime::evaluate_surface(PatchId patch_id, float u, float v) const {
    return evaluator_.evaluate(read_state(), topology_, patch_id, u, v);
}

/* ── compute_seam_error() ─────────────────────────────────────────── */
float BodyRuntime::compute_seam_error(PatchId patch_a, PatchId patch_b) const {
    return evaluator_.compute_seam_error(read_state(), topology_, patch_a, patch_b);
}

/* ── rebuild_cache() ──────────────────────────────────────────────── */
bool BodyRuntime::rebuild_cache(SurfelCache& cache, std::string* error) {
    return cache_updater_.rebuild(cache, read_state(), topology_, materials_, error);
}

bool BodyRuntime::rebuild_cache_with_lod(SurfelCache& cache, float cam_dist,
                                          std::string* error) {
    return cache_updater_.rebuild_with_lod(cache, read_state(), topology_,
                                            materials_, cam_dist, error);
}

/* ── bounding_center() ────────────────────────────────────────────── */
Vec3 BodyRuntime::bounding_center() const {
    const auto& anchors = read_state().anchors;
    if (anchors.empty()) return {};
    Vec3 sum{};
    for (const auto& a : anchors)
        sum = sum + a.position;
    return sum * (1.0f / static_cast<float>(anchors.size()));
}

/* ── bounding_radius() ────────────────────────────────────────────── */
float BodyRuntime::bounding_radius() const {
    const auto& anchors = read_state().anchors;
    Vec3 center = bounding_center();
    float max_r = 0.0f;
    for (const auto& a : anchors) {
        float r = length(a.position - center);
        if (r > max_r) max_r = r;
    }
    return max_r;
}

} // namespace delirium