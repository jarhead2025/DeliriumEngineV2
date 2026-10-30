#include "delirium_m4/body_runtime.hpp"

#include <utility>

namespace delirium::m4 {

BodyRuntime::BodyRuntime(BodyTopology topology)
    : topology_(std::move(topology)) {}

bool BodyRuntime::initialize(const std::vector<Vec3>& rest_positions,
                              std::string* error) {
    if (!topology_.finalized()) {
        if (error) *error = "topology must be finalized before runtime initialization";
        return false;
    }
    if (rest_positions.size() != topology_.anchor_count()) {
        if (error) *error = "rest position count does not match topology anchor count";
        return false;
    }

    for (BodyState& state : states_) {
        state.resize(topology_.anchor_count(), topology_.patch_count());
        state.generation = 0;
        state.topology_generation = topology_.topology_generation();
        for (AnchorId id = 0; id < rest_positions.size(); ++id) {
            state.anchors[id].rest_position = rest_positions[id];
            state.anchors[id].position = rest_positions[id];
        }
        // Initialize patch effective stiffness
        for (std::size_t pid = 0; pid < topology_.patch_count(); ++pid) {
            state.patches[pid].effective_stiffness = 1.0f;
        }
        state.clear_dirty();
    }

    front_ = 0;
    initialized_ = true;
    return true;
}

void BodyRuntime::set_materials(MaterialRegistry materials) {
    materials_ = std::move(materials);
}

void BodyRuntime::set_modal_system(ModalSystem modal_system) {
    modal_system_ = std::move(modal_system);
}

void BodyRuntime::mark_incident_patches(BodyState& state, AnchorId anchor) {
    for (PatchId patch : topology_.patches_for_anchor(anchor)) {
        state.dirty_patches.mark(patch);
    }
}

bool BodyRuntime::step(float fixed_dt, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    if (fixed_dt <= 0.0f) {
        if (error) *error = "fixed step must be positive";
        return false;
    }

    const std::size_t back = 1 - front_;
    const BodyState& read = states_[front_];
    BodyState& write = states_[back];
    write = read;
    write.clear_dirty();

    // 1. Mark dirty anchors (moved from rest position)
    for (AnchorId anchor = 0; anchor < write.anchors.size(); ++anchor) {
        AnchorState& state = write.anchors[anchor];
        if (!nearly_equal(state.position, state.rest_position)) {
            write.dirty_anchors.mark(anchor);
            mark_incident_patches(write, anchor);
        }
    }

    // 2. Modal displacement (global elastic stage — BEFORE local solve)
    time_ += fixed_dt;
    modal_system_.compute(write, time_);

    // Apply modal displacement to anchor positions
    for (AnchorId anchor = 0; anchor < write.anchors.size(); ++anchor) {
        write.anchors[anchor].position += write.anchors[anchor].modal_displacement;
    }

    // 3. Local elastic/plastic solve (sparse, dirty patches only)
    elastic_solver_.solve(write, topology_, materials_, fixed_dt, error);
    if (error && !error->empty()) return false;

    // 4. Damage accumulation (monotonic)
    damage_accumulator_.accumulate(write, topology_, materials_);

    // 5. Strain propagation between adjacent patches
    elastic_solver_.propagate_strain(write, topology_);

    // 6. Publish new state
    write.generation = read.generation + 1;
    write.topology_generation = topology_.topology_generation();
    front_ = back;

    return true;
}

SurfacePoint BodyRuntime::evaluate_surface(PatchId patch_id, float u, float v) const {
    return evaluator_.evaluate_patch(read_state(), topology_, patch_id, u, v);
}

float BodyRuntime::compute_seam_error(PatchId patch_a, PatchId patch_b) const {
    return evaluator_.compute_seam_error(read_state(), topology_, patch_a, patch_b);
}

bool BodyRuntime::solve_welded_seams(std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    BodyState& write = states_[1 - front_];
    return seam_solver_.solve(write, topology_, error);
}

bool BodyRuntime::update_cache(SurfelCache& cache, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    return cache_updater_.update(cache, read_state(), topology_, materials_, error);
}

bool BodyRuntime::rebuild_cache(SurfelCache& cache, std::string* error) {
    if (!initialized_) {
        if (error) *error = "runtime not initialized";
        return false;
    }
    return cache_updater_.rebuild(cache, read_state(), topology_, materials_, error);
}

// ---------------------------------------------------------------------------
// SurfaceEvaluator implementation
// ---------------------------------------------------------------------------

SurfacePoint SurfaceEvaluator::evaluate_patch(const BodyState& state,
                                               const BodyTopology& topology,
                                               PatchId patch_id,
                                               float u, float v) const {
    const PatchTopology& patch = topology.patch(patch_id);

    const Vec3& p00 = state.anchors[patch.anchors[0]].position;
    const Vec3& p10 = state.anchors[patch.anchors[1]].position;
    const Vec3& p01 = state.anchors[patch.anchors[2]].position;
    const Vec3& p11 = state.anchors[patch.anchors[3]].position;

    Vec3 pos = evaluate_patch_position(p00, p10, p01, p11,
                                        patch.basis, patch.coefficients.scalars,
                                        patch.coefficients.residuals, u, v);
    Vec3 normal = evaluate_patch_normal(p00, p10, p01, p11,
                                         patch.basis, patch.coefficients.scalars,
                                         patch.coefficients.residuals, u, v);

    return {pos, normal, patch.material, patch_id, u, v};
}

std::vector<SurfacePoint> SurfaceEvaluator::evaluate_dirty_patches(
    const BodyState& state,
    const BodyTopology& topology,
    const DirtySet& dirty_patches,
    std::uint32_t grid_resolution) const {

    std::vector<SurfacePoint> points;
    auto dirty_indices = dirty_patches.indices();
    float step = 1.0f / grid_resolution;

    points.reserve(dirty_indices.size() * (grid_resolution + 1) * (grid_resolution + 1));

    for (PatchId pid : dirty_indices) {
        for (std::uint32_t i = 0; i <= grid_resolution; ++i) {
            for (std::uint32_t j = 0; j <= grid_resolution; ++j) {
                float u = i * step;
                float v = j * step;
                points.push_back(evaluate_patch(state, topology, pid, u, v));
            }
        }
    }

    return points;
}

float SurfaceEvaluator::compute_seam_error(const BodyState& state,
                                            const BodyTopology& topology,
                                            PatchId patch_a, PatchId patch_b) const {
    const PatchTopology& pa = topology.patch(patch_a);
    const PatchTopology& pb = topology.patch(patch_b);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (pa.anchors[i] == pb.anchors[j]) {
                return 0.0f;
            }
        }
    }

    return 1e6f;
}

} // namespace delirium::m4