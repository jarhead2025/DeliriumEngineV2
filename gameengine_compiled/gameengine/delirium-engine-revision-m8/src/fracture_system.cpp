#include "delirium_m8/fracture_system.hpp"
#include <algorithm>

namespace delirium::m8 {

FractureSystem::FractureSystem(FractureParams params) : params_(std::move(params)) {}

void FractureSystem::set_params(FractureParams p) {
    params_ = std::move(p);
}

bool FractureSystem::should_fracture(const BodyState& state, const BodyTopology& topology) const {
    if (!params_.enable_fracture) return false;

    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        if (state.patches[pid].damage >= params_.damage_threshold) {
            return true;
        }
    }
    return false;
}

std::vector<PatchId> FractureSystem::find_fractured_patches(const BodyState& state,
                                                             const BodyTopology& topology) const {
    std::vector<PatchId> result;
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        if (state.patches[pid].damage >= params_.damage_threshold) {
            result.push_back(pid);
        }
    }
    return result;
}

FractureEvent FractureSystem::evaluate(const BodyState& state, const BodyTopology& topology,
                                        BodyId body_id) const {
    FractureEvent event;
    event.source_body = body_id;
    event.damage_threshold = params_.damage_threshold;

    if (!params_.enable_fracture) return event;

    auto fractured = find_fractured_patches(state, topology);
    if (fractured.size() >= params_.min_fragment_patches) {
        event.triggered = true;
        event.fractured_patches = std::move(fractured);
    }

    return event;
}

} // namespace delirium::m8