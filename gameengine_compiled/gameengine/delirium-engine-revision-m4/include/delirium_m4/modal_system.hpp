#pragma once

#include "delirium_m4/types.hpp"
#include "delirium_m4/body_state.hpp"

#include <vector>

namespace delirium::m4 {

// ---------------------------------------------------------------------------
// ModalSystem — M4: global elastic displacement stage
// ---------------------------------------------------------------------------
//
// Computes modal displacements for all anchors. Modes compose BEFORE the
// local material solve. Each mode has an amplitude, frequency, and phase
// that can be animated over time.
//
// Pipeline order: bone pose → modal displacement → local material solve
// ---------------------------------------------------------------------------
class ModalSystem {
public:
    ModalSystem();

    // Register a modal mode
    void add_mode(const ModalMode& mode);

    // Remove all modes
    void clear();

    // Compute modal displacements for all anchors
    // Writes into state.anchors[i].modal_displacement
    void compute(BodyState& state, float time) const;

    // Get total modal displacement for a single anchor
    Vec3 compute_anchor_displacement(const Vec3& rest_position,
                                     float time,
                                     AnchorId anchor_id) const;

    // Number of registered modes
    std::size_t mode_count() const { return modes_.size(); }

    // Get a mode by index
    const ModalMode& mode(std::size_t index) const;

private:
    std::vector<ModalMode> modes_;

    // Compute displacement for a single mode at a single anchor
    Vec3 compute_mode_displacement(const ModalMode& mode,
                                   const Vec3& rest_position,
                                   float time) const;
};

} // namespace delirium::m4