#pragma once

#include "types.h"
#include "body_topology.h"
#include "body_state.h"
#include <string>
#include <vector>

namespace delirium {

class ModalSystem {
public:
    void add_mode(ModalMode mode);
    const std::vector<ModalMode>& modes() const { return modes_; }
    std::size_t mode_count() const { return modes_.size(); }

    /* Integrate modal amplitudes using semi-implicit (symplectic) Euler.
     * CONDITIONALLY stable: requires dt * omega < 2 for the highest mode. */
    void integrate(std::vector<float>& amplitudes, std::vector<float>& velocities,
                   float dt) const;

    /* Write modal displacement back onto patch coefficients.
     * Each mode contributes to the 8-term corner-vanishing basis. */
    void write_back(const std::vector<float>& amplitudes,
                    BodyState& state, const BodyTopology& topology) const;

private:
    std::vector<ModalMode> modes_;
};

} // namespace delirium