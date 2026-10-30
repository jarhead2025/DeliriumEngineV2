#pragma once

#include "delirium_m5/types.hpp"
#include "delirium_m5/body_topology.hpp"
#include "delirium_m5/body_state.hpp"
#include "delirium_m5/material_registry.hpp"
#include <string>

namespace delirium::m5 {

// ThermalSimulator — heat diffusion through anchors + thermal expansion
class ThermalSimulator {
public:
    explicit ThermalSimulator(float diffusion_rate = 0.1f);

    // Diffuse heat through anchor network
    void diffuse(BodyState& state,
                 const BodyTopology& topology,
                 const MaterialRegistry& materials,
                 float fixed_dt) const;

    // Apply thermal expansion to anchor positions
    void apply_expansion(BodyState& state,
                         const BodyTopology& topology,
                         const MaterialRegistry& materials,
                         float fixed_dt) const;

    // Cool anchors toward ambient temperature
    void cool(BodyState& state,
              const BodyTopology& topology,
              float ambient_temperature,
              float cooling_rate,
              float fixed_dt) const;

private:
    float diffusion_rate_;
};

} // namespace delirium::m5