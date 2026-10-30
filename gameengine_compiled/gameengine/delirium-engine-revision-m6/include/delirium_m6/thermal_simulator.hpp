#pragma once
#include "delirium_m6/types.hpp"
#include "delirium_m6/body_topology.hpp"
#include "delirium_m6/body_state.hpp"
#include "delirium_m6/material_registry.hpp"
namespace delirium::m6 {
class ThermalSimulator {
public:
    explicit ThermalSimulator(float diffusion_rate = 0.1f);
    void diffuse(BodyState& state, const BodyTopology& topology,
                 const MaterialRegistry& materials, float fixed_dt) const;
    void apply_expansion(BodyState& state, const BodyTopology& topology,
                         const MaterialRegistry& materials, float fixed_dt) const;
    void cool(BodyState& state, const BodyTopology& /*topology*/,
              float ambient, float rate, float fixed_dt) const;
private:
    float diffusion_rate_;
};
} // namespace delirium::m6