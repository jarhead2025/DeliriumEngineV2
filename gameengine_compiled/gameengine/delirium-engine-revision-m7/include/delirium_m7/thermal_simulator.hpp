#pragma once
#include "delirium_m7/types.hpp"
#include "delirium_m7/body_topology.hpp"
#include "delirium_m7/body_state.hpp"
#include "delirium_m7/material_registry.hpp"
namespace delirium::m7 {
class ThermalSimulator {
public:
    explicit ThermalSimulator(float ambient_temp = 293.15f);
    void diffuse(BodyState& state, const BodyTopology& topology,
                 const MaterialRegistry& materials, float fixed_dt) const;
    void cool(BodyState& state, float fixed_dt) const;
private:
    float ambient_temp_;
};
} // namespace delirium::m7