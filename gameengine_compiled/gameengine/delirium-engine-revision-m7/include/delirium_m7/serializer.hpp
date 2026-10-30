#pragma once
#include "delirium_m7/types.hpp"
#include "delirium_m7/body_topology.hpp"
#include "delirium_m7/body_state.hpp"
#include "delirium_m7/material_registry.hpp"
namespace delirium::m7 {
class Serializer {
public:
    SerializationBlob serialize_topology(const BodyTopology& topology) const;
    SerializationBlob serialize_state(const BodyState& state) const;
    SerializationBlob serialize_materials(const MaterialRegistry& registry) const;
    BodyTopology deserialize_topology(const SerializationBlob& blob) const;
    BodyState deserialize_state(const SerializationBlob& blob) const;
    MaterialRegistry deserialize_materials(const SerializationBlob& blob) const;
private:
    static constexpr std::uint32_t kMagicTopology = 0x544F504Fu; // "TOP"
    static constexpr std::uint32_t kMagicState = 0x53544154u;    // "STAT"
    static constexpr std::uint32_t kMagicMaterials = 0x4D4154u;  // "MAT"
};
} // namespace delirium::m7