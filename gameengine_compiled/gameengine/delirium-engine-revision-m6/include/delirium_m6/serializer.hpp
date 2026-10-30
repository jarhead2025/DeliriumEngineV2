#pragma once
#include "delirium_m6/types.hpp"
#include "delirium_m6/body_topology.hpp"
#include "delirium_m6/body_state.hpp"
#include "delirium_m6/material_registry.hpp"
#include <string>
namespace delirium::m6 {
// Serializer — save/load body topology, state, and materials
class Serializer {
public:
    // Serialize topology to binary blob
    SerializationBlob serialize_topology(const BodyTopology& topology) const;
    // Serialize body state to binary blob
    SerializationBlob serialize_state(const BodyState& state) const;
    // Serialize material registry to binary blob
    SerializationBlob serialize_materials(const MaterialRegistry& registry) const;
    // Deserialize topology from binary blob
    BodyTopology deserialize_topology(const SerializationBlob& blob) const;
    // Deserialize body state from binary blob
    BodyState deserialize_state(const SerializationBlob& blob) const;
    // Deserialize material registry from binary blob
    MaterialRegistry deserialize_materials(const SerializationBlob& blob) const;
private:
    static constexpr std::uint32_t kMagicTopology = 0x44454C54; // DELT
    static constexpr std::uint32_t kMagicState = 0x44454C53;    // DELS
    static constexpr std::uint32_t kMagicMaterials = 0x44454C4D; // DELM
};
} // namespace delirium::m6