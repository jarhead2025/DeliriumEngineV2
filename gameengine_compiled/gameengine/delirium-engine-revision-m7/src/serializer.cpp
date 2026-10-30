#include "delirium_m7/serializer.hpp"
#include <cstring>
#include <stdexcept>

namespace delirium::m7 {

SerializationBlob Serializer::serialize_topology(const BodyTopology& topology) const {
    SerializationBlob blob;
    std::uint32_t magic = kMagicTopology;
    std::uint32_t anchor_count = static_cast<std::uint32_t>(topology.anchor_count());
    std::uint32_t patch_count = static_cast<std::uint32_t>(topology.patch_count());
    std::uint32_t seam_count = static_cast<std::uint32_t>(topology.welded_seams().size());

    std::size_t patch_size = sizeof(std::uint32_t)*4 + sizeof(std::uint32_t) + sizeof(std::uint8_t);
    std::size_t seam_size = sizeof(std::uint32_t)*2 + sizeof(float)*8 + sizeof(float);

    blob.data.resize(16 + patch_size * patch_count + seam_size * seam_count);
    std::uint8_t* p = blob.data.data();

    std::memcpy(p, &magic, 4); p += 4;
    std::memcpy(p, &anchor_count, 4); p += 4;
    std::memcpy(p, &patch_count, 4); p += 4;
    std::memcpy(p, &seam_count, 4); p += 4;

    for (std::uint32_t i = 0; i < patch_count; ++i) {
        const auto& patch = topology.patch(i);
        std::memcpy(p, patch.anchors.data(), sizeof(std::uint32_t)*4); p += sizeof(std::uint32_t)*4;
        std::memcpy(p, &patch.material, sizeof(std::uint32_t)); p += sizeof(std::uint32_t);
        std::memcpy(p, &patch.basis, sizeof(std::uint8_t)); p += sizeof(std::uint8_t);
    }

    for (std::uint32_t i = 0; i < seam_count; ++i) {
        const auto& seam = topology.welded_seams()[i];
        std::memcpy(p, &seam.patch_a, sizeof(std::uint32_t)*2); p += sizeof(std::uint32_t)*2;
        std::memcpy(p, &seam.a_u0, sizeof(float)*4); p += sizeof(float)*4;
        std::memcpy(p, &seam.b_u0, sizeof(float)*4); p += sizeof(float)*4;
        std::memcpy(p, &seam.tolerance, sizeof(float)); p += sizeof(float);
    }

    return blob;
}

SerializationBlob Serializer::serialize_state(const BodyState& state) const {
    SerializationBlob blob;
    std::uint32_t magic = kMagicState;
    std::uint32_t anchor_count = static_cast<std::uint32_t>(state.anchors.size());
    std::uint32_t patch_count = static_cast<std::uint32_t>(state.patches.size());

    std::size_t header_size = 28; // magic(4) + counts(8) + gen(8) + topo_gen(8)
    std::size_t anchor_size = sizeof(Vec3)*4 + sizeof(float)*2; // rest, pos, vel, modal_disp + temp, damage
    std::size_t patch_size = sizeof(float)*8 + sizeof(Vec3)*4 + sizeof(float)*5; // coeffs, residuals, strain, temp, damage, cum_strain, eff_stiff

    blob.data.resize(header_size + anchor_size * anchor_count + patch_size * patch_count);
    std::uint8_t* p = blob.data.data();

    std::memcpy(p, &magic, 4); p += 4;
    std::memcpy(p, &anchor_count, 4); p += 4;
    std::memcpy(p, &patch_count, 4); p += 4;
    std::memcpy(p, &state.generation, 8); p += 8;
    std::memcpy(p, &state.topology_generation, 8); p += 8;

    for (std::uint32_t i = 0; i < anchor_count; ++i) {
        const auto& a = state.anchors[i];
        std::memcpy(p, &a.rest_position, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(p, &a.position, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(p, &a.velocity, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(p, &a.modal_displacement, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(p, &a.temperature, sizeof(float)); p += sizeof(float);
        std::memcpy(p, &a.damage, sizeof(float)); p += sizeof(float);
    }

    for (std::uint32_t i = 0; i < patch_count; ++i) {
        const auto& ps = state.patches[i];
        std::memcpy(p, ps.coefficients.data(), sizeof(float)*8); p += sizeof(float)*8;
        for (int j = 0; j < 4; ++j) { std::memcpy(p, &ps.residuals[j], sizeof(Vec3)); p += sizeof(Vec3); }
        std::memcpy(p, &ps.strain_energy, sizeof(float)); p += sizeof(float);
        std::memcpy(p, &ps.temperature, sizeof(float)); p += sizeof(float);
        std::memcpy(p, &ps.damage, sizeof(float)); p += sizeof(float);
        std::memcpy(p, &ps.cumulative_plastic_strain, sizeof(float)); p += sizeof(float);
        std::memcpy(p, &ps.effective_stiffness, sizeof(float)); p += sizeof(float);
    }

    return blob;
}

SerializationBlob Serializer::serialize_materials(const MaterialRegistry& registry) const {
    SerializationBlob blob;
    std::uint32_t magic = kMagicMaterials;
    auto ids = registry.ids();
    std::uint32_t count = static_cast<std::uint32_t>(ids.size());

    std::size_t mat_size = sizeof(float)*6 + sizeof(std::uint8_t)*3 + sizeof(float)*2;
    blob.data.resize(8 + count * mat_size);
    std::uint8_t* p = blob.data.data();

    std::memcpy(p, &magic, 4); p += 4;
    std::memcpy(p, &count, 4); p += 4;

    for (MaterialId id : ids) {
        const auto& mat = registry.get(id);
        std::memcpy(p, &mat.youngs_modulus, sizeof(float)*4); p += sizeof(float)*4;
        std::memcpy(p, &mat.thermal_conductivity, sizeof(float)*2); p += sizeof(float)*2;
        std::memcpy(p, &mat.r, sizeof(std::uint8_t)*3); p += 3;
        std::memcpy(p, &mat.roughness, sizeof(float)*2); p += sizeof(float)*2;
    }
    return blob;
}

BodyTopology Serializer::deserialize_topology(const SerializationBlob& blob) const {
    if (blob.data.size() < 16) throw std::runtime_error("invalid topology blob");
    std::uint32_t magic;
    std::memcpy(&magic, blob.data.data(), 4);
    if (magic != kMagicTopology) throw std::runtime_error("bad topology magic");

    std::uint32_t anchor_count, patch_count, seam_count;
    std::memcpy(&anchor_count, blob.data.data() + 4, 4);
    std::memcpy(&patch_count, blob.data.data() + 8, 4);
    std::memcpy(&seam_count, blob.data.data() + 12, 4);

    BodyTopology topology;
    for (std::uint32_t i = 0; i < anchor_count; ++i) topology.add_anchor();

    const std::uint8_t* p = blob.data.data() + 16;
    for (std::uint32_t i = 0; i < patch_count; ++i) {
        std::array<AnchorId, 4> anchors;
        std::memcpy(anchors.data(), p, sizeof(std::uint32_t)*4); p += sizeof(std::uint32_t)*4;
        MaterialId mat; std::memcpy(&mat, p, sizeof(std::uint32_t)); p += sizeof(std::uint32_t);
        BasisType basis; std::memcpy(&basis, p, sizeof(std::uint8_t)); p += sizeof(std::uint8_t);
        topology.add_patch(anchors, mat, basis);
    }

    for (std::uint32_t i = 0; i < seam_count; ++i) {
        WeldedSeam seam;
        std::memcpy(&seam.patch_a, p, sizeof(std::uint32_t)*2); p += sizeof(std::uint32_t)*2;
        std::memcpy(&seam.a_u0, p, sizeof(float)*4); p += sizeof(float)*4;
        std::memcpy(&seam.b_u0, p, sizeof(float)*4); p += sizeof(float)*4;
        std::memcpy(&seam.tolerance, p, sizeof(float)); p += sizeof(float);
        topology.add_welded_seam(seam);
    }

    topology.finalize();
    return topology;
}

BodyState Serializer::deserialize_state(const SerializationBlob& blob) const {
    if (blob.data.size() < 28) throw std::runtime_error("invalid state blob");
    std::uint32_t magic;
    std::memcpy(&magic, blob.data.data(), 4);
    if (magic != kMagicState) throw std::runtime_error("bad state magic");

    std::uint32_t anchor_count, patch_count;
    std::memcpy(&anchor_count, blob.data.data() + 4, 4);
    std::memcpy(&patch_count, blob.data.data() + 8, 4);

    BodyState state;
    state.resize(anchor_count, patch_count);
    std::memcpy(&state.generation, blob.data.data() + 12, 8);
    std::memcpy(&state.topology_generation, blob.data.data() + 20, 8);

    const std::uint8_t* p = blob.data.data() + 28;
    for (std::uint32_t i = 0; i < anchor_count; ++i) {
        std::memcpy(&state.anchors[i].rest_position, p, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(&state.anchors[i].position, p, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(&state.anchors[i].velocity, p, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(&state.anchors[i].modal_displacement, p, sizeof(Vec3)); p += sizeof(Vec3);
        std::memcpy(&state.anchors[i].temperature, p, sizeof(float)); p += sizeof(float);
        std::memcpy(&state.anchors[i].damage, p, sizeof(float)); p += sizeof(float);
    }

    for (std::uint32_t i = 0; i < patch_count; ++i) {
        std::memcpy(state.patches[i].coefficients.data(), p, sizeof(float)*8); p += sizeof(float)*8;
        for (int j = 0; j < 4; ++j) { std::memcpy(&state.patches[i].residuals[j], p, sizeof(Vec3)); p += sizeof(Vec3); }
        std::memcpy(&state.patches[i].strain_energy, p, sizeof(float)); p += sizeof(float);
        std::memcpy(&state.patches[i].temperature, p, sizeof(float)); p += sizeof(float);
        std::memcpy(&state.patches[i].damage, p, sizeof(float)); p += sizeof(float);
        std::memcpy(&state.patches[i].cumulative_plastic_strain, p, sizeof(float)); p += sizeof(float);
        std::memcpy(&state.patches[i].effective_stiffness, p, sizeof(float)); p += sizeof(float);
    }

    return state;
}

MaterialRegistry Serializer::deserialize_materials(const SerializationBlob& blob) const {
    if (blob.data.size() < 8) throw std::runtime_error("invalid materials blob");
    std::uint32_t magic;
    std::memcpy(&magic, blob.data.data(), 4);
    if (magic != kMagicMaterials) throw std::runtime_error("bad materials magic");

    std::uint32_t count;
    std::memcpy(&count, blob.data.data() + 4, 4);

    MaterialRegistry registry;
    const std::uint8_t* p = blob.data.data() + 8;
    for (std::uint32_t i = 0; i < count; ++i) {
        MaterialProperties mat;
        std::memcpy(&mat.youngs_modulus, p, sizeof(float)*4); p += sizeof(float)*4;
        std::memcpy(&mat.thermal_conductivity, p, sizeof(float)*2); p += sizeof(float)*2;
        std::memcpy(&mat.r, p, sizeof(std::uint8_t)*3); p += 3;
        std::memcpy(&mat.roughness, p, sizeof(float)*2); p += sizeof(float)*2;
        registry.register_material(i, mat);
    }
    return registry;
}

} // namespace delirium::m7