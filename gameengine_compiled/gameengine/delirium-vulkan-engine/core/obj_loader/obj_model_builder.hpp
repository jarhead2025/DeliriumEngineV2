#pragma once

#include "delirium_m8/body_runtime.hpp"
#include "delirium_m8/types.hpp"
#include <string>
#include <vector>

namespace delirium {

/// Build a BodyRuntime from an OBJ file.
/// Pipeline: OBJ → quad decomposition → basis fitting → BodyTopology → BodyRuntime
struct ObjBuildResult {
    bool success = false;
    std::string error;
    int num_anchors = 0;
    int num_patches = 0;
    int num_quads = 0;
    int num_remaining_tris = 0;
};

/// Load an OBJ file and build a complete BodyRuntime with:
/// - Anchor positions and UV coordinates
/// - Patches with fitted basis types and coefficients
/// - Material colors derived from the OBJ's .mtl (or default gray)
///
/// The returned BodyRuntime is ready to be added to a MultiBodyManager.
delirium::m8::BodyRuntime build_model_from_obj(
    const std::string& obj_path,
    const std::string& texture_path,  // optional: path to the texture file
    ObjBuildResult* result = nullptr,
    std::string* error = nullptr);

/// Convert an ObjModel + QuadMesh into a BodyTopology with fitted bases.
/// This is the core function — build_model_from_obj calls this internally.
delirium::m8::BodyTopology build_topology_from_mesh(
    const std::vector<FitVertex>& vertices,
    const std::vector<std::array<std::uint32_t, 4>>& quads,
    std::vector<std::uint32_t>* out_basis_types = nullptr);

} // namespace delirium