#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>

namespace delirium {

/// Raw OBJ data before decomposition into patches
struct ObjVertex {
    float x, y, z;
    float u, v;
};

struct ObjFace {
    std::array<std::int32_t, 3> vertices;  // OBJ uses 1-based indexing
};

struct ObjMaterial {
    std::string name;
    float ambient[3] = {0.2f, 0.2f, 0.2f};
    float diffuse[3] = {0.8f, 0.8f, 0.8f};
    float specular[3] = {1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;
    std::string texture_path;
};

struct ObjModel {
    std::vector<ObjVertex> vertices;
    std::vector<ObjFace> faces;
    std::vector<ObjMaterial> materials;
    std::string texture_path;
    std::string mtllib_path;
};

/// Load a .obj file (and optional .mtl) into raw vertex/face data.
/// Returns false on parse error; sets `error` message.
ObjModel load_obj(const std::string& path, std::string* error = nullptr);

/// Merge adjacent triangles into quads for bilinear patch decomposition.
/// Two triangles form a quad if they share an edge and the resulting quad
/// is roughly planar (dihedral angle < threshold).
struct QuadMesh {
    struct Quad {
        std::array<std::uint32_t, 4> vertices;  // indices into ObjModel::vertices
        std::uint32_t material = 0;
    };
    std::vector<Quad> quads;
    std::vector<std::uint32_t> remaining_tris;  // triangles that couldn't be merged
};

QuadMesh decompose_to_quads(const ObjModel& model, float planarity_threshold = 0.1f);

} // namespace delirium