#include "obj_loader/obj_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstring>
#include <array>
#include <numeric>

namespace delirium {

// ─── Vec3 helpers (local, to avoid M8 dependency in loader) ─────────
static std::array<float, 3> vec3_sub(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
static std::array<float, 3> vec3_cross(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}
static float vec3_dot(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static float vec3_len(const std::array<float, 3>& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}
static std::array<float, 3> vec3_normalize(const std::array<float, 3>& v) {
    float l = vec3_len(v);
    if (l < 1e-8f) return {0, 0, 1};
    return {v[0]/l, v[1]/l, v[2]/l};
}

// ─── Load OBJ ──────────────────────────────────────────────────────
ObjModel load_obj(const std::string& path, std::string* error) {
    ObjModel model;

    std::ifstream file(path);
    if (!file.is_open()) {
        if (error) *error = "Cannot open file: " + path;
        return model;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            ObjVertex vert;
            iss >> vert.x >> vert.y >> vert.z;
            vert.u = 0.0f; vert.v = 0.0f;
            model.vertices.push_back(vert);
        }
        else if (token == "vt") {
            float u, v;
            iss >> u >> v;
            // Store UVs — we'll apply them when building faces below
            // For now store in a temp buffer indexed by vt order
            // (OBJ vt indices are 1-based)
            // We'll handle this in the face parsing below
        }
        else if (token == "f") {
            // Face: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            // or:   f v1/vt1 v2/vt2 v3/vt3
            // or:   f v1 v2 v3
            std::string vstr;
            std::vector<std::uint32_t> vi;
            std::vector<float> vu, vv;

            while (iss >> vstr) {
                std::istringstream viss(vstr);
                std::string part;
                std::vector<std::string> parts;
                while (std::getline(viss, part, '/')) {
                    parts.push_back(part);
                }

                if (!parts[0].empty()) {
                    int idx = std::stoi(parts[0]);
                    // OBJ uses 1-based (or negative for relative) indexing
                    if (idx < 0) idx = (int)model.vertices.size() + idx + 1;
                    vi.push_back((std::uint32_t)(idx - 1));
                }
                if (parts.size() > 1 && !parts[1].empty()) {
                    int idx = std::stoi(parts[1]);
                    if (idx < 0) idx = (int)model.vertices.size() + idx + 1;
                    // We'll look up the UV from stored UV data
                    vu.push_back(0.0f);
                    vv.push_back(0.0f);
                }
            }

            if (vi.size() >= 3) {
                ObjFace face;
                face.vertices = {(int32_t)vi[0], (int32_t)vi[1], (int32_t)vi[2]};
                model.faces.push_back(face);

                // If we have UVs for this face, apply them
                if (!vu.empty() && !vv.empty() && vu.size() >= 3) {
                    for (int c = 0; c < 3; c++) {
                        if (c < (int)vi.size() && c < (int)vu.size()) {
                            // UVs will be set from vt buffer below
                        }
                    }
                }
            }
        }
        else if (token == "mtllib") {
            iss >> model.mtllib_path;
        }
    }

    // Second pass: re-read to collect UVs properly
    file.clear();
    file.seekg(0, std::ios::beg);

    std::vector<std::array<float, 2>> vt_buffer;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "vt") {
            float u, v;
            iss >> u >> v;
            vt_buffer.push_back({u, v});
        }
    }

    // Third pass: parse faces with UV lookups
    file.clear();
    file.seekg(0, std::ios::beg);

    model.vertices.clear();
    model.faces.clear();

    std::vector<ObjVertex> raw_vertices;

    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            ObjVertex vert;
            iss >> vert.x >> vert.y >> vert.z;
            vert.u = 0.0f; vert.v = 0.0f;
            raw_vertices.push_back(vert);
        }
        else if (token == "f") {
            std::string vstr;
            std::vector<std::uint32_t> vi;
            std::vector<std::uint32_t> vti;

            while (iss >> vstr) {
                std::istringstream viss(vstr);
                std::string part;
                std::vector<std::string> parts;
                while (std::getline(viss, part, '/')) {
                    parts.push_back(part);
                }

                if (!parts[0].empty()) {
                    int idx = std::stoi(parts[0]);
                    if (idx < 0) idx = (int)raw_vertices.size() + idx + 1;
                    vi.push_back((std::uint32_t)(idx - 1));
                }
                if (parts.size() > 1 && !parts[1].empty()) {
                    int idx = std::stoi(parts[1]);
                    if (idx < 0) idx = (int)vt_buffer.size() + idx + 1;
                    vti.push_back((std::uint32_t)(idx - 1));
                }
            }

            if (vi.size() >= 3 && vi.size() == vti.size()) {
                // Create unique vertices with UVs (split at UV boundaries)
                std::vector<std::uint32_t> face_vi;
                for (size_t c = 0; c < vi.size(); c++) {
                    // Find or create a vertex with this position+UV combo
                    uint32_t uv_idx = vti[c];
                    float u = (uv_idx < vt_buffer.size()) ? vt_buffer[uv_idx][0] : 0.0f;
                    float v_uv = (uv_idx < vt_buffer.size()) ? vt_buffer[uv_idx][1] : 0.0f;

                    // Check if we already have this vertex+UV combo
                    bool found = false;
                    for (size_t k = 0; k < model.vertices.size(); k++) {
                        if (std::fabs(model.vertices[k].x - raw_vertices[vi[c]].x) < 1e-6f &&
                            std::fabs(model.vertices[k].y - raw_vertices[vi[c]].y) < 1e-6f &&
                            std::fabs(model.vertices[k].z - raw_vertices[vi[c]].z) < 1e-6f &&
                            std::fabs(model.vertices[k].u - u) < 1e-6f &&
                            std::fabs(model.vertices[k].v - v_uv) < 1e-6f) {
                            face_vi.push_back((std::uint32_t)k);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        ObjVertex vert = raw_vertices[vi[c]];
                        vert.u = u;
                        vert.v = v_uv;
                        face_vi.push_back((std::uint32_t)model.vertices.size());
                        model.vertices.push_back(vert);
                    }
                }

                // Triangulate if needed (OBJ can have n-gons)
                for (size_t t = 1; t < face_vi.size() - 1; t++) {
                    ObjFace face;
                    face.vertices = {(int32_t)face_vi[0], (int32_t)face_vi[t], (int32_t)face_vi[t+1]};
                    model.faces.push_back(face);
                }
            } else if (vi.size() >= 3) {
                // No UVs — use raw vertices directly
                ObjFace face;
                face.vertices = {(int32_t)vi[0], (int32_t)vi[1], (int32_t)vi[2]};
                model.faces.push_back(face);
            }
        }
    }

    if (model.vertices.empty() && model.faces.empty() && error) {
        *error = "No vertices or faces found in OBJ";
    }

    std::cout << "[OBJ] Loaded: " << model.vertices.size() << " vertices, "
              << model.faces.size() << " faces\n";

    return model;
}

// ─── Triangle-to-Quad Decomposition ────────────────────────────────
QuadMesh decompose_to_quads(const ObjModel& model, float planarity_threshold) {
    QuadMesh result;

    if (model.faces.empty()) return result;

    // Build edge-to-face adjacency map
    // Edge key: sorted pair of vertex indices
    using EdgeKey = std::array<std::uint64_t, 2>;
    auto make_edge = [](std::uint32_t a, std::uint32_t b) -> EdgeKey {
        return {(std::uint64_t)std::min(a, b), (std::uint64_t)std::max(a, b)};
    };

    std::unordered_map<EdgeKey, std::pair<std::uint32_t, bool>, 
                       struct EdgeHash {
        std::size_t operator()(const EdgeKey& e) const {
            return std::hash<std::uint64_t>()(e[0]) ^ (std::hash<std::uint64_t>()(e[1]) << 1);
        }
    }> edge_to_face;

    for (std::uint32_t fi = 0; fi < (std::uint32_t)model.faces.size(); fi++) {
        const auto& f = model.faces[fi];
        for (int i = 0; i < 3; i++) {
            std::uint32_t a = (std::uint32_t)f.vertices[i];
            std::uint32_t b = (std::uint32_t)f.vertices[(i + 1) % 3];
            EdgeKey edge = make_edge(a, b);
            auto it = edge_to_face.find(edge);
            if (it == edge_to_face.end()) {
                edge_to_face[edge] = {fi, false};
            } else {
                it->second.second = true;  // edge is shared
            }
        }
    }

    std::unordered_set<std::uint32_t> used_faces;

    for (auto& [edge, data] : edge_to_face) {
        if (!data.second) continue;  // edge not shared, can't form quad

        // Find the two triangles sharing this edge
        std::uint32_t fi = data.first;
        if (used_faces.count(fi)) continue;

        const auto& f1 = model.faces[fi];
        std::uint32_t f2_idx = UINT32_MAX;

        // Find the other face sharing this edge
        for (std::uint32_t fj = 0; fj < (std::uint32_t)model.faces.size(); fj++) {
            if (fj == fi) continue;
            const auto& f2 = model.faces[fj];
            for (int i = 0; i < 3; i++) {
                std::uint32_t a = (std::uint32_t)f2.vertices[i];
                std::uint32_t b = (std::uint32_t)f2.vertices[(i + 1) % 3];
                if (make_edge(a, b) == edge) {
                    f2_idx = fj;
                    break;
                }
            }
            if (f2_idx != UINT32_MAX) break;
        }

        if (f2_idx == UINT32_MAX || used_faces.count(f2_idx)) continue;

        // Get the 4 unique vertices
        const auto& f1f = model.faces[fi];
        const auto& f2f = model.faces[f2_idx];

        std::array<std::uint32_t, 6> all_v = {
            (std::uint32_t)f1f.vertices[0], (std::uint32_t)f1f.vertices[1], (std::uint32_t)f1f.vertices[2],
            (std::uint32_t)f2f.vertices[0], (std::uint32_t)f2f.vertices[1], (std::uint32_t)f2f.vertices[2]
        };
        std::sort(all_v.begin(), all_v.end());
        auto last = std::unique(all_v.begin(), all_v.end());

        if (std::distance(all_v.begin(), last) != 4) continue;  // not a clean quad

        // Check planarity: compute normals of both triangles
        auto get_normal = [&](std::uint32_t vi0, std::uint32_t vi1, std::uint32_t vi2) {
            auto& v0 = model.vertices[vi0];
            auto& v1 = model.vertices[vi1];
            auto& v2 = model.vertices[vi2];
            std::array<float, 3> e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
            std::array<float, 3> e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
            return vec3_normalize(vec3_cross(e1, e2));
        };

        auto n1 = get_normal((std::uint32_t)f1f.vertices[0], (std::uint32_t)f1f.vertices[1], (std::uint32_t)f1f.vertices[2]);
        auto n2 = get_normal((std::uint32_t)f2f.vertices[0], (std::uint32_t)f2f.vertices[1], (std::uint32_t)f2f.vertices[2]);

        float angle = std::fabs(vec3_dot(n1, n2));
        if (angle < (1.0f - planarity_threshold)) continue;  // too much dihedral angle

        // Order vertices for bilinear patch: bottom-left, bottom-right, top-left, top-right
        // Use the shared edge as the diagonal
        std::uint32_t shared_a = (std::uint32_t)edge[0];
        std::uint32_t shared_b = (std::uint32_t)edge[1];

        // The two unique vertices (one from each triangle)
        std::uint32_t unique_a = UINT32_MAX, unique_b = UINT32_MAX;
        for (auto v : {all_v[0], all_v[1], all_v[2], all_v[3]}) {
            if (v != shared_a && v != shared_b) {
                if (unique_a == UINT32_MAX) unique_a = v;
                else unique_b = v;
            }
        }

        // Order by Y then X for consistent winding
        QuadMesh::Quad quad;
        auto& va = model.vertices[shared_a];
        auto& vb = model.vertices[shared_b];
        auto& vu = model.vertices[unique_a];
        auto& vuu = model.vertices[unique_b];

        // Determine winding: shared edge is one diagonal, unique vertices are the other
        // Order: A(bottom-left), B(bottom-right), C(top-left), D(top-right)
        if (va.y <= vb.y) {
            quad.vertices[0] = shared_a;  // A: bottom-left of shared edge
            quad.vertices[1] = shared_b;  // B: bottom-right of shared edge
        } else {
            quad.vertices[0] = shared_b;
            quad.vertices[1] = shared_a;
        }
        // Top vertices
        if (vu.y <= vuu.y) {
            quad.vertices[2] = unique_a;  // C: top-left
            quad.vertices[3] = unique_b;  // D: top-right
        } else {
            quad.vertices[2] = unique_b;
            quad.vertices[3] = unique_a;
        }
        quad.material = 0;

        result.quads.push_back(quad);
        used_faces.insert(fi);
        used_faces.insert(f2_idx);
    }

    // Remaining triangles that couldn't be merged
    for (std::uint32_t fi = 0; fi < (std::uint32_t)model.faces.size(); fi++) {
        if (!used_faces.count(fi)) {
            result.remaining_tris.push_back(fi);
        }
    }

    std::cout << "[Mesh] Decomposed: " << result.quads.size() << " quads, "
              << result.remaining_tris.size() << " remaining triangles\n";

    return result;
}

} // namespace delirium