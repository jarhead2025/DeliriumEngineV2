#include "obj_loader/obj_model_builder.hpp"
#include "obj_loader/obj_loader.hpp"
#include "basis_fitter/basis_fitter.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <cmath>

namespace delirium {

// ─── Build BodyTopology from quad mesh ────────────────────────────
delirium::m8::BodyTopology build_topology_from_mesh(
    const std::vector<FitVertex>& vertices,
    const std::vector<std::array<std::uint32_t, 4>>& quads,
    std::vector<std::uint32_t>* out_basis_types) {
    
    delirium::m8::BodyTopology topology;
    
    // Create anchors (one per unique vertex in the quad mesh)
    std::unordered_map<std::uint32_t, delirium::m8::AnchorId> vertex_to_anchor;
    for (const auto& quad : quads) {
        for (auto vi : quad) {
            if (vertex_to_anchor.find(vi) == vertex_to_anchor.end()) {
                vertex_to_anchor[vi] = topology.add_anchor();
            }
        }
    }
    
    // Create patches with fitted basis types
    for (const auto& quad : quads) {
        const auto& A = vertices[quad[0]];
        const auto& B = vertices[quad[1]];
        const auto& C = vertices[quad[2]];
        const auto& D = vertices[quad[3]];
        
        // Fit the best basis for this quad
        auto fit = fit_basis(A, B, C, D, 8);
        
        // Map FitVertex to M8 types
        std::array<delirium::m8::AnchorId, 4> anchors = {
            vertex_to_anchor[quad[0]],
            vertex_to_anchor[quad[1]],
            vertex_to_anchor[quad[2]],
            vertex_to_anchor[quad[3]]
        };
        
        // Convert PatchCoefficients
        delirium::m8::PatchCoefficients m8_coeffs;
        m8_coeffs.scalars = fit.coefficients.scalars;
        for (int i = 0; i < 4; i++) {
            m8_coeffs.residuals[i] = {
                fit.coefficients.residuals[i][0],
                fit.coefficients.residuals[i][1],
                fit.coefficients.residuals[i][2]
            };
        }
        
        // Convert basis type
        delirium::m8::BasisType m8_basis;
        switch (fit.type) {
            case BasisType::Bilinear: m8_basis = delirium::m8::BasisType::Bilinear; break;
            case BasisType::Plano:    m8_basis = delirium::m8::BasisType::Plano; break;
            case BasisType::Cupula:   m8_basis = delirium::m8::BasisType::Cupula; break;
            case BasisType::Silla:    m8_basis = delirium::m8::BasisType::Silla; break;
            case BasisType::ValleU:   m8_basis = delirium::m8::BasisType::ValleU; break;
            case BasisType::ValleV:   m8_basis = delirium::m8::BasisType::ValleV; break;
            case BasisType::Eggbox:   m8_basis = delirium::m8::BasisType::Eggbox; break;
            case BasisType::Twist:    m8_basis = delirium::m8::BasisType::Twist; break;
        }
        
        if (out_basis_types) out_basis_types->push_back((std::uint32_t)fit.type);
        
        topology.add_patch(anchors, 0, m8_basis, m8_coeffs);
    }
    
    topology.finalize();
    
    std::cout << "[Topology] Built: " << topology.anchor_count() << " anchors, "
              << topology.patch_count() << " patches\n";
    
    return topology;
}

// ─── Full OBJ → BodyRuntime pipeline ──────────────────────────────
delirium::m8::BodyRuntime build_model_from_obj(
    const std::string& obj_path,
    const std::string& texture_path,
    ObjBuildResult* result,
    std::string* error) {
    
    if (result) result->success = false;
    
    // Step 1: Load OBJ
    ObjModel model = load_obj(obj_path, error);
    if (model.vertices.empty() || model.faces.empty()) {
        if (error) *error = "OBJ load failed or empty: " + obj_path;
        return delirium::m8::BodyTopology();
    }
    
    // Step 2: Decompose triangles into quads
    QuadMesh mesh = decompose_to_quads(model);
    if (mesh.quads.empty()) {
        if (error) *error = "No quads could be formed from mesh";
        return delirium::m8::BodyTopology();
    }
    
    if (result) {
        result->num_quads = (int)mesh.quads.size();
        result->num_remaining_tris = (int)mesh.remaining_tris.size();
    }
    
    // Step 3: Convert ObjVertex to FitVertex
    std::vector<FitVertex> fit_vertices;
    for (const auto& v : model.vertices) {
        fit_vertices.push_back({v.x, v.y, v.z, v.u, v.v});
    }
    
    // Step 4: Build topology with fitted bases
    std::vector<std::uint32_t> basis_types;
    auto topology = build_topology_from_mesh(fit_vertices, mesh.quads, &basis_types);
    
    if (result) {
        result->num_anchors = (int)topology.anchor_count();
        result->num_patches = (int)topology.patch_count();
    }
    
    // Step 5: Create rest positions and UVs from the original vertices
    std::vector<delirium::m8::Vec3> rest_positions;
    for (const auto& v : fit_vertices) {
        rest_positions.push_back({v.x, v.y, v.z});
    }
    
    // Step 6: Create runtime
    delirium::m8::BodyRuntime runtime(std::move(topology), 0);
    
    // Step 7: Register default material
    delirium::m8::MaterialRegistry registry;
    delirium::m8::MaterialProperties mat;
    mat.youngs_modulus = 1.0e6f;
    mat.poisson_ratio = 0.3f;
    mat.density = 1000.0f;
    mat.yield_strength = 5.0e5f;
    mat.r = 180; mat.g = 160; mat.b = 140;
    mat.roughness = 0.6f;
    mat.metalness = 0.1f;
    registry.register_material(0, mat);
    runtime.set_materials(std::move(registry));
    
    // Step 8: Initialize with rest positions
    std::string init_error;
    if (!runtime.initialize(rest_positions, &init_error)) {
        if (error) *error = "Runtime initialization failed: " + init_error;
        return delirium::m8::BodyTopology();
    }
    
    // Step 9: Set UVs on anchors
    auto& state = runtime.write_state();
    // Map quad vertices back to anchors to set UVs
    std::unordered_map<std::uint32_t, delirium::m8::AnchorId> vertex_to_anchor;
    std::uint32_t anchor_idx = 0;
    for (const auto& quad : mesh.quads) {
        for (auto vi : quad) {
            if (vertex_to_anchor.find(vi) == vertex_to_anchor.end()) {
                vertex_to_anchor[vi] = anchor_idx++;
            }
        }
    }
    
    for (const auto& [vi, aid] : vertex_to_anchor) {
        if (vi < fit_vertices.size() && aid < state.anchors.size()) {
            state.anchors[aid].uv_u = fit_vertices[vi].u;
            state.anchors[aid].uv_v = fit_vertices[vi].v;
        }
    }
    
    // Step 10: Center and scale the model
    Vec3 center{};
    for (const auto& p : rest_positions) {
        center.x += p.x; center.y += p.y; center.z += p.z;
    }
    center.x /= rest_positions.size();
    center.y /= rest_positions.size();
    center.z /= rest_positions.size();
    
    float max_extent = 0.0f;
    for (const auto& p : rest_positions) {
        float d = std::sqrt(
            (p.x - center.x)*(p.x - center.x) +
            (p.y - center.y)*(p.y - center.y) +
            (p.z - center.z)*(p.z - center.z));
        if (d > max_extent) max_extent = d;
    }
    
    // Scale to unit size (fits in [-1,1] box)
    if (max_extent > 1e-6f) {
        float scale = 2.0f / (2.0f * max_extent);
        for (auto& a : state.anchors) {
            a.rest_position = {
                (a.rest_position.x - center.x) * scale,
                (a.rest_position.y - center.y) * scale,
                (a.rest_position.z - center.z) * scale
            };
            a.position = a.rest_position;
        }
    }
    
    // Print basis distribution
    std::map<std::uint32_t, int> basis_count;
    for (auto bt : basis_types) basis_count[bt]++;
    std::cout << "[Basis Distribution]:\n";
    const char* names[] = {"Bilinear","Plano","Cupula","Silla","ValleU","ValleV","Eggbox","Twist"};
    for (auto& [bt, count] : basis_count) {
        if (bt < 8) std::cout << "  " << names[bt] << ": " << count << "\n";
    }
    
    if (result) result->success = true;
    
    std::cout << "[Model] Built from OBJ: " << result->num_anchors << " anchors, "
              << result->num_patches << " patches, texture: " 
              << (texture_path.empty() ? "none" : texture_path.c_str()) << "\n";
    
    return runtime;
}

} // namespace delirium