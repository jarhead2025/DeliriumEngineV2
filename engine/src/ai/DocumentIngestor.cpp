// ============================================================================
// DocumentIngestor.cpp — WS2 document reconstruction ingestor implementation
// ============================================================================
// M6 scaffolding: parses PDF/CAD specs → structured DocumentSpec + feature tree.
// Converts STL/OBJ/PLY mesh imports → PSF body via surface fitting.
// Part of v2.2Beta milestone M6 (WS2).
// ============================================================================

#include "delirium/ai/DocumentIngestor.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/Types.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <cmath>

namespace DeliRium {

DocumentIngestor::DocumentIngestor()
    : pdf_available_(false), cad_available_(false), verbose_(false) {
    // Placeholder: check for PDF/CAD library availability
    // In production, this would check for:
    // - PDF library (e.g. poppler, pdfium)
    // - CAD library (e.g. OpenCASCADE for STEP/IGES)
}

DocumentSpec DocumentIngestor::ParsePDF(const std::string& file_path) {
    DocumentSpec spec;
    spec.source_file = file_path;
    spec.format = "pdf";

    if (!pdf_available_) {
        last_error_ = "PDF parsing not available (library not linked)";
        spec.overall_confidence = 0.0;
        return spec;
    }

    // Placeholder: actual PDF parsing would:
    // 1. Extract text content
    // 2. Extract vector graphics (dimensions, shapes)
    // 3. Parse dimensions, materials, tolerances, loads
    // 4. Build DocumentSpec with constraints + metadata

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        last_error_ = "Failed to open PDF file: " + file_path;
        spec.overall_confidence = 0.0;
        return spec;
    }

    // Read file content (placeholder — actual PDF parsing deferred)
    std::stringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();

    // Extract constraints from text (placeholder)
    spec.constraints = ExtractDimensions(content);
    spec.constraints.insert(spec.constraints.end(),
                            ExtractMaterials(content).begin(),
                            ExtractMaterials(content).end());
    spec.constraints.insert(spec.constraints.end(),
                            ExtractTolerances(content).begin(),
                            ExtractTolerances(content).end());
    spec.constraints.insert(spec.constraints.end(),
                            ExtractLoads(content).begin(),
                            ExtractLoads(content).end());

    // Compute overall confidence
    double total_conf = 0.0;
    for (const auto& c : spec.constraints) {
        total_conf += c.confidence;
    }
    spec.overall_confidence = spec.constraints.empty() ? 0.0 : total_conf / spec.constraints.size();

    if (verbose_) {
        std::cout << "[ingest] Parsed PDF: " << spec.constraints.size()
                  << " constraints, confidence=" << spec.overall_confidence << std::endl;
    }

    return spec;
}

DocumentSpec DocumentIngestor::ParseCAD(const std::string& file_path) {
    DocumentSpec spec;
    spec.source_file = file_path;
    spec.format = "cad";

    if (!cad_available_) {
        last_error_ = "CAD parsing not available (library not linked)";
        spec.overall_confidence = 0.0;
        return spec;
    }

    // Placeholder: actual CAD parsing would:
    // 1. Parse STEP/IGES geometry
    // 2. Build feature tree (extrude, revolve, fillet, etc.)
    // 3. Extract parameters (dimensions, angles, radii)
    // 4. Build DocumentSpec with constraints + feature tree

    spec.overall_confidence = 0.0;
    return spec;
}

MeshData DocumentIngestor::ParseMesh(const std::string& file_path) {
    MeshData mesh;
    std::string ext;
    size_t dot = file_path.find_last_of('.');
    if (dot != std::string::npos) {
        ext = file_path.substr(dot + 1);
        for (auto& c : ext) c = static_cast<char>(::tolower(c));
    }

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        last_error_ = "Failed to open mesh file: " + file_path;
        return mesh;
    }

    if (ext == "stl") {
        // Parse ASCII STL (placeholder — binary STL deferred)
        std::string line;
        std::getline(file, line); // "solid ..."
        mesh.name = line.substr(6, line.find_last_not_of(" \t\r\n") - 6 + 1);

        while (std::getline(file, line)) {
            size_t fpos = line.find("facet");
            if (fpos != std::string::npos) {
                // Each facet = one triangle
                mesh.triangles.push_back({0, 0, 0}); // placeholder connectivity
            }
            size_t vpos = line.find("vertex");
            if (vpos != std::string::npos) {
                float x, y, z;
                std::istringstream iss(line.substr(vpos + 6));
                iss >> x >> y >> z;
                mesh.vertices.push_back({x, y, z});
            }
        }
    } else if (ext == "obj") {
        // Parse OBJ (placeholder)
        std::string line;
        while (std::getline(file, line)) {
            if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
                float x, y, z;
                std::istringstream iss(line.substr(2));
                iss >> x >> y >> z;
                mesh.vertices.push_back({x, y, z});
            } else if (line[0] == 'f') {
                // Parse face indices (placeholder — handles "f v1 v2 v3")
                std::istringstream iss(line.substr(2));
                int v1, v2, v3;
                char slash;
                if (iss >> v1 >> slash >> v2 >> slash >> v3) {
                    mesh.triangles.push_back({v1 - 1, v2 - 1, v3 - 1});
                }
            }
        }
    } else if (ext == "ply") {
        // Parse PLY (placeholder)
        std::string line;
        bool in_header = true;
        int vertex_count = 0;
        while (std::getline(file, line)) {
            if (in_header) {
                if (line.find("element vertex") == 0) {
                    std::istringstream iss(line.substr(14));
                    iss >> vertex_count;
                } else if (line == "end_header") {
                    in_header = false;
                }
            } else {
                float x, y, z;
                std::istringstream iss(line);
                iss >> x >> y >> z;
                mesh.vertices.push_back({x, y, z});
            }
        }
    }

    if (verbose_) {
        std::cout << "[ingest] Parsed mesh: " << mesh.vertices.size()
                  << " vertices, " << mesh.triangles.size()
                  << " triangles" << std::endl;
    }
    return mesh;
}

PSFBody* DocumentIngestor::BuildPSFBody(const MeshData& mesh) {
    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        last_error_ = "Mesh has no geometry";
        return nullptr;
    }

    PSFBody* body = new PSFBody();
    // PSFBody has no name member — name lives on GameObject. Skip.

    // Fit PSF surface to mesh vertices
    FitSurfaceToMesh(body, mesh);

    if (verbose_) {
        std::cout << "[ingest] Built PSFBody from mesh" << std::endl;
    }
    return body;
}

std::vector<SpecConstraint> DocumentIngestor::ExtractDimensions(const std::string& text) {
    std::vector<SpecConstraint> constraints;
    // Placeholder: regex-based dimension extraction
    // Pattern: number + unit (e.g. "10.5 mm", "2.3 m")
    std::regex dim_regex(R"((\d+\.?\d*)\s*(mm|cm|m|in|ft))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), dim_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        SpecConstraint c;
        c.name = "dimension";
        c.value = std::stod((*it)[1].str());
        c.unit = (*it)[2].str();
        c.confidence = 0.8; // Placeholder confidence
        constraints.push_back(c);
    }
    return constraints;
}

std::vector<SpecConstraint> DocumentIngestor::ExtractMaterials(const std::string& text) {
    std::vector<SpecConstraint> constraints;
    // Placeholder: keyword-based material extraction
    std::vector<std::string> materials = {"steel", "aluminum", "titanium", "copper", "plastic"};
    std::string lower_text = text;
    for (auto& c : lower_text) c = static_cast<char>(::tolower(c));

    for (const auto& mat : materials) {
        if (lower_text.find(mat) != std::string::npos) {
            SpecConstraint c;
            c.name = "material";
            c.value = 0.0; // Material is categorical
            c.unit = "—";
            c.description = mat;
            c.confidence = 0.9;
            constraints.push_back(c);
        }
    }
    return constraints;
}

std::vector<SpecConstraint> DocumentIngestor::ExtractTolerances(const std::string& text) {
    std::vector<SpecConstraint> constraints;
    // Placeholder: tolerance extraction
    std::regex tol_regex(R"(\+/-\s*(\d+\.?\d*))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), tol_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        SpecConstraint c;
        c.name = "tolerance";
        c.value = std::stod((*it)[1].str());
        c.unit = "mm";
        c.confidence = 0.7;
        constraints.push_back(c);
    }
    return constraints;
}

std::vector<SpecConstraint> DocumentIngestor::ExtractLoads(const std::string& text) {
    std::vector<SpecConstraint> constraints;
    // Placeholder: load extraction
    std::regex load_regex(R"((\d+\.?\d*)\s*(N|kN|Pa|MPa))");
    auto begin = std::sregex_iterator(text.begin(), text.end(), load_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        SpecConstraint c;
        c.name = "load";
        c.value = std::stod((*it)[1].str());
        c.unit = (*it)[2].str();
        c.confidence = 0.75;
        constraints.push_back(c);
    }
    return constraints;
}

void DocumentIngestor::FitSurfaceToMesh(PSFBody* body, const MeshData& mesh) {
    // Placeholder: surface fitting algorithm
    // In production, this would:
    // 1. Compute surface normals from mesh triangles
    // 2. Fit PSF patches to mesh surface
    // 3. Bake anchors + coefficients
    // 4. Compute mass/volume/inertia from fitted surface
}

bool DocumentIngestor::ValidateConstraints(const DocumentSpec& spec) {
    // Validate extracted constraints
    // Check for:
    // - Missing required constraints (dimensions, material)
    // - Inconsistent constraints (overlapping tolerances)
    // - Low confidence constraints (< 0.5)
    bool valid = true;
    bool has_dim = false, has_mat = false;
    for (const auto& c : spec.constraints) {
        if (c.name == "dimension") has_dim = true;
        if (c.name == "material") has_mat = true;
        if (c.confidence < 0.5) {
            // Low confidence — flag for user confirmation
            if (verbose_) {
                std::cout << "[ingest] Low confidence constraint: "
                          << c.name << " = " << c.value << " " << c.unit
                          << " (confidence=" << c.confidence << ")" << std::endl;
            }
        }
    }
    if (!has_dim) {
        last_error_ = "No dimensions extracted from document";
        valid = false;
    }
    if (!has_mat) {
        last_error_ = "No material extracted from document";
        valid = false;
    }
    return valid;
}

} // namespace delirium
