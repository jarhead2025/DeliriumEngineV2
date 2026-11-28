// ============================================================================
// DocumentIngestor.h — WS2 document reconstruction ingestor header
// ============================================================================
// M6 scaffolding: parses PDF/CAD specs → structured DocumentSpec + feature tree.
// Converts STL/OBJ/PLY mesh imports → PSF body via surface fitting.
// Part of v2.2Beta milestone M6 (WS2).
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <map>
#include <array>
#include "delirium/core/Types.h"

namespace DeliRium {

// Forward declarations
class PSFBody;

// ============================================================================
// SpecConstraint — a single constraint extracted from a document
// ============================================================================
struct SpecConstraint {
    std::string name;          // Constraint name (e.g. "length", "mass", "material")
    std::string unit;          // Physical unit (e.g. "mm", "kg", "steel")
    double value;              // Specified value
    double tolerance;          // Tolerance (+/-)
    double confidence;         // Confidence score [0..1]
    std::string source;        // Source location in document (page, section)
    std::string description;   // Human-readable description

    SpecConstraint() : value(0.0), tolerance(0.0), confidence(0.0) {}
};

// ============================================================================
// FeatureNode — a node in the editable feature tree (for CAD formats)
// ============================================================================
struct FeatureNode {
    std::string type;          // Feature type (extrude, revolve, fillet, etc.)
    std::string name;          // Feature name
    std::map<std::string, double> params; // Parameters (dimensions, angles, etc.)
    std::vector<int> children; // Child feature indices
    int parent;                // Parent feature index (-1 = root)
    double confidence;         // Confidence score [0..1]

    FeatureNode() : parent(-1), confidence(0.0) {}
};

// ============================================================================
// DocumentSpec — structured specification extracted from a document
// ============================================================================
struct DocumentSpec {
    std::string title;                     // Document title
    std::string source_file;               // Source file path
    std::string format;                    // Format (pdf, step, iges, stl, obj)
    std::vector<SpecConstraint> constraints; // Extracted constraints
    std::vector<FeatureNode> features;     // Feature tree (CAD formats)
    std::map<std::string, std::string> metadata; // Key-value metadata
    double overall_confidence;             // Overall extraction confidence [0..1]
    std::string notes;                     // Notes / ambiguities

    DocumentSpec() : overall_confidence(0.0) {}
};

// ============================================================================
// MeshData — raw mesh data from STL/OBJ/PLY import
// ============================================================================
struct MeshData {
    std::vector<Vec3> vertices;   // Vertex positions
    std::vector<std::array<int,3>> triangles;    // Triangle indices
    std::vector<Vec3> normals;    // Vertex normals (optional)
    std::vector<std::array<float,2>> uvs;        // UV coordinates (optional)
    std::string name;               // Mesh name
    std::string material;           // Material name (optional)

    MeshData() = default;
};

// ============================================================================
// DocumentIngestor — WS2 document reconstruction ingestor
// ============================================================================
// Parses PDF/CAD specs → structured DocumentSpec + editable feature tree.
// Converts STL/OBJ/PLY mesh imports → PSF body via surface fitting.
//
// Usage:
//   DocumentIngestor ingestor;
//   DocumentSpec spec = ingestor.ParsePDF("spec.pdf");
//   DocumentSpec spec = ingestor.ParseCAD("model.step");
//   MeshData mesh = ingestor.ParseMesh("model.stl");
//   PSFBody* body = ingestor.BuildPSFBody(mesh);
// ============================================================================
class DocumentIngestor {
public:
    DocumentIngestor();
    ~DocumentIngestor() = default;

    // Parse a PDF document (text + vector graphics)
    DocumentSpec ParsePDF(const std::string& file_path);

    // Parse a CAD file (STEP/IGES) — placeholder for complex formats
    DocumentSpec ParseCAD(const std::string& file_path);

    // Parse a mesh file (STL/OBJ/PLY)
    MeshData ParseMesh(const std::string& file_path);

    // Build a PSFBody from mesh data via surface fitting
    PSFBody* BuildPSFBody(const MeshData& mesh);

    // Get the last parse error message
    const std::string& LastError() const { return last_error_; }

    // Check if PDF parsing is available (requires PDF library)
    bool IsPDFAvailable() const { return pdf_available_; }

    // Check if CAD parsing is available (requires CAD library)
    bool IsCADAvailable() const { return cad_available_; }

    // Set verbose mode
    void SetVerbose(bool verbose) { verbose_ = verbose; }

    // --- Public extraction wrappers (for testing) --------------------------
    std::vector<SpecConstraint> ExtractDimensionsPublic(const std::string& text) {
        return ExtractDimensions(text);
    }
    std::vector<SpecConstraint> ExtractMaterialsPublic(const std::string& text) {
        return ExtractMaterials(text);
    }
    std::vector<SpecConstraint> ExtractTolerancesPublic(const std::string& text) {
        return ExtractTolerances(text);
    }
    std::vector<SpecConstraint> ExtractLoadsPublic(const std::string& text) {
        return ExtractLoads(text);
    }

private:
    // Extract dimensions from parsed text
    std::vector<SpecConstraint> ExtractDimensions(const std::string& text);

    // Extract materials from parsed text
    std::vector<SpecConstraint> ExtractMaterials(const std::string& text);

    // Extract tolerances from parsed text
    std::vector<SpecConstraint> ExtractTolerances(const std::string& text);

    // Extract loads/boundary conditions from parsed text
    std::vector<SpecConstraint> ExtractLoads(const std::string& text);

    // Fit PSF surface to mesh vertices
    void FitSurfaceToMesh(PSFBody* body, const MeshData& mesh);

    // Validate extracted constraints
    bool ValidateConstraints(const DocumentSpec& spec);

    std::string last_error_;         // Last parse error message
    bool pdf_available_;             // Whether PDF parsing is available
    bool cad_available_;             // Whether CAD parsing is available
    bool verbose_;                   // Print parse messages
};

} // namespace delirium
