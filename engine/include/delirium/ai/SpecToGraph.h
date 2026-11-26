// ============================================================================
// SpecToGraph.h — WS2 spec-to-constraint-graph mapper header
// ============================================================================
// M6 scaffolding: maps extracted DocumentSpec constraints to StructuralGraph
// constraints (bond distances, SE(3) alignment, mass tolerances).
// Resolves ambiguities. Confidence scoring per constraint.
// Part of v2.2Beta milestone M6 (WS2).
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <map>

namespace delirium {

// Forward declarations
class StructuralGraph;
class SpecConstraint;
class DocumentSpec;

// ============================================================================
// GraphConstraint — a constraint mapped to the structural graph
// ============================================================================
struct GraphConstraint {
    enum Type {
        BondDistance,       // Distance between two elements
        SE3Alignment,       // Relative pose between two elements
        MassTolerance,      // Mass constraint on element
        VolumeTolerance,    // Volume constraint on element
        MaterialProperty,   // Material property constraint
        LoadBoundary,       // Load/boundary condition
        Custom              // User-defined constraint
    };

    Type type;
    int element_a;          // First element index (-1 = global)
    int element_b;          // Second element index (-1 = global)
    std::string name;       // Constraint name
    double value;           // Constraint value
    double tolerance;       // Tolerance (+/-)
    double confidence;      // Confidence score [0..1]
    std::string unit;       // Physical unit
    std::string source;     // Source constraint name from DocumentSpec

    GraphConstraint() : type(BondDistance), element_a(-1), element_b(-1),
                        value(0.0), tolerance(0.0), confidence(0.0) {}
};

// ============================================================================
// SpecToGraph — maps DocumentSpec constraints to StructuralGraph constraints
// ============================================================================
// Resolves ambiguities in extracted specs and maps them to graph constraints.
// Each constraint gets a confidence score based on extraction quality.
//
// Usage:
//   SpecToGraph mapper;
//   std::vector<GraphConstraint> graph_constraints = mapper.Map(spec);
//   StructuralGraph* graph = mapper.BuildGraph(graph_constraints);
// ============================================================================
class SpecToGraph {
public:
    SpecToGraph();
    ~SpecToGraph() = default;

    // Map DocumentSpec constraints to graph constraints
    std::vector<GraphConstraint> Map(const DocumentSpec& spec);

    // Build a StructuralGraph from mapped constraints
    StructuralGraph* BuildGraph(const std::vector<GraphConstraint>& constraints);

    // Resolve ambiguities in constraints
    // Returns resolved constraints with updated confidence scores
    std::vector<GraphConstraint> ResolveAmbiguities(
        const std::vector<GraphConstraint>& constraints);

    // Get ambiguity report (constraints that need user confirmation)
    const std::vector<GraphConstraint>& AmbiguousConstraints() const { return ambiguous_; }

    // Get mapping report
    const std::string& MappingReport() const { return mapping_report_; }

    // Set verbose mode
    void SetVerbose(bool verbose) { verbose_ = verbose; }

private:
    // Map dimension constraints to bond distances
    std::vector<GraphConstraint> MapDimensions(const DocumentSpec& spec);

    // Map material constraints to material properties
    std::vector<GraphConstraint> MapMaterials(const DocumentSpec& spec);

    // Map tolerance constraints to tolerances
    std::vector<GraphConstraint> MapTolerances(const DocumentSpec& spec);

    // Map load constraints to boundary conditions
    std::vector<GraphConstraint> MapLoads(const DocumentSpec& spec);

    // Compute confidence for a mapped constraint
    double ComputeConfidence(const SpecConstraint& src) const;

    std::vector<GraphConstraint> ambiguous_; // Constraints needing user confirmation
    std::string mapping_report_;             // Mapping report text
    bool verbose_;                           // Print mapping messages
};

} // namespace delirium
