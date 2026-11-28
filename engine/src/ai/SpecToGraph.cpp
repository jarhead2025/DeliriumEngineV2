// ============================================================================
// SpecToGraph.cpp — WS2 spec-to-constraint-graph mapper implementation
// ============================================================================
// M6 scaffolding: maps extracted DocumentSpec constraints to StructuralGraph
// constraints (bond distances, SE(3) alignment, mass tolerances).
// Resolves ambiguities. Confidence scoring per constraint.
// Part of v2.2Beta milestone M6 (WS2).
// ============================================================================

#include "delirium/ai/SpecToGraph.h"
#include "delirium/ai/StructuralGraph.h"
#include "delirium/ai/DocumentIngestor.h"

#include <sstream>
#include <iostream>
#include <algorithm>

namespace DeliRium {

SpecToGraph::SpecToGraph() : verbose_(false) {}

std::vector<GraphConstraint> SpecToGraph::Map(const DocumentSpec& spec) {
    std::vector<GraphConstraint> constraints;

    // Map each constraint type
    auto dims = MapDimensions(spec);
    auto mats = MapMaterials(spec);
    auto tols = MapTolerances(spec);
    auto loads = MapLoads(spec);

    constraints.insert(constraints.end(), dims.begin(), dims.end());
    constraints.insert(constraints.end(), mats.begin(), mats.end());
    constraints.insert(constraints.end(), tols.begin(), tols.end());
    constraints.insert(constraints.end(), loads.begin(), loads.end());

    // Resolve ambiguities
    constraints = ResolveAmbiguities(constraints);

    // Build report
    std::ostringstream oss;
    oss << "SpecToGraph mapping report:" << std::endl;
    oss << "  Dimensions: " << dims.size() << std::endl;
    oss << "  Materials: " << mats.size() << std::endl;
    oss << "  Tolerances: " << tols.size() << std::endl;
    oss << "  Loads: " << loads.size() << std::endl;
    oss << "  Total constraints: " << constraints.size() << std::endl;
    oss << "  Ambiguous (need confirmation): " << ambiguous_.size() << std::endl;
    mapping_report_ = oss.str();

    if (verbose_) {
        std::cout << mapping_report_ << std::endl;
    }

    return constraints;
}

std::vector<GraphConstraint> SpecToGraph::MapDimensions(const DocumentSpec& spec) {
    std::vector<GraphConstraint> constraints;
    for (const auto& c : spec.constraints) {
        if (c.name == "dimension") {
            GraphConstraint gc;
            gc.type = GraphConstraint::BondDistance;
            gc.element_a = 0; // Global element (placeholder)
            gc.element_b = -1;
            gc.name = "dim_" + std::to_string(constraints.size());
            gc.value = c.value;
            gc.tolerance = c.tolerance;
            gc.confidence = ComputeConfidence(c);
            gc.unit = c.unit;
            gc.source = c.name;
            constraints.push_back(gc);
        }
    }
    return constraints;
}

std::vector<GraphConstraint> SpecToGraph::MapMaterials(const DocumentSpec& spec) {
    std::vector<GraphConstraint> constraints;
    for (const auto& c : spec.constraints) {
        if (c.name == "material") {
            GraphConstraint gc;
            gc.type = GraphConstraint::MaterialProperty;
            gc.element_a = 0;
            gc.element_b = -1;
            gc.name = "mat_" + std::to_string(constraints.size());
            gc.value = 0.0; // Categorical
            gc.description = c.description;
            gc.confidence = ComputeConfidence(c);
            gc.unit = "—";
            gc.source = c.name;
            constraints.push_back(gc);
        }
    }
    return constraints;
}

std::vector<GraphConstraint> SpecToGraph::MapTolerances(const DocumentSpec& spec) {
    std::vector<GraphConstraint> constraints;
    for (const auto& c : spec.constraints) {
        if (c.name == "tolerance") {
            GraphConstraint gc;
            gc.type = GraphConstraint::BondDistance; // Tolerances apply to distances
            gc.element_a = 0;
            gc.element_b = -1;
            gc.name = "tol_" + std::to_string(constraints.size());
            gc.value = c.value;
            gc.tolerance = c.value; // Tolerance IS the tolerance
            gc.confidence = ComputeConfidence(c);
            gc.unit = c.unit;
            gc.source = c.name;
            constraints.push_back(gc);
        }
    }
    return constraints;
}

std::vector<GraphConstraint> SpecToGraph::MapLoads(const DocumentSpec& spec) {
    std::vector<GraphConstraint> constraints;
    for (const auto& c : spec.constraints) {
        if (c.name == "load") {
            GraphConstraint gc;
            gc.type = GraphConstraint::LoadBoundary;
            gc.element_a = 0;
            gc.element_b = -1;
            gc.name = "load_" + std::to_string(constraints.size());
            gc.value = c.value;
            gc.tolerance = c.tolerance;
            gc.confidence = ComputeConfidence(c);
            gc.unit = c.unit;
            gc.source = c.name;
            constraints.push_back(gc);
        }
    }
    return constraints;
}

double SpecToGraph::ComputeConfidence(const SpecConstraint& src) const {
    // Base confidence from source extraction
    double base = src.confidence;
    // Reduce confidence if tolerance is large relative to value
    if (src.tolerance > 0.0 && src.value > 0.0) {
        double rel_tol = src.tolerance / src.value;
        if (rel_tol > 0.1) base *= 0.8; // Large tolerance reduces confidence
    }
    return base;
}

std::vector<GraphConstraint> SpecToGraph::ResolveAmbiguities(
    const std::vector<GraphConstraint>& constraints) {
    ambiguous_.clear();
    std::vector<GraphConstraint> resolved = constraints;

    for (auto& c : resolved) {
        if (c.confidence < 0.5) {
            ambiguous_.push_back(c);
            if (verbose_) {
                std::cout << "[spec2graph] Ambiguous constraint: "
                          << c.name << " confidence=" << c.confidence << std::endl;
            }
        }
    }
    return resolved;
}

StructuralGraph* SpecToGraph::BuildGraph(const std::vector<GraphConstraint>& constraints) {
    StructuralGraph* graph = new StructuralGraph();
    // Placeholder: build graph from constraints
    // In production, this would:
    // 1. Create nodes for each element
    // 2. Add edges for bond distances
    // 3. Set constraints on edges/nodes
    return graph;
}

} // namespace delirium
