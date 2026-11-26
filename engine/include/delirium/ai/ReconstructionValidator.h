// ============================================================================
// ReconstructionValidator.h — WS2 reconstruction validator header
// ============================================================================
// M6 scaffolding: bakes body → measures mass, volume, inertia tensor, bbox →
// compares against document specs → flags deviations exceeding tolerance.
// Outputs validation report with pass/fail per spec.
// Part of v2.2Beta milestone M6 (WS2).
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <map>
#include <array>

namespace delirium {

// Forward declarations
class PSFBody;
class DocumentSpec;
class SpecConstraint;

// ============================================================================
// ValidationResult — result of validating one constraint
// ============================================================================
struct ValidationResult {
    std::string constraint_name;   // Constraint being validated
    std::string measured_name;     // What was measured (e.g. "mass", "volume")
    double spec_value;             // Specified value
    double measured_value;         // Measured value
    double tolerance;              // Allowed tolerance
    double deviation;              // |measured - spec|
    bool passed;                   // Whether deviation <= tolerance
    double confidence;             // Confidence in measurement
    std::string unit;              // Physical unit
    std::string notes;             // Notes / warnings

    ValidationResult() : spec_value(0.0), measured_value(0.0),
                         tolerance(0.0), deviation(0.0),
                         passed(false), confidence(0.0) {}
};

// ============================================================================
// ValidationReport — full validation report for a reconstruction
// ============================================================================
struct ValidationReport {
    std::string source_file;       // Source document file
    std::string body_name;         // Reconstructed body name
    std::vector<ValidationResult> results; // Per-constraint results
    int total_constraints;         // Total constraints validated
    int passed_constraints;        // Constraints that passed
    int failed_constraints;        // Constraints that failed
    double overall_pass_rate;      // Pass rate [0..1]
    bool all_passed;               // Whether all constraints passed
    std::string summary;           // Summary text
    std::string recommendations;   // Recommendations for fixing failures

    ValidationReport() : total_constraints(0), passed_constraints(0),
                         failed_constraints(0), overall_pass_rate(0.0),
                         all_passed(false) {}
};

// ============================================================================
// ReconstructionValidator — validates reconstructed body against document specs
// ============================================================================
// Bakes body → measures physical properties → compares against DocumentSpec
// constraints → flags deviations exceeding tolerance.
//
// Usage:
//   ReconstructionValidator validator;
//   ValidationReport report = validator.Validate(body, spec);
//   if (report.all_passed) { /* reconstruction successful */ }
// ============================================================================
class ReconstructionValidator {
public:
    ReconstructionValidator();
    ~ReconstructionValidator() = default;

    // Validate a reconstructed body against document specs
    ValidationReport Validate(PSFBody* body, const DocumentSpec& spec);

    // Measure physical properties of a body
    // Returns map of property name → measured value
    std::map<std::string, double> MeasureProperties(PSFBody* body) const;

    // Measure mass
    double MeasureMass(PSFBody* body) const;

    // Measure volume
    double MeasureVolume(PSFBody* body) const;

    // Measure inertia tensor (returns diagonal components)
    std::array<double, 3> MeasureInertia(PSFBody* body) const;

    // Measure bounding box
    void MeasureBoundingBox(PSFBody* body, float3& min, float3& max) const;

    // Get last error message
    const std::string& LastError() const { return last_error_; }

    // Set verbose mode
    void SetVerbose(bool verbose) { verbose_ = verbose; }

    // Set default tolerance multiplier (applied to spec tolerances)
    void SetToleranceMultiplier(double mult) { tolerance_mult_ = mult; }

private:
    // Validate a single constraint against measured properties
    ValidationResult ValidateConstraint(
        const SpecConstraint& spec,
        const std::map<std::string, double>& measured) const;

    // Generate summary text
    std::string GenerateSummary(const ValidationReport& report) const;

    // Generate recommendations for failed constraints
    std::string GenerateRecommendations(const ValidationReport& report) const;

    std::string last_error_;       // Last error message
    bool verbose_;                 // Print validation messages
    double tolerance_mult_;        // Tolerance multiplier (default 1.0)
};

} // namespace delirium
