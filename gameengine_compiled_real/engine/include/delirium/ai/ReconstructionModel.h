// =============================================================================
// ReconstructionModel.h - Future seam for the intelligent reconstructor.
// Contract: input = semantic request (+ optional reference images /
// point clouds), output = the SAME frozen 64-byte records the fixed
// pipeline emits. Because patches/K/seams ARE the engine's native
// degrees of freedom, a model that learns to emit them gets live
// physics, LOD, sewing and rendering for free.
// =============================================================================
#pragma once
#include <vector>
#include <string>
#include "../core/Types.h"

namespace DeliRium {

struct ModelOutput {
    std::vector<AnchorRecord>  anchors;
    std::vector<SurfaceRecord> surfaces;
    std::vector<BoneRecord>    bones;       // optional articulation
    std::vector<float>         mode_hints;  // suggested deformation modes
};

class ReconstructionModel {
public:
    // Loads the RMW1 weights container (magic + version + weights +
    // mode hints). The NEURAL backend that consumes the weights is
    // still TBD; the format reader is real and exercised.
    bool Load(const std::string& weights_path);
    // Deterministic placeholder: emits a valid unit-quad body for any
    // semantic request. The future backend consumes `reference_blob`
    // (an in-memory PSF container / point cloud) instead.
    bool Infer(const std::string& semantic,
               const std::vector<uint8_t>& reference_blob,
               ModelOutput& out);
    // Validation gate before records enter the engine: layout sizes,
    // anchor index bounds, seam consistency, NaN-free coefficients.
    // Backend-agnostic: runs on ANY ModelOutput.
    static bool Validate(const ModelOutput& out, std::string& why);
};

} // namespace DeliRium
