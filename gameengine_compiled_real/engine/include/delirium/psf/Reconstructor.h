// =============================================================================
// Reconstructor.h - OOP-level object reconstruction.
// A GameObject does not load a mesh: it asks a reconstructor for a
// PSFBody. Today the reconstructor is fixed (files / parametric
// primitives). The interface is the seam where an intelligent model
// plugs in later: same inputs (semantic description + physical
// constraints), same output (patches + K + seams), zero engine changes.
// =============================================================================
#pragma once
#include <memory>
#include <string>
#include "PSFBody.h"
#include "../physics/Material.h"

namespace DeliRium {

struct ReconstructionRequest {
    std::string semantic;        // "barrel", "npc_guard_arm", free text later
    Material    material;        // density, stiffness -> deformation modes
    float       tolerance_pct;   // geometric fidelity budget
    uint32_t    max_patches;     // memory budget
    Vec3        world_offset{0.f, 0.f, 0.f}; // where the body is baked in world space
};

class IReconstructor {
public:
    virtual ~IReconstructor() = default;
    virtual std::unique_ptr<PSFBody> Reconstruct(const ReconstructionRequest&) = 0;
};

// v0: load .psf files / build parametric primitives (sphere, torus, box).
class FixedReconstructor final : public IReconstructor {
public:
    std::unique_ptr<PSFBody> Reconstruct(const ReconstructionRequest&) override; // TODO
};

// v1+: model-driven. Emits the SAME records; engine cannot tell the difference.
// Current backend: the semantic request drives the GENERIC element-system
// generator (ai/ElementSystem.h), which bakes a recursive element tree
// (overlapping systems -> substance leaves) into PSF records. A learned
// categorizer replaces the generator later by emitting element trees; the
// baker and this class are unchanged.
class NeuralReconstructor final : public IReconstructor {
public:
    bool LoadModel(const char* path);                                            // TODO (RMW1 weights)
    std::unique_ptr<PSFBody> Reconstruct(const ReconstructionRequest&) override;
};

} // namespace DeliRium
