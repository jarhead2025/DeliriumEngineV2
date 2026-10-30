// =============================================================================
// SpectralObject.h - spectral primitives as GAME-ENGINE objects.
//
// A SpectralObject is one primitive + a transform + a LOD order. A
// SpectralModel is a COMPOSITION of them (the user's "objetos individuales
// que conforman algo mas grande"): a character is ~15 capsule-primitives,
// each with its own baked detail field, rigged to a transform hierarchy.
//
// Why this is a game-engine representation, not just compression:
//   * ANIMATION IS FREE: moving/rotating/scaling a part transforms its
//     primitive; the detail field (the coefficients) rides along untouched.
//     No per-vertex skinning, no re-evaluation of detail.
//   * LOD IS FREE AND CONTINUOUS: distance selects how many SH terms to sum
//     (lod_order). One file is the whole LOD ladder -- no stored levels, no
//     popping. (Measured: order 4 = far, 8 = medium, 32 = close-up.)
//   * COMPOSITION SCALES: complex/non-star-shaped objects are unions of
//     simple star-shaped parts; physics moves the parts, detail follows.
// =============================================================================
#pragma once
#include "SpectralPrimitive.h"
#include "delirium/core/Types.h"        // Vec3, Quat only
#include <vector>
#include <memory>
#include <string>

namespace DeliRium {

// Transform for spectral objects. Kept local so the spectral subsystem is
// independent of the patch engine (no PSFBody/seam/physics dependency) --
// spectral primitives are a self-contained, cleaner format.
struct SpectralTransform {
    Vec3 position{0,0,0};
    Quat rotation{0,0,0,1};
    Vec3 scale{1,1,1};
};


struct SpectralObject {
    std::shared_ptr<SpectralPrimitive> prim;
    SpectralTransform transform;
    uint32_t  lod_order = 0;     // 0 = auto (pick from distance); else fixed

    // World-space surface point for a direction, applying the transform.
    Vec3 SurfaceWorld(float theta, float phi) const;
    // Pick an SH order from the projected screen size (continuous LOD).
    uint32_t LodForScreenSize(float projected_px) const;

    // The user's insight: one distance picks BOTH the geometry order AND the
    // color order -- so a distant object is cheap and soft, and as it nears,
    // its silhouette AND its texture sharpen progressively. Returns {geom,
    // color} orders. Color tracks geometry but is capped at the baked color
    // order. This is an inherent, continuous texture LOD: no mipmaps, no
    // stored levels, no popping -- just sum more frequency terms when close.
    struct Lod { uint32_t geom, color; };
    Lod LodPairForScreenSize(float projected_px, uint32_t color_max) const;
};

// A composite model: many primitives forming one game object.
class SpectralModel {
public:
    bool LoadPart(const std::string& spf_path, const SpectralTransform& local);
    void AddPart(std::shared_ptr<SpectralPrimitive> p, const SpectralTransform& local);

    size_t PartCount() const { return parts_.size(); }
    const SpectralObject& Part(size_t i) const { return parts_[i]; }
    SpectralObject&       Part(size_t i)       { return parts_[i]; }

    // Total bytes of all parts (the model's footprint).
    size_t ByteSize() const;
    // Move the whole model (parts keep their local transforms).
    void SetRoot(const SpectralTransform& t) { root_ = t; }
    const SpectralTransform& Root() const { return root_; }

private:
    std::vector<SpectralObject> parts_;
    SpectralTransform root_;
};

} // namespace DeliRium
