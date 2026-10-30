// =============================================================================
// LocalFEM.h - The near/expensive scale of the scale-based principle.
// Where the error estimator (or gameplay: impacts, fractures) demands it,
// the region is refined (PSFBody::SplitPatch) and a small dense FEM
// system is solved ON THE PATCH BASIS -- the patches ARE the elements
// (isogeometric analysis). Element integrals over the predetermined
// basis are precomputed once; assembly is table x jacobian FMA work.
// =============================================================================
#pragma once
#include "../psf/PSFBody.h"
#include "Material.h"

namespace DeliRium {

class LocalFEM {
public:
    LocalFEM();
    ~LocalFEM();

    // Collect the patches whose centers lie within `radius` of `center`
    // as elements; their anchors + 8 coefficients become DOFs. Anchors
    // also referenced by patches OUTSIDE the region are clamped (the
    // surrounding body holds the boundary fixed).
    void BindRegion(PSFBody& body, const Vec3& center, float radius);

    // Assemble dense K (membrane stiffness) and M (consistent mass) on the
    // patch basis using the precomputed 12x12 element tables.
    void AssembleSystem(const Material& m);

    // One implicit Newmark/backward-Euler step: unconditionally stable.
    void Step(float dt);

    // Push solved displacements back into anchor positions + patch
    // coefficients, mark dirty, and restitch the region.
    void WriteBack();

    // Inject a displacement into one region patch (test / gameplay impact).
    void Excite(uint32_t patch_local_index, float amount);

    // Residual proxy that drives live refinement (seam gaps).
    float EstimateError(uint32_t patch_id) const;

    int   DofCount() const;
    int   PatchCount() const;
    float Energy() const;

private:
    struct Impl;
    Impl* impl;
};

} // namespace DeliRium
