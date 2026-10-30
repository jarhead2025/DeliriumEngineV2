// =============================================================================
// Patch.h - Runtime view of one SurfaceRecord: evaluation, derivatives,
// and live edition of its 8 equation constants.
// Surface(u,v) = Bilinear(anchors) + h(u,v) * NormalField(u,v)
// h = deform (corner-vanishing basis) + seams. The surface ALWAYS
// interpolates its 4 anchors regardless of coefficients.
// KEEP EVALUATION IN SYNC WITH kernels/psf_splat.cl (validated by
// tests/ against tools/psf_validate.py).
// =============================================================================
#pragma once
#include <cstdint>
#include "../core/Types.h"

namespace DeliRium {

enum class PatchEdge : uint8_t { V0 = 0, V1 = 1, U0 = 2, U1 = 3 };

class AnchorPool;

class Patch {
public:
    Patch() = default;
    Patch(SurfaceRecord* rec, AnchorPool* anchors,
          int8_t* stitch8, float s_scale, uint32_t id);

    // --- evaluation (mirrors the kernel) ---------------------------------
    Vec3  Evaluate(float u, float v) const;
    Vec3  EvaluateNormal(float u, float v) const;   // analytic dPdu x dPdv
    Vec3  EvaluateNormalField(float u, float v) const; // displacement dir n(u,v)
    // position + partials in one pass (Newton solvers live on this)
    void  EvaluateDerivs(float u, float v, Vec3& P,
                         Vec3& dPdu, Vec3& dPdv) const;
    float EvaluateHeight(float u, float v) const;   // h incl. seams
    Vec3  EvaluateColor(float u, float v) const;    // pattern sum K_i*P_i
    // add `k` (0..1 units) to the contribution of pattern `pid` on every
    // channel where it is already in the 4-pattern set (Morph slider)
    void  AddColorK(uint32_t pid, float k);

    // --- live shape manipulation ------------------------------------------
    void  SetCoefficient(int i, float value);
    float GetCoefficient(int i) const;
    void  AddDisplacementMode(const float k[8]);
    void  PinEdgeToCurve(PatchEdge e, float a, float b);  // TODO via SeamGraph

    uint32_t Id() const { return id; }
    bool     Dirty() const { return dirty; }
    void     ClearDirty() { dirty = false; }

private:
    // partial derivatives of h (deform + seams) wrt u and v
    void HeightGrad(float u, float v, float& dh_du, float& dh_dv) const;
    // amplitude scale for procedural detail: tied to the deformation unit
    // (a small multiple of the stitch scale) so detail magnitude is set in
    // the same units the rest of the patch uses, independent of resolution
    float DetailAmpScale() const { return s_scale > 0 ? s_scale * 64.0f : 0.01f; }

    SurfaceRecord* rec     = nullptr;
    AnchorPool*    pool    = nullptr;
    int8_t*        stitch  = nullptr;   // 8 int8: v0,v1,u0,u1 x (a,b)
    float          s_scale = 0.0f;
    uint32_t       id      = 0;
    bool           dirty   = true;
};

} // namespace DeliRium
