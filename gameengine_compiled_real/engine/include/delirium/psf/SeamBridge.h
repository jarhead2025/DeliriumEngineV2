// =============================================================================
// SeamBridge.h - Turn an un-stitchable seam into a GENERATIVE surface.
//
// When two charts meet at an edge with tangential disagreement, the scalar
// stitch (normal-only) cannot close it -- the surfaces genuinely differ in
// position AND tangent there. The mathematically sound fix is not to hide
// the gap but to FILL it with a bridging surface that matches both sides
// with tangent (G1) continuity. This is a Hermite blend (the CAD fillet/
// loft construction):
//
//   S(s,t) = H00(s) A(t) + H01(s) B(t) + H10(s) Ta(t) + H11(s) Tb(t)
//
// where A(t),B(t) are the two edge curves, Ta,Tb their cross-boundary
// tangents (pointing OUT of each patch, into the gap), and H** the cubic
// Hermite basis. At s=0 the bridge equals A(t) with tangent Ta (G1 with
// patch A); at s=1 it equals B(t) with tangent Tb (G1 with patch B). The
// interior is a smooth, curvature-carrying surface -- so the seam region
// becomes real geometry that enriches the form rather than a crack to
// paint over. This directly realizes "use where surfaces are stitched to
// build a more complex surface."
// =============================================================================
#pragma once
#include "PSFBody.h"
#include "Seam.h"
#include <vector>

namespace DeliRium {

struct BridgeSample {
    Vec3 P;      // bridged surface point
    Vec3 n;      // analytic normal (from dS/ds x dS/dt)
};

class SeamBridge {
public:
    // Evaluate the Hermite bridge for a seam at (s,t) in [0,1]^2.
    // s crosses from A's edge (0) to B's edge (1); t runs along the seam.
    static BridgeSample Evaluate(const PSFBody& body, const Seam& s,
                                 float ss, float tt);

    // Does this seam need a bridge? True when, after any stitch, the edges
    // still disagree beyond `tol` (fraction of diagonal) -- i.e. the
    // disagreement is tangential, not normal. Bridging is reserved for
    // these; ordinary seams stay stitched (cheaper, zero new patches).
    static bool NeedsBridge(const PSFBody& body, const Seam& s, float tol);

    // Measure the bridge's G1 continuity error with both flanking patches:
    // max angle (1-cos) between the bridge's boundary tangent and the
    // patch's edge tangent at s=0 and s=1. ~0 means a seamless join.
    static float G1Error(const PSFBody& body, const Seam& s);

    // Largest gap between the bridge boundary and each patch edge (should
    // be ~0 by construction -- the bridge interpolates the edges exactly).
    static float BoundaryGap(const PSFBody& body, const Seam& s);
};

} // namespace DeliRium
