// =============================================================================
// Seam.h - The sewing mechanism at runtime.
// Edges are matched by shared anchor pairs; T-junctions (one coarse edge
// facing two fine half-edges) are detected via the midpoint anchor:
// coarse (A,B) splits against fine (A,M) + (M,B). Stitch s(t)=t(1-t)(a+bt),
// int8 (a,b) per patch edge, fitted by 2x2 least squares with an exact
// apply-measure-revert guard (provably non-worsening, same rule as the
// encoder). Policy: equal patches split the gap; fine absorbs it fully
// against coarse.
// =============================================================================
#pragma once
#include <cstdint>
#include <vector>
#include "../physics/Material.h"
#include <unordered_set>

namespace DeliRium {

class PSFBody;

// one matched seam: side A <-> side B over A's full edge t in [0,1],
// mapping to B's parameter range [tb0, tb1] (handles T-junction halves
// and orientation flips uniformly)
struct Seam {
    uint32_t patch_a, patch_b;
    uint8_t  edge_a,  edge_b;     // 0..3 = v0,v1,u0,u1
    float    tb0, tb1;            // B param at A's t=0 and t=1
    bool     a_is_finer;          // T-junction: A is the half-edge side
};

class SeamGraph {
public:
    // Full rebuild from the body's current records (anchor-pair matching
    // + T-junction detection). O(n_edges) with hash maps.
    void Build(const PSFBody& body);

    // Refit stitches on every seam touching any patch in `ids`
    // (children of a split + their neighbors). Non-worsening guard.
    void RestitchAround(PSFBody& body, const std::vector<uint32_t>& ids);

    // Fit all seams (encode-time equivalent; used by tests).
    void FitAll(PSFBody& body);

    // Global, simultaneous harmonic seam reconstruction. Relaxes all
    // shared anchors together toward the continuous+smooth configuration
    // (Jacobi relaxation of the discrete Laplace equation with seam
    // continuity), rather than stitching seams one at a time. `smooth`
    // in [0,1] trades pure continuity (0) for extra surface smoothing (1).
    // Returns the max seam gap after relaxation.
    float HarmonicWeld(PSFBody& body, int iterations = 24, float smooth = 0.2f);

    // Material-aware weld: the weld behaves per the material's category
    // (rigid holds edges, fluid melts seams). Same global relaxation,
    // tuned by the profile (relax/smooth/iterations/cap).
    float HarmonicWeld(PSFBody& body, const WeldProfile& profile);

    // Enumerate seams whose disagreement is tangential (un-stitchable) and
    // should instead be filled with a Hermite BRIDGE surface (see
    // SeamBridge). Returns the indices into All() that need a bridge. The
    // bridge geometry is DERIVED (regenerated from the seam, zero stored
    // bytes), so this is the list the renderer/exporter iterates to draw
    // or bake the bridging surfaces.
    std::vector<uint32_t> BridgeSeams(const PSFBody& body,
                                      float tol = 0.01f) const;

    // Harmonize the DEFORMATION coefficients along shared edges. When two
    // patches share an edge by index (same corner anchors), each still
    // bows that edge with its OWN 2 deformation coefficients -- and the
    // exporter fits them independently, so the edge curves apart even
    // though the corners coincide (the dominant crack on high-frequency
    // assets). Setting both sides' edge-controlling coefs to their average
    // makes the shared edge a single curve on both patches -> the seam
    // closes to ~0 with no stitch and no added geometry. Returns the max
    // seam gap afterward.
    float HarmonizeEdgeCoefs(PSFBody& body);

    float MaxGap(const PSFBody& body) const;
    size_t SeamCount() const { return seams.size(); }
    const std::vector<Seam>& All() const { return seams; }

private:
    void  FitSeam(PSFBody& body, const Seam& s,
                  std::unordered_set<uint64_t>& edge_used,
                  float strength = 1.0f);
    std::vector<Seam> seams;
};

} // namespace DeliRium
