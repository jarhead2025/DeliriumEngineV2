# M3 handoff contract — Surface continuity and representation upgrade

M3 builds on the M2 articulation + surface bridge. It must not regress M2's generation invariants or introduce ambiguous continuity between patches.

## M3 objectives

1. Upgrade bilinear patch evaluation to higher-order PSF basis functions (plano, cúpula, silla, valle_u, valle_v, eggbox, twist) for lateral geometric detail.
2. Add vector residuals to patch coefficients for sub-anchor geometric features.
3. Implement cage continuity: shared cage boundaries remain the primary mechanism for patch-to-patch seam elimination.
4. Add welded seams: non-shared boundaries between adjacent patches can be stitched with configurable tolerance.
5. Introduce multi-surfel spatial cells in the cache (replacing single-surfel-per-cell grids from the legacy engine).
6. Add temporal coherence: surfels persist across frames unless their source patch is dirty.

## Required invariants

- A welded seam error must be bounded by the configured tolerance after stitching.
- Multi-surfel cells must preserve material identity — surfels from different material regions cannot occupy the same cell without explicit blending rules.
- Temporal coherence must not retain surfels from patches that were removed by a topology change.
- The composition order remains: bone pose → modal → local material. M3 adds vector residuals as a sub-component of the local material stage.
- Cache generation tagging continues to enforce the M2 invariant.

## M3 test additions

- Higher-order basis evaluation at patch centers and boundaries.
- Welded seam error within tolerance after stitching.
- Multi-surfel cell occupancy with material separation.
- Temporal coherence: clean patches retain surfels across frames.
- Topology change invalidates removed patch surfels.