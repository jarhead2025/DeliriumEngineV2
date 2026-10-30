# M3 Scope — Surface continuity and representation upgrade

## What M3 delivers

M3 extends the M2 surface bridge with higher-order PSF basis functions, welded seams, multi-surfel spatial cells, and temporal coherence.

## Objectives achieved

1. **Higher-order basis functions** — `BasisType` enum with 8 types. Each adds lateral geometric detail through 8 scalar coefficients beyond the bilinear base.

2. **Vector residuals** — per-corner displacement vectors (`std::array<Vec3, 4>`) for sub-anchor features (creases, dents, bumps).

3. **Welded seams** — `WeldedSeam` descriptor connects non-shared patch boundaries. `WeldedSeamSolver` adjusts anchor positions iteratively to keep seam error within tolerance.

4. **Multi-surfel spatial grid** — `SpatialGrid` replaces single-surfel-per-cell. Each cell holds compact surfel index lists with material masks.

5. **Temporal coherence** — `SurfelCache` tracks which patches were updated per frame. Clean patches retain surfels across generations.

6. **Topology generation tracking** — `TopologyGeneration` increments on re-finalize, invalidating stale cache entries.

## Design invariants maintained

- Welded seam error bounded by configured tolerance after stitching.
- Multi-surfel cells preserve material identity — material masks prevent incompatible blending.
- Temporal coherence does not retain surfels from patches removed by topology change.
- Composition order: bone pose → basis displacement → vector residuals.
- Cache generation tagging enforces the M2 invariant.

## Files

```
include/delirium_m3/
  types.hpp              Vec3, BasisType, PatchCoefficients, WeldedSeam, Surfel, SpatialCell
  body_topology.hpp      BodyTopology with basis type, coefficients, welded seams
  body_state.hpp         BodyState with vector residuals in PatchState
  basis_functions.hpp    Higher-order basis evaluation (8 types)
  welded_seam.hpp        WeldedSeamSolver (error computation + adjustment)
  spatial_grid.hpp       Multi-surfel spatial grid with material masks
  surfel_cache.hpp       SurfelCache with temporal coherence tracking
  body_runtime.hpp       BodyRuntime orchestrator with seam solving

src/
  body_topology.cpp
  basis_functions.cpp    All 8 basis types + residuals + finite-difference normals
  welded_seam.cpp        Seam sampling + iterative adjustment
  spatial_grid.cpp       Grid build + position-to-cell mapping
  surfel_cache.cpp       Incremental update + full rebuild
  body_runtime.cpp       Orchestrator with seam solving + cache update

tests/
  m3_tests.cpp           10 test groups covering all M3 contract items
```