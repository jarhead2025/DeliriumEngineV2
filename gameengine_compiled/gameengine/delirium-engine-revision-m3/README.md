# DeliRium Engine Revision — M3

M3 upgrades the M2 surface bridge with higher-order PSF basis functions, welded seams, multi-surfel spatial cells, and temporal coherence in the render cache.

## Pipeline

```text
articulation → BodyState (generation-tagged)
                    ↓
          SurfaceEvaluator (higher-order basis + residuals)
                    ↓
          WeldedSeamSolver (stitch non-shared boundaries)
                    ↓
          CacheUpdater (incremental dirty-patch updates)
                    ↓
          SurfelCache (temporal coherence, generation-tagged)
                    ↓
          SpatialGrid (multi-surfel cells, material masks)
```

## Deliverables

- **Higher-order basis functions** — `BasisType` enum (plano, cúpula, silla, valle_u, valle_v, eggbox, twist) with displacement evaluation. Bilinear remains the M2 baseline.
- **Vector residuals** — per-corner displacement vectors for sub-anchor geometric detail.
- **Welded seams** — non-shared patch boundaries stitched within configurable tolerance.
- **Multi-surfel spatial grid** — replaces single-surfel-per-cell. Each cell holds multiple surfels with material identity preserved.
- **Temporal coherence** — surfels persist across frames unless their source patch is dirty.
- **Topology generation tracking** — re-finalize increments `TopologyGeneration`, invalidating stale cache entries.

## Build

```bash
cmake -S . -B build
cmake --build build --parallel
./build/delirium_m3_tests
```

For platform-specific prerequisites, see [`docs/BUILD.md`](docs/BUILD.md).

## Relation to M2

- All M2 concepts preserved: `BodyTopology`, double-buffered `BodyState`, dirty propagation, generation tagging.
- `PatchTopology` upgraded with `BasisType` and `PatchCoefficients` (scalar + vector residuals).
- `SurfaceEvaluator` upgraded from bilinear-only to higher-order basis + residuals.
- New: `WeldedSeamSolver`, `SpatialGrid`, temporal coherence in `SurfelCache`.
- Composition order: bone pose → basis displacement → vector residuals.

See `docs/M3_SCOPE.md` and `docs/M4_HANDOFF_CONTRACT.md` for exact boundaries.