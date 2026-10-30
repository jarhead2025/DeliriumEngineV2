# M4 Scope — Material-aware physical body

## What M4 delivers

M4 extends the M3 surface continuity layer with per-region mechanical behavior, modal displacement, damage accumulation, and material-driven rendering.

## Objectives achieved

1. **Per-region mechanical behavior** — `MaterialProperties` struct maps each `MaterialId` to Young's modulus, Poisson ratio, density, yield strength. The `MaterialRegistry` provides lookup and defaults.

2. **Sparse local elastic/plastic solve** — `ElasticSolver` operates only on dirty patches. Strain energy is computed per-patch; patches exceeding yield strength accumulate plastic deformation.

3. **Modal displacement** — `ModalSystem` adds a global elastic stage between bone pose and local material solve. Modal modes (bending, torsion, breathing) displace anchors before the local solver runs.

4. **Damage accumulation** — `DamageAccumulator` tracks cumulative plastic strain per patch. Damage is monotonic (never decreases within a generation). Stiffness is reduced proportionally to damage.

5. **Material-driven surfel colors** — each material defines base color (RGB), roughness, and metalness. The `CacheUpdater` applies these to surfels during rebuild/update.

6. **Patch-to-patch strain propagation** — adjacent patches sharing anchors distribute strain energy through shared anchor states.

## Design invariants maintained

- Material mechanical properties are consistent between physics and rendering (same `MaterialId`).
- Damage is monotonic within a single generation.
- Modal displacement composes BEFORE local material displacement (declared pipeline order).
- Strain energy is always non-negative.
- Cache generation tagging continues to enforce the M2/M3 invariant.
- Composition order: bone pose → modal displacement → local material solve → damage update.

## Files

```
include/delirium_m4/
  types.hpp              Vec3, IDs, MaterialProperties, ModalMode, Surfel, SpatialCell
  body_topology.hpp      BodyTopology (carried from M3)
  body_state.hpp         BodyState with modal displacement, cumulative strain, damage
  material_registry.hpp  MaterialRegistry — material property lookup + color mapping
  modal_system.hpp       ModalSystem — modal displacement computation
  elastic_solver.hpp     ElasticSolver — sparse dirty-patch elastic/plastic solve
  damage_accumulator.hpp DamageAccumulator — monotonic damage tracking
  basis_functions.hpp    Higher-order basis evaluation (carried from M3)
  welded_seam.hpp        WeldedSeamSolver (carried from M3)
  spatial_grid.hpp       Multi-surfel spatial grid (carried from M3)
  surfel_cache.hpp       SurfelCache with material-driven colors
  body_runtime.hpp       BodyRuntime orchestrator with full M4 pipeline

src/
  body_topology.cpp
  material_registry.cpp
  modal_system.cpp
  elastic_solver.cpp
  damage_accumulator.cpp
  basis_functions.cpp
  welded_seam.cpp
  spatial_grid.cpp
  surfel_cache.cpp
  body_runtime.cpp

tests/
  m4_tests.cpp           Test groups covering all M4 contract items
```