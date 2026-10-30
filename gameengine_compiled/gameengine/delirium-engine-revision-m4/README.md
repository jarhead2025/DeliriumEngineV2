# DeliRium Engine Revision — Milestone M4

**Material-aware physical body** with modal displacement, damage accumulation, and material-driven rendering.

## What M4 Adds to M3

| Feature | Description |
|---------|-------------|
| **Material Properties** | Per-region mechanical behavior (Young's modulus, Poisson ratio, density, yield strength) |
| **Modal Displacement** | Global elastic stage (bending, torsion, breathing) composes before local solve |
| **Elastic/Plastic Solve** | Sparse dirty-patch solver with strain energy computation |
| **Damage Accumulation** | Monotonic damage tracking with stiffness reduction |
| **Material Colors** | PBR properties (color, roughness, metalness) flow into surfel cache |
| **Strain Propagation** | Adjacent patches share strain energy through anchors |

## Pipeline Order

```
Bone Pose → Modal Displacement → Elastic/Plastic Solve → Damage Accumulation → Strain Propagation → Publish
```

## Key Invariants

- **Damage monotonicity**: damage never decreases within a generation
- **Material consistency**: same `MaterialId` in physics and rendering
- **Strain energy**: always non-negative
- **Modal composition**: always before local material solve

## Files

```
include/delirium_m4/
  types.hpp              Vec3, IDs, MaterialProperties, ModalMode, Surfel, SpatialCell
  body_topology.hpp      BodyTopology (from M3)
  body_state.hpp         BodyState with modal displacement, cumulative strain, damage
  material_registry.hpp  MaterialRegistry — property lookup + color mapping
  modal_system.hpp       ModalSystem — modal displacement computation
  elastic_solver.hpp     ElasticSolver — sparse dirty-patch elastic/plastic solve
  damage_accumulator.hpp DamageAccumulator — monotonic damage tracking
  basis_functions.hpp    Higher-order basis evaluation (from M3)
  welded_seam.hpp        WeldedSeamSolver (from M3)
  spatial_grid.hpp       Multi-surfel spatial grid (from M3)
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
  m4_tests.cpp           15 test groups covering all M4 contract items + M3 carryover
```

## Build

See [BUILD.md](docs/BUILD.md) for build instructions.

## Documentation

- [M4_SCOPE.md](docs/M4_SCOPE.md) — Detailed scope and design decisions
- [M5_HANDOFF_CONTRACT.md](docs/M5_HANDOFF_CONTRACT.md) — Contract for next milestone