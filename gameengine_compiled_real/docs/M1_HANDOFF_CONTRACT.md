# M1 handoff contract — canonical body and articulation foundation

M1 builds the next layer from the M0 baseline. It may add code and files, but it must preserve the following contracts.

## Ownership contract

- `BodyTopology` owns stable IDs, adjacency, patch/material assignment, refinement lineage, and skeleton binding metadata.
- `BodyState` owns the current simulation state: anchors, velocities, patch coefficients, damage/field values, and dirty ranges.
- A render cache is derived from a complete body generation. It is not an alternative authoritative geometry state.
- Raw source meshes are import/interchange data, not required runtime truth.

## Articulation contract

A final anchor position is conceptually:

```text
rest anchor
  → weighted bone target
  → spring/constraint correction
  → body/material response
  → seam/shared-boundary enforcement
```

- Bone influence weights are explicit data, not recomputed from inverse distance every frame.
- Animation skeleton and physical support graph are distinct.
- A frame must not expose a render cache partly generated from a body state being written by physics.
- The first implementation may use CPU double buffering; it must still carry explicit generation IDs.

## Surface contract

- Patch boundaries are either shared by construction or governed by an explicit seam constraint.
- Surface normals are derived from current surface geometry whenever practical.
- Scalar normal-height PSF remains supported during migration.
- Any vector residual extension must be optional/versioned; it must not silently reinterpret existing PSF data.

## Material contract

- Materials are assignable per patch/region.
- Render, mechanical, and field properties are separate concepts even if M1 initially implements only a small subset.
- Material identity must reach the surfel cache so marching/fill cannot blend across incompatible surfaces.

## Surfel/march contract

- Surfel caches track source topology and geometry generation.
- A spatial cell may contain multiple surfels. Single-entry cells are a prototype limitation, not the final M1 design.
- Coverage tests use surfel footprint/radius and depth/normal compatibility.
- A fill pass may not create geometry across a clear depth, normal, material, or silhouette discontinuity.

## Required M1 tests

1. Stored anchor-to-bone weights normalize and produce local motion.
2. A body pose update increments geometry generation and dirties only affected patch/cache ranges.
3. Renderer observes only complete generations.
4. Material boundary samples are not averaged together by render composition.
5. Shared-boundary/seam error remains within the declared tolerance after an articulated pose update.
