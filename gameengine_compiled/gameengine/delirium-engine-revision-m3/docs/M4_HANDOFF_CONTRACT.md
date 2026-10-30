# M4 handoff contract — Material-aware physical body

M4 builds on the M3 surface continuity layer. It must not regress M3's seam invariants or introduce ambiguous material behavior.

## M4 objectives

1. Add per-region mechanical behavior: each material ID maps to a `MaterialProperties` struct (Young's modulus, Poisson ratio, density, yield strength).
2. Implement sparse local solves: only dirty patches participate in the local elastic/plastic solve.
3. Add modal displacement as the global elastic stage (between bone pose and local material).
4. Introduce damage accumulation: plastic deformation tracks cumulative strain, reducing stiffness over time.
5. Add material-driven surfel colors: each material defines a base color, roughness, and metalness for the render cache.
6. Implement patch-to-patch strain propagation: adjacent patches share strain energy through shared anchors.

## Required invariants

- A material's mechanical properties must be consistent between physics and rendering (same material ID).
- Damage must be monotonic (never decreases) within a single generation.
- Modal displacement must compose before local material displacement (declared order).
- Strain energy must be non-negative.
- Cache generation tagging continues to enforce the M2/M3 invariant.

## M4 test additions

- Material properties affect spring behavior.
- Damage accumulation is monotonic.
- Modal displacement composes correctly with bone pose.
- Strain energy is non-negative.
- Material-driven colors appear in surfel cache.