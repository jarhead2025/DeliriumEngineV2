# M4 Handoff Contract — Multi-layer simulation bridge

M4 delivers material-aware physical bodies with modal displacement, damage accumulation, and material-driven rendering. M5 must not regress M4's damage monotonicity or material consistency invariants.

## M5 objectives (proposed)

1. Add contact/collision detection between body surfaces using the spatial grid.
2. Implement multi-body interaction: forces from contact propagate through the elastic solver.
3. Add thermal simulation: heat diffusion through anchors, thermal expansion.
4. Implement OpenCL compute kernel for parallel surface evaluation and elastic solve.
5. Add animation keyframe system for modal mode amplitudes.
6. Implement LOD (level-of-detail) for surfel cache based on camera distance.

## Required invariants

- Damage monotonicity: damage never decreases within a generation (M4 invariant).
- Material consistency: same MaterialId maps to same properties in physics and rendering.
- Modal displacement composes before local material solve (pipeline order).
- Strain energy is non-negative.
- Cache generation tagging enforces M2/M3/M4 invariant.
- Contact forces must not create negative strain energy.

## M5 test additions

- Contact detection finds overlapping surfels between two bodies.
- Contact forces propagate through elastic solver.
- Thermal diffusion reduces temperature gradients over time.
- OpenCL kernel produces same results as CPU evaluation.
- Animation keyframes interpolate modal amplitudes correctly.
- LOD reduces surfel count at distance without breaking continuity.