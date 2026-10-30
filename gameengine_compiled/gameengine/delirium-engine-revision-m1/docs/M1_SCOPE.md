# M1 scope and exit gate

## Purpose

M1 makes the object-state architecture concrete. The previous engine had anchors, patches, bones, materials, modal state, FEM state, and render caches, but their ownership boundaries were not yet explicit. M1 provides the minimum durable body core on which those systems can be rebuilt.

## Delivered in M1

### Canonical topology

`BodyTopology` contains stable anchor and patch identities, anchor-to-patch adjacency, patch material ownership, and a future surfel-cache range for each patch. Topology is finalized before runtime state can be created.

### Double-buffered dynamic state

`BodyRuntime` owns two `BodyState` buffers. It reads only a complete front state, writes the next state to the back buffer, marks local dirty ranges, increments generation, and publishes through a front/back swap. This establishes a simple producer/consumer boundary for later physics and rendering.

### Explicit articulation bindings

Every anchor has a fixed four-slot bone influence record. Nonzero weights must sum to 65535. The runtime does not infer weights by inverse distance and does not use reserved serialized bytes as a transform channel.

### Spring target prototype

M1 uses translation-only semi-implicit spring motion. This is intentionally narrow: it validates target-pose → spring → anchor-target → dirty patch propagation. Rotation, angular spring motion, joint limits, IK, and contacts are deferred to M2.

## Non-goals

- No PSF/cage evaluator is ported yet.
- No binary file format is frozen.
- No OpenCL buffers or surfel cache updater is implemented.
- No modal/FEM/material solve is executed.
- No renderer decides color blending yet.

## Exit gate

M1 is complete when the test executable verifies:

1. Topology rejects missing material assignment or out-of-range anchors.
2. Explicit normalized influences bind every anchor.
3. A spring-driven bone moves only anchors assigned to it.
4. A moved anchor dirties precisely all incident patches.
5. A publish increments generation and readers see only the new complete state.
6. Patch material IDs remain associated with topology for later render-composition rules.

## Design note

M1 does not claim that translation-only skinning is final animation. It claims a clean *control path* exists. Later layers may replace the articulation math without breaking topology ownership, state publication, local invalidation, or material-region identity.
