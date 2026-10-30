# DeliRium Engine Revision — M1

M1 implements the first runtime layer defined by M0: a canonical, generation-aware body model that separates stable topology from dynamic state and gives articulation a deterministic path into anchor targets.

M1 deliberately remains renderer- and solver-agnostic. It does not replace the experimental PSF/cage evaluator or progressive marcher. Instead, it provides the state contract they require:

```text
articulation target → anchor targets → BodyState generation
                    → dirty patch ranges → derived render-cache update
```

## Deliverables

- `BodyTopology`: stable IDs, patch/anchor adjacency, material assignment, and patch-to-cache ranges.
- `BodyState`: double-buffered anchor state, patch state, dirty ranges, and monotonically increasing complete generations.
- `ArticulationRig`: explicit four-bone anchor influences, rest/bind/current/target poses, and fixed-step translation spring behavior.
- `BodyRuntime`: applies a completed articulation target to the write buffer, derives dirty patches, and publishes a complete state by swapping buffers.
- Unit-style executable tests for topology validation, local dirty propagation, articulation locality, material boundary ownership, and buffer publication.

## Build

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

M1 uses C++17 only. OpenCL, SDL, and a GPU are deliberately not required to compile or test the body core. M2/M5 can consume this body state on OpenCL-capable systems.

For platform-specific prerequisites (Linux Mint + Intel Arc A770, macOS, Windows), see [`docs/BUILD.md`](docs/BUILD.md).

## Relation to the prior experiment

- PSF/cage data remains the intended authoritative surface model.
- The M1 types do not redefine a binary PSF file format.
- A surfel cache remains derived state. `PatchCacheRange` lets a future cache updater regenerate only patches affected by a completed body generation.
- M1 does not use inverse-distance skinning or hidden transforms in reserved binary fields.

See `docs/M1_SCOPE.md` and `docs/M2_HANDOFF_CONTRACT.md` for exact boundaries.
