# M2 handoff contract — full articulation and first surface bridge

M2 extends the M1 body core. It must not reintroduce ambiguous ownership between animation, simulation, PSF/cage geometry, and render caches.

## M2 objectives

1. Upgrade `Transform` from translation-only to orientation-aware transforms.
2. Add joint hierarchy evaluation, angular spring/damper motion, and explicit joint limits.
3. Add optional IK/gameplay pose controllers that write `target_pose`, never raw anchor positions.
4. Bridge published `BodyState` anchor positions into one canonical PSF/cage surface evaluator.
5. Create a cache updater that consumes dirty patch ranges and emits a generation-tagged surfel cache.
6. Add the first OpenCL-facing upload adapter without making OpenCL required for CPU core tests.

## Required invariants

- A surface/cache update reads one published `BodyState::generation` and tags its output with that same generation.
- A renderer may use only a cache whose source generation is complete and known.
- A topology change is distinct from a geometry-state change. It increments a topology generation and invalidates cache ranges deliberately.
- Bone pose, modal displacement, and local material displacement compose in a declared order.
- Render composition retains patch material identity so hole fill and blending cannot cross incompatible material regions.
- If M2 uses cage evaluation, shared cage boundaries must remain the primary continuity mechanism. Post-process fill is not a substitute.

## M2 test additions

- Quaternion/rotation pose locality.
- Joint-limit convergence.
- Deterministic fixed-step secondary motion.
- Surface seam error after an articulated pose.
- Dirty patch ranges converted to exactly corresponding cache ranges.
- Cache-generation mismatch rejected by renderer/upload adapter.
