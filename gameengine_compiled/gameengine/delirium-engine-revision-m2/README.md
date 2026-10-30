# DeliRium Engine Revision — M2

M2 extends the M1 body core with full articulation (orientation-aware transforms, angular springs, joint limits), pose controllers, and the first surface-to-cache bridge.

## Pipeline

```text
pose controller → target_pose → spring solve (linear + angular)
                                              ↓
                                     bone current_pose
                                              ↓
                              weighted anchor target positions
                                              ↓
                              BodyState (double-buffered, generation-tagged)
                                              ↓
                              SurfaceEvaluator → dirty patch surfels
                                              ↓
                              SurfelCache (generation-tagged render cache)
                                              ↓
                              OpenCLUploadAdapter (generation-gated GPU upload)
```

## Deliverables

- **Orientation-aware `Transform`** — `Quat` + `Vec3` translation. Rotation composes through quaternion multiplication.
- **Angular spring motion** — semi-implicit Euler integrates both linear and angular springs per bone.
- **Joint limits** — translation bounds and angular velocity clamping applied after each step.
- **Pose controllers** — `DirectPoseController` (explicit targets) and `BlendPoseController` (slerp between two sources). Controllers write `target_pose`, never raw anchor positions.
- **Surface evaluator** — bilinear patch evaluation with finite-difference normals. Seam error between shared-anchor patches is zero by construction.
- **Surfel cache** — generation-tagged render cache. `CacheUpdater` rebuilds surfels from all patches, tagged with the `BodyState::generation`.
- **OpenCL upload adapter** — stub implementation compiles without OpenCL SDK. Enforces generation invariant: refuses to upload a cache whose source generation doesn't match the expected `BodyState` generation.

## Build

```bash
cmake -S . -B build
cmake --build build --parallel
./build/delirium_m2_tests
```

For platform-specific prerequisites (Linux Mint + Intel Arc A770, macOS, Windows), see [`docs/BUILD.md`](docs/BUILD.md).

## Relation to M1

- All M1 concepts preserved: `BodyTopology`, double-buffered `BodyState`, explicit anchor influences, dirty propagation.
- `Transform` upgraded from translation-only to translation + quaternion rotation.
- `ArticulationRig` gains angular springs, joint limits, and rotational anchor displacement.
- New: `SurfaceEvaluator`, `SurfelCache`, `CacheUpdater`, `OpenCLUploadAdapter`.
- New: `PoseController` hierarchy for external input (IK, gameplay, animation).

See `docs/M2_SCOPE.md` and `docs/M3_HANDOFF_CONTRACT.md` for exact boundaries.