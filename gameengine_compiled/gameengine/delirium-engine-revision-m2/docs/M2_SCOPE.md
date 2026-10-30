# M2 Scope — Full articulation and first surface bridge

## What M2 delivers

M2 extends the M1 body core with orientation-aware transforms, angular spring motion, joint limits, pose controllers, and the first surface-to-cache bridge.

## Objectives achieved

1. **Orientation-aware `Transform`** — `Quat` (unit quaternion) + `Vec3` translation. Rotation composes through quaternion multiplication.

2. **Joint hierarchy with angular springs** — `BoneState` carries `AngularSpringParams` and `JointLimits`. Semi-implicit Euler integrates both linear and angular springs independently.

3. **Pose controllers** — `PoseController` abstract base with `DirectPoseController` (explicit targets) and `BlendPoseController` (slerp between two sources). Controllers write `target_pose` into the rig, never raw anchor positions.

4. **Surface evaluator** — `SurfaceEvaluator` reads one published `BodyState` generation and evaluates bilinear patch positions + finite-difference normals. Seam error between shared-anchor patches is zero by construction.

5. **Surfel cache** — `SurfelCache` is a generation-tagged render cache. `CacheUpdater` consumes dirty patch ranges and emits surfels tagged with the same `BodyState::generation`.

6. **OpenCL upload adapter** — `OpenCLUploadAdapter` enforces the generation invariant: it refuses to upload a cache whose source generation does not match the expected `BodyState` generation. Stub implementation compiles without OpenCL SDK.

## Design invariants maintained

- A surface/cache update reads one published `BodyState::generation` and tags its output with that same generation.
- A renderer may use only a cache whose source generation is complete and known.
- Topology changes increment `TopologyGeneration` and invalidate cache ranges deliberately.
- Bone pose → modal displacement → local material displacement compose in declared order (M2: only bone pose is active).
- Render composition retains patch material identity.
- Shared cage boundaries are the primary continuity mechanism.

## Files

```
include/delirium_m2/
  types.hpp              Vec3, Quat, Transform, JointLimits, SpringParams
  body_topology.hpp      BodyTopology with topology_generation tracking
  body_state.hpp         BodyState with topology_generation field
  articulation.hpp       ArticulationRig with angular springs + joint limits
  pose_controller.hpp    PoseController, DirectPoseController, BlendPoseController
  surface_evaluator.hpp  SurfaceEvaluator (bilinear patches, seam error)
  surfel_cache.hpp       Surfel, SurfelCache, CacheUpdater
  opencl_adapter.hpp     OpenCLUploadAdapter (generation-gated upload)
  body_runtime.hpp       BodyRuntime orchestrator

src/
  types.cpp              Quat implementation
  body_topology.cpp
  articulation.cpp       Angular spring integration + joint limit clamping
  pose_controller.cpp    Direct + blend controller implementations
  surface_evaluator.cpp  Bilinear evaluation + finite-difference normals
  surfel_cache.cpp       Cache rebuild with generation tagging
  opencl_adapter.cpp     Stub adapter (no OpenCL SDK required)
  body_runtime.cpp       Full frame step with controller → rig → anchor → cache

tests/
  m2_tests.cpp           10 test groups covering all M2 contract items
```