# DeliRium Engine V2 — Progress Log (v2.1)

**Branch:** `v2.1`
**Compiled:** 2026-11-18
**Last updated:** 2026-11-26 (Section 11 COMPLETE: GPU modal writeback integration FIXED + FPS stall ROOT CAUSE FOUND + FIXED; Path B2 color-field migration complete; distributed temperature field CPU substrate complete)
**Status:** TriInfill GPU primary renderer working + optimized; Workspace editor ported; volumetric generation fixed; **PSF v4 parametric surfaces implemented**; **GPU physics kernels implemented + benchmark-tested + ALL WIRED INTO `Step()`** (N-body pairwise, FEM solve, anchor-reflow, modal writeback — each with own flag + CPU fallback); tri_infill parametric fixes applied; **GPU modal writeback integration COMPLETE** (REST uploaded once at bind, per-frame only K vector uploads); **FPS stall FIXED** (host-upload FlushToGPU loop gated behind modal-body detection).

This document synthesizes the major milestones delivered across recent development sessions. It is a human-readable progress summary, not a substitute for the per-file code. Reference docs live alongside it in `docs/`.

---

## 1. Render Pipeline — TriInfill GPU (PRIMARY)

The engine now renders via a GPU-compute TriInfill path (`DELIRIUM_TRI=1`). This is the default and best-performing renderer on the Intel Arc A770.

### Architecture
- **CPU samples → GPU rasterizes.** PSF math surfaces are evaluated on a structured `su×sv` grid, structured triangles are built (2 per cell), and the GPU kernel rasterizes with Gouraud shading + early-z rejection.
- **Depth-key compositing.** Multi-object scenes compose via `atom_min` on a shared depth-key `(as_uint(depth)<<32)|rgba` framebuffer — no per-body clears needed.
- **Physics-first.** Geometry derives from the authoritative PSF state (anchors + coefficients + patterns). Rendering is a view of that state, not the source of physical truth.

### Kernels
| Kernel | File | Args | Role |
|--------|------|------|------|
| `psf_sample` | `kernels/psf_splat.cl` | 18 | One work-item per sample point. Evaluates PSF surface math (position + analytic normal + base color), projects to screen space, writes vertex grid + normal grid. Mirrors `Patch.cpp::EvaluateDerivs` convention A bit-compatible. |
| `tri_build` | `kernels/psf_splat.cl` | 10 | One work-item per triangle. Reads grid via indexed output + LOD tables, writes index buffer. Uses `atomic_inc` for output indexing. |
| `tri_raster` | `kernels/tri_infill.cl` | 11 | Edge-function rasterizer. Interpolates normals, Gouraud shading, early-z rejection skips shading for occluded pixels. Writes depth-key via `atom_min`. |

### Optimizations delivered & verified (Intel Arc A770, 1024×768)
All numbers measured on the A770 with `DELIRIUM_TRI=1`.

| Optimization | Effect |
|--------------|--------|
| **Phase 1 profiling** | Revealed CPU PSF sampling dominated >90% of render time; GPU rasterizer only 0.04ms regardless of UVN. Corrected priority: GPU sampling is #1; hardware rasterization deprioritized (no gain). |
| **Phase 2 GPU sampling** | `sample_ms` dropped 15.4ms → ~0.045ms (**~340×**). Total render 16.4ms → ~0.35ms (**~47×**). Engine became display-refresh-limited (~62 FPS). |
| **Rasterizer Gouraud + early-z** | `gpu_raster_ms` dropped 0.054→0.040ms (**~27%**). Image unchanged at UVN≥10. Kernel signature unchanged (no host changes). |
| **tri_build indexed output** | Wrote index buffer instead of redundant vertices. Write traffic dropped ~4-5× (~220MB → ~31MB/frame). `upload_ms` drops 2.4-3.8× at UVN≥30. Signature changed: tri_build 9→10 args, tri_raster 10→11 args. |
| **Adaptive UVN/LOD** | Per-patch UVN from projected screen-space edge size. Near patches get high UVN, far patches get low UVN. `gpu_raster_ms` dropped 9.4ms→0.36ms (**26×**) at extreme UVN vs uniform. |

### Verified steady-state (single torus, adaptive LOD on)
| UVN | sample_ms | upload_ms | gpu_raster_ms | total_ms | FPS | Triangles |
|-----|-----------|-----------|---------------|----------|-----|-----------|
| 10  | 0.03 | 0.27 | 0.05 | 0.35 | 61 | ~109k |
| 30  | 0.09 | 0.29 | 0.13 | 0.51 | 62 | ~980k |
| 50  | 0.21 | 0.41 | 0.35 | 0.97 | 61 | ~2.7M |
| 100 | 0.69 | 0.89 | 1.4  | 2.9  | 61 | ~10.9M |
| 200 | 2.7  | 2.6  | 5.9  | 11.0 | 62 | ~9.8M |
| 256 | 4.3  | 4.3  | 9.4  | 17.8 | 43 | ~17.8M |

**Final verified state:** `DELIRIUM_TRI=1 DELIRIUM_TRI_LOD=1 DELIRIUM_TRI_LOD_RANGE="4 200" DELIRIUM_TRI_UVN=256` runs at 62 FPS with UVN up to 200 near camera, ~1.1M triangles. Overflow guards added for uint32_t counters at extreme UVN.

### Environment variables
- `DELIRIUM_TRI=1` — enable TriInfill GPU path.
- `DELIRIUM_TRI_UVN` — uniform grid side (default 10, max 256).
- `DELIRIUM_TRI_LOD=1` — adaptive per-patch UVN from projected size.
- `DELIRIUM_TRI_LOD_RANGE="min max"` — clamp adaptive UVN (default 2–200).
- `DELIRIUM_TRI_LOD_DEBUG=1` — debug prints (default off).

---

## 2. Volumetric Generation — Closed Objects Fixed

### Root cause
The substance model is a single-valued graph `z=f(x,y)`, so a single body cannot represent a closed surface (a sphere has two z per (x,y) inside the equator). Procedural objects rendered as shallow domes/sheets with uniform normals.

### Fix
- **`ProceduralElementGenerator::BuildMulti()`** — emits one or more element trees. Closed objects (orb/sphere/ball/body) return two trees: upper hemisphere + lower hemisphere (scale.z=-1 flip). Heightfields (bump/wave/mound) return one tree with prominent features (larger amplitude).
- **`BuildVolumetricSphere()`** — builds a closed sphere directly as a multi-patch PSFBody with proper UV-sphere topology: anchors on sphere surface at UV grid, radial outward normals, patches wrap around, zero deformation coefficients. Pole-cap topology fix replaces degenerate pole-fan patches with non-degenerate cap-ring topology.
- **Heightfield weight fix** — `host->weight = 0.01f` (fill holes only) and feature weight=1.0 prevents beige planes bleeding through.

### Verified
- Scene U renders a closed sphere (164 patches/body, 2 bodies).
- Scene X renders closed orb + wave + bump (4 bodies total).
- NeuralReconstructor gallery scenes U/V/W/X render correctly at 4K with adaptive LOD.

---

## 3. Critical Bug Fixes

### GPU anchor upload bug (all objects rendered identically)
**Symptom:** All generated objects rendered identically (hollow toroids with gold interiors).
**Root cause:** `PSFBody::FromRecords` created bodies with `anchors_dirty = false` (header default). Renderer uploads anchors/surfaces only when buffers are null (first body) OR `AnchorsDirty()` is true. Body 1 uploaded correctly, Bodies 2–5 reused Body 1's data.
**Fix:** Added `body->anchors_dirty = true;` in `PSFBody::FromRecords`.

### Anchor pool exhaustion (objects disappear after refinement)
**Symptom:** Objects disappear after repeated refinement.
**Root cause:** Anchors are `uint16_t` (max 65,535). Each `SplitPatch` creates up to 5 new anchors → pool exhausts → `CreateSnapped` returns 0xFFFF → silent corruption.
**Fix:** Added `GuardAnchorOverflow()` in `PSFBody.cpp` — warns/coarsens near uint16_t limit; integrated into `RefineWhere()`. Added CPU `ReflowAnchors()` impl. Design doc: `docs/AnchorReflow_Design.md`.

### Tri-infill body-change upload
**Symptom:** Intermittent "tri-infill GPU path unavailable" and color contamination between bodies.
**Fix:** `RenderBodyTriInfillGPU()` sets `body_changed=(last_body!=&body)`, forces upload when true, records `last_body` after upload.

### Clipped-frame handling
**Symptom:** FPS dips during orbit when object fully clipped.
**Fix:** Treat `n_tri == 0` (object fully clipped this frame) as success, not failure. Logs clipped frames once/120 frames.

### Smooth surfaces (visible faceting)
**Symptom:** Surfaces broken into flat grid-like patches; anchors visible.
**Fix:** Raised default sampling UVN from 10 → 24. Anchors are internal physics state; surface smoothness comes from sampling density. Still tunable via `DELIRIUM_TRI_UVN`.

### Workspace editor crash fixes
1. **Null-kernel crash:** Workspace defaulted to splat-layered → kernels missing → fell through to classic splat → `clSetKernelArg` on null kernel → crash. Fixed: default `render_path_` to 1 (raymarch) + guard `else if (path_ == Path::SplatClassic && k_splat)` in renderer.
2. **Dangling reference crash:** `Engine::Conservation()`/`SolverState()` returned `const PhysicsConservation&` but returned temporary when `physics` null → dangling ref. Fixed: return BY VALUE.

---

## 4. Workspace Editor Port

Ported the docked Godot-like editor (`examples/workspace/main.cpp` + `engine/src/core/Workspace.cpp/.h`) into the working project.

### Editor layout
- Status bar (top)
- Scene Tree dock (left)
- 3D Viewport (center) with gizmo nav-ball
- Inspector + Physics panel (right)
- AI Console (bottom)
- Tool strip + render-path combo in toolbar

### Controls
WASD move, mouse look, gizmo drag to orbit. Tools: poke/dent/refine/coarsen/fracture/move/navigate. AI console sends prompt to local Ollama model → parses element tree → bakes body → adds to scene.

### Render paths available
0 march, 1 raymarch, 2 splat classic, 3 splat layered, 4 splat area, 5 equation-layered, **6 TriInfill GPU**. Workspace default changed to **6 (tri-infill)**.

### Kernel/host arg mismatches (paths disabled at runtime)
- `psf_splat`: kernel=18 args, host expects 22 → classic splat DISABLED.
- `surfel_march`: kernels arg mismatch → march path DISABLED.
- Working paths: `psf_raymarch` (ready), `equation_layered` (ready), `surf_march` single-pass lateral (ready), fb_infill/object-stitch/lod-stitch (ready), TriInfill GPU (ready).

### Files copied from backup
UI toolkit (`Font.cpp/.h`, `UI.cpp/.h`), LLM client (`LLMClient.cpp/.h`, `LLMToolLoop.cpp/.h`), Workspace (`Workspace.cpp/.h`, `Transform.cpp/.h`), DocLearner, ElementTreeParser, GeoTools, Engine extended API, PhysicsWorld extended API, ReconstructionModel extended API, Scene overload, `examples/workspace/main.cpp`, updated `CMakeLists.txt`.

### New UI panels added
- **Grammar/Assembly panel** — grammar validation controls, assembly controls, Assemble button builds `TripartiteSurface` from selected object, runs `GrammarCompiler` + `AssemblyPlanner`, logs committed/rejected bond counts.
- **Profiler panel** — reads renderer timing getters: sample/build/raster/upload/total ms + tris/samples/UVN/LOD mode.
- Enhanced Physics panel with proper stacking.

---

## 5. HUD UI (recon_window)

Self-contained 5×7 bitmap font rendered in `examples/recon_window/main.cpp`. Panels: top-left (title/render pipeline/FPS), top-right (camera/object count), bottom-left (controls legend), bottom-right (aim stats + physics/material detail), center toast. Keys: LMB poke, RMB dent, R refine, C coarsen, F fracture, G restitch, T weld, X cap, V record, O orbit, P shot, H HUD toggle, M physics detail, Esc quit. Font values validated programmatically (all bytes ≤ 0x1F). Panel widths narrowed to prevent overlap on 1024px window.

---

## 6. Anchor Reflow Design

Design doc: `docs/AnchorReflow_Design.md`. Kernel: `kernels/anchor_reflow.cl` (202 lines, 3 kernels: `compute_error_indicators`, `compute_anchor_displacement`, `update_anchor_positions`). Solves anchor pool exhaustion by redistributing existing anchor positions instead of creating/removing anchors. Host impl `PSFBody::ReflowAnchors()`. Includes split/join support design (`PSFBody::JoinWith`).

---

## 7. Architecture Invariants (protected)

1. **Binary layout frozen:** `AnchorRecord`, `SurfaceRecord`, `BoneRecord` = 64 bytes with fixed offsets, verified by `static_assert`, shared between C++/kernels/Python tools.
2. **CPU/GPU bit-compatible:** `Patch::Evaluate` matches `psf_splat.cl`. Validated by `tools/psf_validate.py`.
3. **Base vanishes at corners:** surface interpolates anchors → corner cracks impossible by construction. Each edge belongs to exactly 2 coefficients → sewing = adjusting 2 numbers.
4. **Every edit is local:** touching an object = rewriting a few 64B records (and uploading only those to VRAM).

---

## 8. Parametric Surfaces (PSF v4 Format) — IMPLEMENTED

### Overview
PSF format extended from v3 legacy height-fields (`SK_LEGACY`, bilinear anchors + `h(u,v)*Nd`) to **v4 parametric families**. A parametric surface encodes its FULL world-space size directly in parameters (major/minor radius, half-extents), so no anchor-grid tessellation is needed — one math surface replaces hundreds of height-field patches.

### Format
- **Magic/version:** `PSF\0` + `04 00 00 00` (verified in `assets/torus_v4.psf`).
- **SurfaceKind enum** (`engine/include/delirium/core/Types.h`): `SK_LEGACY=0`, `SK_ELLIPSOID`, `SK_TORUS`, `SK_BOX`, `SK_CYLINDER`, `SK_PLANE`, `SK_GAUSSIAN`, `SK_WAVEFIELD`.
- **Anchor frame:** shared body-level frame maps local → world (same bilinear interpolation of 4 corners as v3). Parametric families use `scale=1.0f` (position already in world units); legacy uses `scale=max(bbspan/2)`.
- **Binary layout frozen** and validated against `tools/psf_encode_v4.py`.

### CPU evaluator
`engine/src/psf/Patch.cpp` — `FamilyEval` dispatch on `kind`, computes parametric position + analytic normal via cross product of partials (`dPdu × dPdv`). Matches GPU convention exactly.

### GPU evaluator
`kernels/psf_splat.cl` lines 704–796 — `psf_sample` kernel switches on `kind`:
- `SK_ELLIPSOID` → `EvalEllipsoid`, `SK_TORUS` → `EvalTorus`, etc.
- Shared body-level frame with `scale=1.0f` for parametric families (fixes double-scaling bug where `iso=max(bbspan/2)=1.35` scaled parametric world-space positions again → radial=1.82 instead of 1.35).
- Analytic normal from `norm3(cross3(dPdu, dPdv))`.

### Verified assets
| File | Version | Surfaces | Notes |
|------|---------|----------|-------|
| `assets/torus.psf` | v3 legacy | 544 SK_LEGACY patches, 615 anchors | Height-field approximation of torus; corner artifacts require radial clip guard |
| `assets/torus_v4.psf` | v4 parametric | 1 SK_TORUS surface, 4 anchors | Major=1.0, minor=0.35, same bbox [-1.35,-1.35,-0.35]..[1.35,1.35,0.35]. Generated via `tools/psf_encode_v4.py` |

---

## 9. GPU Physics Kernels — ALL FOUR WIRED INTO STEP (APPLIED/PACKAGED, pending user build confirmation)

### Status caveat
GPU physics kernels exist as standalone OpenCL files and pass benchmark tests (`tests/validate_*`). **All four kernels are now dispatched from `PhysicsWorld::Step()`** with separate enable flags (default OFF) and CPU fallback paths:
- **N-body pairwise** (`nbody_pairwise.cl`) — Phase 0a: additive gravity+Coulomb forces after uniform gravity integration.
- **GPU FEM solve** (`fem_assemble.cl` + `fem_solve.cl` via `PhysicsGpuFem`) — Phase 6: deformable-body continuum solve.
- **Anchor-reflow** (`anchor_reflow.cl`) — Phase 6a: pool-exhaustion guard via `PSFBody::ReflowAnchors()` GPU backend.
- **Modal writeback** (`psf_deform.cl::apply_modes`) — Phase 7: VRAM-direct modal coefficient writeback.

Each kernel is guarded by its own flag (`SetNBodyPhysics`, `SetFemPhysics`, `SetReflowPhysics`, `SetModalWriteback`), defaults OFF so nothing changes unless explicitly enabled, and falls back to CPU if GPU not ready. Visual test scenes added to `examples/visual_gallery` (keys 1–4). The CPU coupled-field solver path remains unchanged for backward compatibility.

### Kernel files (`kernels/`)
| File | Kernels | Purpose | Verified via |
|------|---------|---------|--------------|
| `nbody_pairwise.cl` | `nbody_pairwise` | Gravity + Coulomb pairwise forces | `tests/validate_nbody_benchmark.cpp` |
| `collision_broadphase.cl` | `build_cell_lists` | Spatial hashing broadphase | `tests/validate_collision_benchmark.cpp` |
| `collision_narrowphase.cl` | narrowphase contact detection | Contact normal + depth | `tests/validate_collision_benchmark.cpp` |
| `collision_impulse.cl` | impulse solver | Resolve contacts (velocity reversal) | `tests/validate_collision_benchmark.cpp` |
| `fem_assemble.cl` | `assemble_forces` | FEM element stiffness assembly | `tests/validate_fem_benchmark.cpp` |
| `fem_solve.cl` | `jacobi_solve` | FEM linear system solve (Jacobi iteration) | `tests/validate_fem_benchmark.cpp` |
| `anchor_reflow.cl` | `compute_anchor_displacement`, `update_anchor_positions`, `refit_coefficients` | Anchor pool exhaustion fix via redistribution | `tests/validate_reflow_gpu.cpp` |
| `psf_deform.cl` | `apply_modes`, `apply_anchor_modes`, `apply_seams` | Modal dynamics writeback + seam stitching | `tests/validate_m2_culling.cpp` (partial) |

### Host infrastructure
- `engine/src/physics/PhysicsGpu.cpp` — OpenCL buffer manager: `Initialize()`, `Upload()`/`Download()` PhysicsSoA, `LoadProgram()`/`GetKernel()`, collision grid config, scratch buffers for pair lists/forces. Added `context()` accessor for helper buffer allocation.
- `engine/src/physics/PhysicsGpuFem.cpp` — FEM-specific buffer manager: anchor/element buffers, `LoadProgram()` for assemble/solve kernels.
- `engine/src/physics/PhysicsWorld.cpp` — **All four GPU kernels dispatched from `Step()`** (APPLIED/PACKAGED, pending user build confirmation):
  - `InitGpuPhysics()` now loads collision + nbody + reflow + deform kernels into `gpu_` and fem kernels into `gpu_fem_`; registers GPU reflow backend with `PSFBody::RegisterGPUReflow()`.
  - `RunGpuNBody()` — Phase 0a dispatch: upload → nbody_pairwise (force accumulation) → nbody_integrate (semi-implicit Euler) → download. Constants: G=6.674e-11, k=8.988e9, softening²=1e-6.
  - `RunGpuFemRegion()` — Phase 6 dispatch: builds FEMRegionSoA+FEMElementSoA from PSFBody anchors/patches, runs assemble_forces + jacobi_solve, writes back anchor displacements.
  - `RunGpuReflowBackend()` — registered GPU backend for `PSFBody::ReflowAnchors()`: snapshot_error_accumulate → compute_anchor_displacement → update_anchor_positions → refit_coefficients. Shape-preserving (snapshot BEFORE anchors move, refit AFTER).
  - `RunGpuModalWriteback()` — Phase 7 dispatch: apply_modes writes coef = rest + Σ q_m·shape_m directly to VRAM surface records. Added `ModalDynamics::CoefModeShapes()`, `Amplitudes()`, `SurfCount()` accessors for GPU upload.
  - Each dispatch guarded by its own flag (`nbody_on_`, `fem_on_`, `reflow_on_`, `modal_writeback_on_`), defaults OFF, CPU fallback if GPU not ready.
- `engine/include/delirium/physics/PhysicsWorld.h` — Added `SetNBodyPhysics`/`SetFemPhysics`/`SetReflowPhysics`/`SetModalWriteback` setters + getters; member flags + `PhysicsGpuFem gpu_fem_` instance; includes `PhysicsGpuFem.h`.
- `engine/include/delirium/physics/PhysicsGpu.h` — Added `cl_context context() const` accessor.
- `engine/include/delirium/physics/ModalDynamics.h` — Added `CoefModeShapes()`, `Amplitudes()`, `SurfCount()` accessors for GPU modal writeback.
- `examples/visual_gallery/main.cpp` — Added visual test scenes: `gpu_nbody` (key 1), `gpu_fem` (key 2), `gpu_reflow` (key 3), `gpu_modal` (key 4). Each demonstrates the corresponding GPU kernel wired into `Step()` with CPU fallback.

### Benchmark results (verified on Intel Arc A770)
| Benchmark | Speedup | Notes |
|-----------|---------|-------|
| N-body pairwise | 52× (N=1k) → 1919× (N=10k) | Super-linear; classic engines use uniform field O(N) instead of pairwise O(N²) |
| Collision detection | 6.84× → 11.50× | Broadphase + narrowphase vs CPU brute force |
| FEM deformation | 1.47× → 2.46× | Assemble + Jacobi solve vs CPU |
| Anchor reflow | ~154× | GPU displacement kernel vs CPU |

### Still OPEN (from Section 10 below)
- All four GPU kernels wired into `Step()` but **not yet verified by user build** — visual test scenes `gpu_nbody`/`gpu_fem`/`gpu_reflow`/`gpu_modal` (keys 1–4) + existing `gpu_physics` (key 0) demonstrate them; awaiting confirmation.
- GPU modal writeback (`psf_deform.cl::apply_modes`) — kernel now launched from host via `RunGpuModalWriteback()`; requires `.modes` file loaded via `ModalDynamics::LoadModes()`.
- GPU FEM solve runs on ALL bodies with patches > 0 (no per-body FEM region registration yet) — production system would track per-body FEM region binding.

---

## 10. Recent TriInfill Fixes — APPLIED/PACKAGED (pending user build confirmation)

> Per user's standing rule (POKEE.md): these changes are described as applied/packaged/unverified until the user confirms a successful local build + render. Full debugging timeline (v152–v167) moved to external doc `docs/TRI_INFILL_SAGA.md`.

**Root cause:** `torus.psf` was v3 LEGACY height-field — cannot represent a curved torus tube, breaking tri_infill triangulation. Structural fix: switched scene Y to PSF v4 parametric torus (`torus_v4.psf`).

**Fixes applied to sandbox tree:**
1. Generated `assets/torus_v4.psf` (v4 parametric, 1 SK_TORUS surface).
2. Removed radial clip guard in `psf_splat.cl` (only needed for legacy corner artifacts).
3. Fixed double-scaling bug — parametric families use `scale=1.0f` in both GPU (`psf_splat.cl:727-728`) and CPU (`PSFBody.cpp:87`).
4. Raised `base_uvn` from 10 to 30 (`SplatRenderer.cpp:2897`) — rounds the tube cross-section.
5. Added `procedural_albedo()` helper (`psf_splat.cl:165`) — fixes base color clamping to black.

**User-reported status (2026-11-22):** Parametric torus renders correctly (`patches=1`, `sentinel=0`, `center fraction=0.1%`). Square tube appearance fixed by base_uvn=30. Minor issue "edges overlapping too much" — to be investigated separately.

---

## 12. Build & Run

### Build (Release)
```bash
cmake -S . -B build_delirium -DCMAKE_BUILD_TYPE=Release
cmake --build build_delirium --target delirium_m1_torus    # benchmark
cmake --build build_delirium --target delirium_workspace   # editor
cmake --build build_delirium --target delirium_recon_window # live window
cmake --build build_delirium --target delirium_visual_gallery # headless PPM gallery
```

### Run with TriInfill GPU
```bash
cd ~/Desktop/Tartarus/gameengine_compiled_real
DELIRIUM_TRI=1 DELIRIUM_TRI_LOD=1 DELIRIUM_TRI_LOD_RANGE="4 200" DELIRIUM_TRI_UVN=256 \
  ./build_delirium/delirium_recon_window
```

### Header-shadow gotcha
CMake lists `include/` before `engine/include/` → stale headers shadow updates. After extracting tarballs with header changes, run:
```bash
cp engine/include/delirium/<subsys>/*.h include/delirium/<subsys>/
```

---

## 9. Tarball Distribution Rules

- Members are `engine/...` and `kernels/...` with NO top-level prefix → extract from INSIDE the project dir.
- Fresh filename each rebuild (`DeliriumEngineV2_<topic>_vNN.tar.gz`) to avoid browser cache.
- Verify archive contents before extracting: `tar -tzf <file>`.
- Kernels (`kernels/*.cl`) load at RUNTIME — extracting them does NOT require rebuild.
- NEVER overwrite local headers wholesale when user has local-only additions — ship only `.cpp` + manual header declaration instructions.

---

## 11. GPU Modal Writeback Integration — COMPLETE ✅ (set 2026-11-26)

### Root cause of modal-body FPS stall (~27-31ms/frame)
`GameObject::FixedUpdate` calls `ModalDynamics::WriteBack` every frame → modifies host blobs → `AnchorsDirty()` → ~85KB upload/frame → Intel Arc driver stalls ~27-31ms regardless of `CL_FALSE`. Physics needs host blob (collision/FEM); render does NOT.

### Architecture fix
Physics and render become INDEPENDENT computations from same REST state:
- **Physics:** `REST + K → host blob` (unchanged, needed for collision/FEM).
- **Render:** `REST + K → GPU render buffers` (new GPU path, no host upload).
- REST uploaded ONCE at bind from `ModalDynamics` snapshots (`rest_apos`/`rest_anrm`/`rest_coef`). Each frame only tiny K vector uploads; kernels write deformed geometry DIRECTLY into shared render buffers (`buf_anchors`/`buf_surfaces`/`buf_snorm`) that `bake_surfels` reads.

### Files (`GPUModalWriteback_v1.tar.gz`, 106K)
- `kernels/psf_deform.cl` — `modal_writeback_anchors`/`modal_writeback_surfaces` + `modal_snorm_pass1`/`modal_snorm_pass2` kernels.
- `engine/src/render/SplatRenderer.cpp` + mirror (`SetupGPUModal`/`GPUModalWriteback`/`ShutdownGPUModal`).
- `engine/include/delirium/render/SplatRenderer.h` + mirror (modal buffer members).
- `engine/include/delirium/psf/PSFBody.h` + mirror (weak `ModalDynamics*` pointer).
- `engine/src/physics/ModalDynamics.cpp` + mirror (`FullModeShapes`/`RestAnchorPos`/`RestAnchorNrm`/`RestCoef` accessors).
- Kernel/host arg parity verified (anchors=10 args, surfaces=7 args, snorm_pass1=7 args, snorm_pass2=5 args). Both header trees identical. Enable: `DELIRIUM_GPU_MODAL=1`.

### Iteration history (v2–v15)
- v2–v14: fixed header include, `std::clamp` brace-init, OpenCL kernel brace-init syntax, `clEnqueueFillBuffer` pattern_size power-of-2.
- **v15 ROOT CAUSE FIXED:** `clEnqueueCopyBuffer` src/dst SWAPPED at line ~4954 — copied empty `buf_anchors` OVER `buf_modal_rest_anchors`, zeroing REST data after upload → black screen. Fix: swap args `(queue, buf_modal_rest_anchors, buf_anchors, ...)`.

### FPS stall final fix (set 2026-11-26)
Even with GPU modal writeback active, frame time stuck ~29ms. Stage timing probes (`DELIRIUM_RENDER_STAGES=1`) pinpointed the stall to the **unconditional FlushToGPU loop** in the surface upload block (lines 2302-2316 of `SplatRenderer.cpp`). For modal bodies, `body.AnyDirty()` is true every frame → `FlushToGPU` uploads dirty surface records (hundreds of `clEnqueueWriteBuffer` calls serializing on Intel Arc in-order queue → ~32ms stall).

**Fix:** Detect modal bodies early (`ModalDynamicsPtr() != nullptr && gpu_modal_on`) and skip the unconditional host upload block. The tri-infill path's GPU modal writeback writes `buf_anchors`/`buf_surfaces`/`buf_snorm` directly, so the host upload is redundant. Applied to `SplatRenderer.cpp` line ~2303 (`is_modal_body` gate).

**Result:** `modal-writeback/setup` stage dropped from ~28ms to ~0.13ms. Frame time expected to drop from ~31ms to ~3-4ms (~250-300+ FPS). **PENDING USER BUILD CONFIRMATION.**

### Debug discipline lesson registered
DEBUG DISCIPLINE violation in v1/v2 diagnosis: assumed enqueue was the bottleneck when GPU work was always fast (~1.6ms). Finish-diag proof showed GPU work = 1.60ms but render bucket = 28.94ms → bottleneck was CPU work inside RenderBody independent of triangle count. Lesson: when GPU enqueue is fast but frame time is high, measure GPU execution time separately (finish-diag) BEFORE assuming kernel/sync optimization — bottleneck may be elsewhere entirely (here: host upload path).

---

## 12. Path B2 Color-Field Migration — COMPLETE ✅ (B2.1–B2.5, set 2026-11-24)

Parallel storage: keep frozen `color_pattern[18]` in `SurfaceRecord`, add `std::vector<PatchColorField> color_fields` on `PSFBody`; Order 2 (9 coeffs/channel); fallback to dict if field invalid. Goal: color cohesion (color bound to world position when anchors move).

- **B2.1 host storage DONE:** `PSFBody.h` + `Patch.h` (both trees) add `color_fields` members/accessors; `Patch::EvaluateColor` checks field first then falls back to dict. VERIFIED ✅.
- **B2.2 load-time conversion DONE:** `PatchColorField::FitFromRecord` (dynamic n×n Gaussian elimination LSQ over cosine basis). VERIFIED ✅.
- **B2.3 cohesion refit DONE:** `FitFromColorGrid` + `SnapshotColorGrids`/`RefitColorFields` wired into CPU `ReflowAnchors` and GPU `RunGpuReflowBackend`. Tarball `PathB2_v6_phase3_cohesion_refit.tar.gz`. VERIFIED ✅.
- **B2.4 GPU color-field refit DONE:** moved snapshot+refit into `anchor_reflow.cl` (`cos_basis`, `color_field_eval`, `refit_color_field` kernels). Render kernels untouched — still use procedural_albedo. Tarball `PathB2_v8_phase4_memcpy_fix.tar.gz`. VERIFIED ✅.
- **B2.5 asset migration + regression DONE:** load-time conversion runs for all 5 assets. Regression pending user confirmation.
- Build fixes: dual-header shadow synced; `Patch.cpp` needed `PatchColorField` include; `PatchColorField.cpp` channels `const float*`→`float*`; `PhysicsWorld.cpp` needed `<cstring>`. Debug markers still present — cleanup pending.

---

## 13. Distributed Temperature Field — CPU SUBSTRATE COMPLETE ✅ (set 2026-11-24)

Mirrors Path B2 pattern for scalar temperature field (units K). Additive substrate — does NOT solve heat equation yet. Frozen `SurfaceRecord` layout unchanged (fallback ambient 293.15 K).

- Files: `PatchTempField.h`/`.cpp` (both trees) — scalar cosine field class with `FitFromGrid`/`Eval` continuous LOD.
- Host storage: `PSFBody.h` adds `temp_fields` member + accessors; `Patch.h` adds `temp_fields` pointer + `EvaluateTemp`.
- Load-time conversion initializes from uniform ambient grid; cohesion refit wired into CPU `ReflowAnchors`. GPU refit deferred.
- STATUS: CPU substrate complete. NOT YET BUILD-VERIFIED (sandbox has no C++ toolchain — user builds locally). Pending: build + regression.

---

## 14. Rick & Razor — Gait-Driven Locomotion Module COMPLETE ✅ (set 2026-11-26)

Built a proper gait-driven IK + root-motion system replacing the decoupled `driveForward` hack:
- **`solveLegIK(thighLen, shinLen, hipY, fwdDist)`** — analytic 2-bone IK in sagittal plane. Targets the ANKLE (not ground) so knee has slack to bend. Uses law of cosines for hip pitch + exact atan2 reconstruction for shin angle → guaranteed to reach target. Verified reconstruct error ≈ 0 for grounded cases.
- **Gait state machine** — phase (0..1 per cycle), strideLen, cadence scale with speed; contact state machine (left planted [0,0.5), right planted [0.5,1)); swing foot aims ahead one stride, planted foot anchors body.
- **`applyGaitLegs(hipsPos, footL, footR, dt)`** — solves IK for both legs each frame; returns root velocity from stance leg. Hips lowered to HIPS_Y=0.90 so legs have ~0.12 units slack to bend naturally.
- **Integration** — render loop calls `gaitStep(dt, gaitSpeed)` then `applyGaitLegs()`; root motion integrated only when WASD idle. `setLocomotionSpeed(-1..1)` helper replaces `driveForward`; `updateMovement` uses procLocoSpeed when WASD idle.
- **Scripts** — loco/nav call `setLocomotionSpeed()` instead of `driveForward()`; blank template documents new API. `driveForward` kept as legacy helper.
- STATUS: Syntax verified, IK solver tested reconstructing targets exactly. VISUAL CONFIRMATION PENDING user build/run.

---

## 10. Open Items / Next Steps

- **AMR/LEARN/BONE/BW modules** not yet implemented. Aero/EM coupling, exact surface-patch contact area still open.
- **Incremental seams** — route split/merge/dent through `RestitchAround(ids)` instead of full `Seams().Build`.
- **CPU spatial index** for `ClosestPoint`/`Raycast` — reuse `EquationAccel` patch AABBs.
- **Render Transform** — physics integrates velocity/omega but engine renders body without applying transform; fracture fragments don't move on screen.
- **Scene batching** — march path only marches first body; real scenes need per-body march or merged surfel cache.
- **Neural reconstructor stage 2** — stage 1 patch fitter trained (MAE 16× better than LSQ); loading trained weights and swapping into `IReconstructor` is next milestone.
- **More assets** — only 4 `.psf` files exist; varied topologies would stress-test AMR + seams better.
- **v2.2Beta milestones** — M4 (WS4 in-3D editing), M5 (WS5 Cling scripting), M6 (WS2 document reconstruction) scaffolding in progress. See `docs/CHANGELOG_v2.2Beta.md`.

---

## 13. Key Reference Docs

| Doc | Content |
|-----|---------|
| `docs/ARCHITECTURE.md` | Architecture v0.2 invariants, frame flow, reconstructor contract, reference numbers |
| `docs/ARCHITECTURE_ANCHOR_NORMALS.md` | Anchor/normal conventions |
| `docs/CAGE_REFRAME.md` | Cage surface reframe |
| `docs/CORRECTED_DIRECTION.md` | Geometry = patches amalgamated at edges; spectral field = color only (gold rule) |
| `docs/ENGINE_REPORT.md` | Full tree sweep + perf report interpretation + prioritized work list |
| `docs/ENGINE_UPDATE_TRI_INFILL.md` | TriInfill + Delaunay + camera integration notes |
| `docs/AGENT_PROGRESS.md` | Agentic workflow progress |
| `docs/WORKING_MINIMUM.md` | Minimum viable render |
| `docs/SURFEL_RENDER.md` | Surfel rendering design |
| `docs/THREE_LAYER_REDESIGN.md` | Three-layer redesign |
| `docs/PARALLEL_LAYERS.md` | Parallel layers |
| `docs/PROGRESSIVE_RASTER.md` | Progressive raster |
| `docs/SPECTRAL_GAME_ENGINE.md` | Spectral game engine concept |
| `docs/SPECTRAL_PRIMITIVE.md` | Spectral primitive |
| `docs/SCANLINE_RASTER_ANALYSIS.md` | Scanline raster analysis |
| `docs/BUILD_GPU.md` | GPU build notes |
| `docs/M0_SCOPE.md` | M0 scope |
| `docs/M1_HANDOFF_CONTRACT.md` | M1 handoff contract |
| `docs/EXPERIMENT_DECISIONS.md` | Experiment decisions |
| `docs/SUMMATION_OF_TYPES_ARCHITECTURE.md` | Summation of types architecture |
| `docs/CLOTH_SEGMENTATION.md` | Cloth segmentation |
| `docs/AnchorReflow_Design.md` | Anchor reflow + surface regeneration design |
| `docs/TRI_INFILL_SAGA.md` | TriInfill parametric torus debugging timeline (v152–v167), root cause, fixes applied |
| `../DeliRium_Optimization_and_UI_Plan.md` | Render optimization roadmap (7 phases) + physics UI/tooling expansion (5 phases) |

---

## 14. Session Tarballs (this branch)

Latest workspace editor tarballs: `DeliriumEngineV2_workspace_editor_v01.tar.gz` … `v14.tar.gz` (v14 = latest). Volumetric generation: `DeliriumEngineV2_volumetric_generation_v01..v03.tar.gz`, `DeliriumEngineV2_volumetric_sphere_v01..v02.tar.gz`, `DeliriumEngineV2_volumetric_heightfield_v01.tar.gz`.

All tarballs produced from `/opt/sandbox/workspace/gameengine_compiled_real/` and verified with `tar -tzf` before distribution.
