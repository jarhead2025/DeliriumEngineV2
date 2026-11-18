# DeliRium Engine V2 — Progress Log (v2.1)

**Branch:** `v2.1`
**Compiled:** 2026-11-18
**Status:** TriInfill GPU primary renderer working + optimized; Workspace editor ported; volumetric generation fixed.

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

## 8. Build & Run

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

## 10. Open Items / Next Steps

- **AMR/LEARN/BONE/BW modules** not yet implemented. Distributed surface temperature field, aero/EM coupling, exact surface-patch contact area still open.
- **GPU modal writeback** via `psf_deform.cl::apply_modes` — kernel exists but not launched.
- **Incremental seams** — route split/merge/dent through `RestitchAround(ids)` instead of full `Seams().Build`.
- **CPU spatial index** for `ClosestPoint`/`Raycast` — reuse `EquationAccel` patch AABBs.
- **Render Transform** — physics integrates velocity/omega but engine renders body without applying transform; fracture fragments don't move on screen.
- **Scene batching** — march path only marches first body; real scenes need per-body march or merged surfel cache.
- **Neural reconstructor stage 2** — stage 1 patch fitter trained (MAE 16× better than LSQ); loading trained weights and swapping into `IReconstructor` is next milestone.
- **More assets** — only 4 `.psf` files exist; varied topologies would stress-test AMR + seams better.

---

## 11. Key Reference Docs

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
| `../DeliRium_Optimization_and_UI_Plan.md` | Render optimization roadmap (7 phases) + physics UI/tooling expansion (5 phases) |

---

## 12. Session Tarballs (this branch)

Latest workspace editor tarballs: `DeliriumEngineV2_workspace_editor_v01.tar.gz` … `v14.tar.gz` (v14 = latest). Volumetric generation: `DeliriumEngineV2_volumetric_generation_v01..v03.tar.gz`, `DeliriumEngineV2_volumetric_sphere_v01..v02.tar.gz`, `DeliriumEngineV2_volumetric_heightfield_v01.tar.gz`.

All tarballs produced from `/opt/sandbox/workspace/gameengine_compiled_real/` and verified with `tar -tzf` before distribution.
