# DeliRium Engine — Exploration Report + Work List

Generated 2026-10-26 from a full sweep of `gameengine_compiled_real/` plus the
A770 perf run. Companion deliverables:

- `examples/visual_gallery/main.cpp` — headless render gallery (20 PPMs)
- `examples/playground/main.cpp` — live interactive window (every subsystem)
- `tests/perf_engine.cpp` — the perf report you just ran

---

## 1. Run the new visual tests

### Headless gallery (no window, works over SSH)

```bash
cmake --build build_delirium --target delirium_visual_gallery
./build_delirium/delirium_visual_gallery              # -> visual_gallery/*.ppm
# options:
./build_delirium/delirium_visual_gallery --list       # manifest only
./build_delirium/delirium_visual_gallery --only A,B,L # just those scenes
./build_delirium/delirium_visual_gallery --size 960x720 --out /tmp/gal
```

Each scene loads a **fresh body** (independent), renders one frame, saves a PPM.
The console prints "what to look for" per scene. Scenes:

| # | Scene | What it exercises | Look for |
|---|-------|-------------------|----------|
| A | march_default | layered lateral march (default path) | baseline coverage, hole-fill quality |
| B | raymarch | GPU equation ray-march + glow | emissive look, edge softness |
| C | splat_classic | anisotropic splat (`psf_splat.cl`) | sample-density streaks vs A |
| D | march_highres | march UVN=12, 64 steps | sharper edges than A, slower |
| E | amr_refine | live `SplitPatch` on +X side | fine patches one side, coarse other |
| F | amr_lod | refine +X, coarsen −X | **LOD seams at the size transition** |
| G | dent | `ApplyRadialDent` | smooth dimple vs A |
| H | dent_refit | dent + `RefitLocal` | coefs re-optimized (compare G) |
| I/J | mode0/mode1 | ROM `WriteBack` static deflection | global bending shape |
| K | modal_ring | impulse + 24 modal steps | frozen wave mid-flight |
| L | fracture | `DetachPatches` + offset | the crack + floating fragment |
| M | morph_color | pattern K edit | color shift where pattern 1 lives |
| N/O/P | sphere / asteroid / roughball | other `.psf` assets | robustness on different topologies |
| Q | closeup | tight camera | patch detail, seam visibility |
| R | orbit | low wide angle | back side, silhouette |
| S | split_edges | a few splits | new seams at child edges |
| T | weld | perturb anchors + `HarmonicWeld` | seams closed back (gap printed) |

### Live playground (window)

```bash
cmake --build build_delirium --target delirium_playground
./build_delirium/delirium_playground
```

Mouse-look aim (window center), wheel zoom. Keys: **LMB** poke (modal ring),
**RMB** dent, **R** refine, **C** coarsen, **F** fracture, **M** morph,
**G** restitch, **T** weld, **1/2/3** switch render path live (march /
ray-march / splat — this is the fastest way to A/B the three paths),
**O** auto-orbit, **P** screenshot. WASD/Space/Ctrl move the camera. The
modal loop runs on the engine's 120 Hz fixed step, so pokes ring and decay
in real time.

---

## 2. Tree map (what's where)

```
engine/include/delirium/   41 headers        engine/src/   32 sources
  psf/       PSFBody (live topology: Split/Merge/Detach/Refine/Coarsen/
             Dent/RefitLocal/ApplyMode/FlushToGPU), Patch (eval+derivs+
             color K), SeamGraph (stitch/HarmonicWeld/HarmonizeEdgeCoefs/
             BridgeSeams), AnchorPool, Reconstructor, CageSurface,
             ProceduralDetail, PatternLibrary, SurfaceTypeLibrary, TreeFormat
  render/    SplatRenderer (3 paths: march / raymarch / splat), EquationAccel,
             LateralRayMarch, TriInfill, SurfelRasterizer, Camera
  physics/   ModalDynamics (12 ROM modes), PhysicsWorld (rigid + weld + FEM
             promote), LocalFEM, CollisionAnalytic, Material (+stress-strain)
  core/      Engine (SDL window, 120Hz fixed step, input/hud/render hooks),
             GameObject (Poke/Dent/Fracture/Morph/RecalculateStructure),
             Scene, SceneTree, Node, Editor
  ai/        ReconstructionModel (RMW1 loader + Validate; Infer = fallback)
  io/        PSFFile (protected load path)
  servers/   RenderServer, PhysicsServer (Godot-like layer)
  scene/     SplatView

kernels/     13 OpenCL kernels:
  wired:     psf_splat, surfel_march (6 kernels), psf_raymarch, fb_infill,
             fb_lod_stitch, fb_object_stitch
  impl, no host caller: psf_deform, psf_refine   <- implemented, never launched
  unreferenced: hit_fill_stub, spf_splat, surfel_inverse, surfel_splat

assets/      torus.psf (615 anchors / 544 surfaces / 12 modes), sphere.psf,
             asteroid.psf, roughball.psf, spot/ (surfel format)
tools/       obj_to_psf.py, psf_modes.py (offline mode assembly),
             march_gpu_render.cpp, march_gpu_interactive.cpp
training/    neural patch-fitter (stage 1 validated: MAE 0.28 -> 0.018)
legacy/      era-2/era-3 code + old examples (01_torus_deform, 03_visual_stress,
             g3_frame, 04_spectral_live) — NOT built by the live CMake
docs/        19 design docs (ARCHITECTURE, CAGE_REFRAME, SURFEL_RENDER, ...)
tests/       validate_* gates + perf_engine (auto-globbed by CMake)
```

**Key architectural fact:** geometry is *patches* (4 shared anchors + 8
coefficients + basis type), not meshes. Pipeline: articulation → anchor/body
solve → surface eval → surfel cache → GPU march → screen. The PSF/cage is the
authoritative body; surfels are a derived render cache.

---

## 3. Perf report interpretation (your A770 run)

```
eval     0.0003 ms/op   3.60 Mop/s   <- trivial, not a concern
closest  0.17 ms/op     0.01 Mop/s   <- brute-force K-nearest + Gauss-Newton
raycast  0.35 ms/op     0.00 Mop/s   <- same, 3D Newton
modal    0.23 ms/op                    <- Step+WriteBack, fine
AMR      189 ms/op                     <- THE bottleneck (SplitPatch: LSQ fit
                                        + restitch per split; 840 merges total)
refit    5.8 ms/op                     <- LSQ refit region
seams    8.9 ms/op                     <- Seams().Build full rebuild (hash maps)
flush    full 0.15 / partial 0.04 ms   <- GPU partial upload working (3.6x)
```

Read: **the render path is fast; the *editing* path is slow.** Everything that
re-fits equations on the CPU (split, refit, seam rebuild) is where the time
goes. That's the natural target for the already-written-but-unlaunched
`psf_refine.cl` / `psf_deform.cl` GPU kernels.

---

## 4. Things to work on (prioritized by evidence)

### P0 — make the slow path fast
1. **Launch `psf_refine.cl::split_patches` from `RefineWhere`** (GPU AMR).
   The kernel is implemented (atomic free-list, 5 hanging anchors, 9×9 LSQ
   child fit, stitch zeroing) but *no host code calls it*. 189 ms/op CPU
   splits are the single biggest number in the perf table. Host work: upload
   the region, launch, read back the new records, `RebuildViews()`, then a
   cheap CPU seam pass. Expected: order-of-magnitude on AMR.
2. **Incremental seams.** `Seams().Build` is a full O(n) rebuild at 8.9 ms/op
   and the playground calls it on every status line. `RestitchAround(ids)`
   already exists — route split/merge/dent through it instead of Build, and
   cache `MaxGap` per seam instead of recomputing.
3. **CPU spatial index for ClosestPoint/Raycast.** Both are brute-force over
   all patches (0.17/0.35 ms/op). `EquationAccel` already builds patch AABBs
   for the GPU ray-march — reuse that structure on the CPU (grid or BVH over
   patch AABBs) to seed the Newton solvers.

### P1 — wire the existing pieces together
4. **GPU modal writeback via `psf_deform.cl::apply_modes`.** `WriteBack`
   touches every anchor + every coefficient on the CPU each frame. The kernel
   exists; launch it when the body is on-GPU and only `FlushToGPU` the delta.
5. **Render `Transform`.** `PhysicsWorld::Step` integrates velocity/omega and
   the engine renders `o->Body()` *without applying the transform* — so
   fracture fragments and rigid bodies don't actually move on screen. Either
   apply the transform in `RenderBody` (uniform) or bake it into the body
   before upload. This is why "watch the fragment fly off" in the old stress
   test never really happens.
6. **`SplatRenderer::Resize` is still a stub** (`/* TODO realloc fb/img/tex */`)
   — needed for window resize; all the buffers are already capacity-tracked.
7. **Scene batching** — header says "Milestone 3 scope: one body per frame".
   The march path only marches the first body; composites use splat. A real
   scene (torus + fragments + NPCs) needs per-body march or a merged surfel
   cache.

### P2 — close the remaining stubs
8. `PatternLibrary::Fit` (CPU mirror of the .cl fitter) — needed for the
   offline tooling to match runtime.
9. `PSFBody::Volume` — analytic via divergence theorem (physics needs it for
   mass/volume without a mesh).
10. `Patch::PinEdgeToCurve` — intentional no-op; seam layer owns it, but the
    API should either do it or be removed.
11. 37 of 41 `TODO` comments are stale (code exists behind the comment) — a
    cleanup pass to keep the signal honest.
12. 4 unreferenced kernels (`hit_fill_stub`, `spf_splat`, `surfel_inverse`,
    `surfel_splat`) — wire or delete so the kernel list matches reality.

### P3 — content & AI
13. **Restore `examples/` from `legacy/examples/`** — the live CMake references
    `examples/01_torus_deform`, `examples/03_visual_stress`, `g3_frame` but the
    live `examples/` dir is empty (they only exist under `legacy/`), so those
    targets silently don't build. Copy them up (they're the pre-existing
    interactive examples) or update CMake paths.
14. **Neural reconstructor stage 2** — stage 1 (patch fitter) is trained and
    validated (MAE 16× better than LSQ baseline on 97 real asteroid windows).
    `ReconstructionModel::Infer` is still the deterministic fallback; loading
    the trained weights and swapping it in behind `IReconstructor` is the next
    milestone. Stage 2 = predict quadtree layout; stage 3 = reconstruct from
    photos.
15. **More assets** — only 4 `.psf` files exist. `tools/obj_to_psf.py` can
    convert any OBJ; a handful of varied topologies (concave, thin features,
    high-frequency) would stress-test AMR + seams far better than the torus.

### Housekeeping
- Your build printed `Clock skew detected` — that's NTP/local-clock drift on
  your machine, not the build; harmless here but can make incremental builds
  think files changed. `sudo timedatectl set-ntp true` if it recurs.
- The `m1_fix` SplatRenderer mismatch (2-arg `RenderBody` vs 3-arg header)
  from last session is resolved in this tree — the file here is the 1079-line
  version with `bool clear = true`.