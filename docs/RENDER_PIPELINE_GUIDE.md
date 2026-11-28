# DeliRium Engine v2.2 — Render Pipeline Intro Guide

A physics-first C++17 engine for the Intel Arc A770 (OpenCL). This guide explains the
render pipeline that delivers **60+ FPS** on parametric surfaces, and the optimizations
that made it possible.

---

## 1. The Pipeline at a Glance

DeliRium renders **parametric PSF v4 surfaces** (objects defined by equations, not meshes).
The primary renderer is the **TriInfill GPU path**:

```
REST state (modal) ──bind──► GPU buffers
                               │
    per frame: tiny K-vector upload (modal coefficients only)
                               │
    psf_sample.cl  ─► evaluate PSF math → anchors + normals + colors
    tri_build.cl   ─► one work-item per triangle → index buffer (atomic_inc)
    tri_raster.cl  ─► structured-tri GPU raster → framebuffer
                               │
    resolve ─► present to window
```

Key architectural choice: **physics and render are independent computations from the same
REST state.** Physics needs the host blob (collision/FEM); render does NOT. REST uploads
**once at bind**; each frame only uploads the tiny K vector (modal coefficients). Kernels
write deformed geometry directly into shared GPU render buffers — no per-frame host upload.

---

## 2. The Performance Gains (measured on Intel Arc A770)

### Stage-by-stage speedups

| Stage / Metric | Before | After | Speedup |
|----------------|--------|-------|---------|
| `psf_sample` kernel (`sample_ms`) | 15.4 ms | ~0.045 ms | **~340×** |
| Total render time | 16.4 ms | ~0.35 ms | **~47×** |
| GPU raster (`gpu_raster_ms`, adaptive UVN) | 9.4 ms | 0.36 ms | **26×** |
| Modal writeback/setup stage | ~28 ms | ~0.13 ms | **~215×** |

### Steady-state result

Single torus, adaptive LOD on (`DELIRIUM_TRI_LOD=1`, `DELIRIUM_TRI_LOD_RANGE="4 200"`,
`DELIRIUM_TRI_UVN=256`):

- **62 FPS** sustained
- UVN up to 200 near camera, ~1.1M triangles
- GPU work: ~1.6 ms per frame (display-refresh-limited)

### FPS stall eliminated

Root cause of the ~27–31 ms/frame stall: an unconditional `FlushToGPU` loop in the surface
upload block. For modal bodies, `body.AnyDirty()` was true every frame → hundreds of
`clEnqueueWriteBuffer` calls serializing on the Intel Arc in-order queue → ~32 ms stall.

Fix: gate the host upload behind modal-body detection. When GPU modal writeback is active,
the host blob is never uploaded per frame. Result: frame time dropped from ~31 ms to ~3–4 ms.

**Lesson registered:** when GPU enqueue is fast but frame time is high, measure GPU execution
time separately (finish-diag) BEFORE assuming kernel/sync optimization — the bottleneck may
be elsewhere entirely (here: the CPU host-upload path).

---

## 3. Optimizations in v2.2

### GPU modal writeback (`DELIRIUM_GPU_MODAL=1`)
REST uploaded once at bind; per-frame only the K vector uploads. Kernels write deformed
geometry directly into shared render buffers (`buf_anchors`/`buf_surfaces`/`buf_snorm`).
No host blob upload → no driver stall.

### Phase 2 GPU sampling kernels (`psf_splat.cl`)
Moved PSF math evaluation onto the GPU. Sample time dropped from 15.4 ms to ~0.045 ms.

### Adaptive UVN / LOD (`DELIRIUM_TRI_LOD=1`)
Per-patch UVN derived from projected screen-space edge size. Near patches get high UVN
(detailed), far patches get low UVN (cheap). Raster time dropped 26× at extreme UVN vs uniform.

### Frustum culling (`DELIRIUM_FRUSTUM_CULL=1`, default ON)
Bodies outside the view frustum are skipped entirely.

### Distance-based LOD (`DELIRIUM_TRI_DIST_LOD=1`, default OFF; range `DELIRIUM_TRI_DIST_LOD_RANGE`)
Scales the base UVN by distance to camera. Combined with frustum culling, GPU work dropped
from ~0.45 ms to ~0.02–0.04 ms; FPS went from ~7.7 to 60+ on multi-body scenes.

### Debug gating (`DELIRIUM_GPU_MODAL_DIAG=1`)
All `[gpu-modal-diag]` diagnostic prints are behind a flag — release builds stay quiet.

---

## 4. Tuning the March Path (recon_window)

The recon_window scene uses the march/raymarch fill path for generated objects. These env
vars tune it without recompiling:

```bash
DELIRIUM_TRI=1 DELIRIUM_MARCH=1 \
DELIRIUM_MARCH_STEPS=20 \   # raymarch steps (default 40)
DELIRIUM_MARCH_PASSES=2 \   # full-screen hole-fill passes (default 4)
DELIRIUM_MARCH_UVN=8 \      # surfel bake density (default 10)
DELIRIUM_MARCH_DIAG=1 \
./build_delirium/delirium_recon_window
```

Result: **50–62 FPS** in recon_window (per-body time roughly halved).

---

## 5. Benefits of This Architecture

1. **Physics-first correctness.** Objects are defined by equations; render and physics derive
   from the same authoritative REST state. Graphical splats/surfaces are *views*, not the
   source of physical truth. No silent mutation of mass or physical state by the renderer.

2. **GPU-bound, not CPU-bound.** The bottleneck moved from the CPU host-upload path (~27 ms
   stall) to the GPU (~1.6 ms work), which is display-refresh-limited. This scales with scene
   complexity rather than collapsing under it.

3. **Continuous LOD.** Adaptive UVN means detail where it matters (near camera) and speed
   where it doesn't (far away) — automatically, per patch, every frame.

4. **No per-frame host round-trips.** REST uploads once; modal writeback happens on the GPU.
   This is the single biggest win — it eliminated the driver stall that capped FPS at ~30.

5. **Extensible.** The pipeline is structured as explicit modules (sample → build → raster →
   resolve). New kernels, new surface types, new fill strategies plug in without rewriting
   the whole path.

6. **Observable.** `DELIRIUM_RENDER_STAGES=1` prints per-stage timing; the workspace editor's
   Profiler panel shows sample/build/raster/upload ms + tris/samples/UVN/LOD mode live.

---

## 6. Quick Reference — Env Vars

| Variable | Default | Effect |
|----------|---------|--------|
| `DELIRIUM_TRI=1` | off | Enable TriInfill GPU path |
| `DELIRIUM_GPU_MODAL=1` | off | GPU modal writeback (no per-frame host upload) |
| `DELIRIUM_TRI_LOD=1` | off | Adaptive per-patch UVN |
| `DELIRIUM_TRI_LOD_RANGE="min max"` | `"2 200"` | Clamp adaptive UVN |
| `DELIRIUM_FRUSTUM_CULL=1` | on | Skip bodies outside view frustum |
| `DELIRIUM_TRI_DIST_LOD=1` | off | Distance-based UVN scaling |
| `DELIRIUM_TRI_DIST_LOD_RANGE="min max"` | `"4 30"` | LOD distance bounds |
| `DELIRIUM_MARCH_STEPS=N` | 40 | Raymarch step count |
| `DELIRIUM_MARCH_PASSES=N` | 4 | Hole-fill passes |
| `DELIRIUM_MARCH_UVN=N` | 10 | Surfel bake density |
| `DELIRIUM_GPU_MODAL_DIAG=1` | off | Modal writeback diagnostics |
| `DELIRIUM_RENDER_STAGES=1` | off | Per-stage timing prints |

---

*Generated for DeliRium Engine v2.2. Performance figures measured on Intel Arc A770
(driver 26.40.40196.15). Actual FPS depends on scene complexity, resolution, and tuning.*
