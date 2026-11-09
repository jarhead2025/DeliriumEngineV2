# Vulkan backend — feasibility assessment

**Date: 2026-11-08.** Assessed *before* the G1–G19 goals are executed, per the
agreed ordering. This document records the reasoning so the decision can be
revisited when the goals land, without re-deriving it.

---

## The short answer

**Do not rewrite the backend in Vulkan now.** The engine's GPU surface is
**small, narrow, and already correct**; a Vulkan port would consume roughly
the same effort as the entire bones/articulations spine (G7–G12), which is
the goal that actually changes what the engine *is*.

But there is a **real, narrower option**: replace the OpenCL **host
plumbing** with Vulkan compute while keeping the kernels' mathematical
contract — a *transport* port, not a *renderer* rewrite. That is worth doing
**only if** a specific driver or platform constraint appears.

---

## What the GPU surface actually is

Measured on the current tree (`gameengine_compiled_real`):

| | |
|---|---|
| Kernels | **13** `.cl` files, **3,158 lines** total |
| Host renderer | `SplatRenderer.cpp`, **2,596 lines** |
| Distinct OpenCL features used | `__global` (210×), `atomic_min` (12×), `atomic_add` (8×), `cl_khr_int64_base_atomics` (5×) |
| Image/stencil usage | **none** — no `read_image`, no `write_image`, no `image2d_t` |
| Shared/local memory | **none** — no `__local`, no `barrier` |
| Host API calls | dominated by `clSetKernelArg` (268×) and `clEnqueueNDRangeKernel` (29×) |

**Vulkan present anywhere in the tree: none.** The sibling tree named
`delirium-vulkan-engine` is the **OpenCL** lineage — its `BUILD_GPU.md`
documents `ocl-icd-opencl-dev` and the NVIDIA OpenCL ICD. The name is
historical, not a Vulkan implementation.

### The decisive observation

The kernels use **no images, no shared memory, no barriers, no subgroups, no
floating-point atomics**. They are:

```
buffer A ──▶ compute shader ──▶ buffer B ──▶ (repeat) ──▶ resolve ──▶ upload
```

with `atomic_min` / `atomic_add` on `uint`/`int` buffers as the only
synchronisation primitive. That is a **pure compute pipeline over plain
storage buffers** — which is the *easiest* class of workload to express in
Vulkan, because it maps almost 1:1 onto a storage buffer + a compute pass.

So the port is not technically hard. The question is whether it is *worth*
the cost, and what that cost actually buys.

---

## The four possibilities

### Option A — Full Vulkan renderer rewrite (all 13 kernels, new host)
**Effort:** very high. 13 kernels → SPIR-V compute shaders, a new host with
`VkDevice`/`VkQueue`/`VkCommandPool`/descriptor sets/pipeline layout,
resource barriers replacing OpenCL's implicit sync, and a new presentation
path (swapchain vs. the current `SDL_RenderReadPixels` blit).

**What it buys:**
- **Explicit sync + in-command-buffer execution.** OpenCL's `clFinish` and
  `clEnqueueReadBuffer` are the current bottleneck (the readback to SDL was
  already noted as the limiting cost in the sibling tree's `BUILD_GPU.md`:
  *"el límite será el readback CL->SDL"*). Vulkan's staging-buffer +
  fence model makes the readback asynchronous and overlap-able.
- **Subgroup ops** — the only path to real gains for the surfel-march
  neighbourhood search, which is the hottest kernel.
- **Portability**: a Vulkan backend reaches Windows/Linux/Android without
  depending on an OpenCL ICD being installed. **This is the strongest
  argument** — OpenCL is effectively deprecated on many platforms, and this
  box already needed an explicit `ocl-icd` setup.
- **Compute+graphics interop** in one queue, enabling screen-space passes
  (SSAO, bloom) over the same depth buffer the renderer already produces.

**What it costs:**
- Every one of the 13 kernels must be **re-validated against the CPU path**.
  The current kernels have a CPU mirror in `Patch.cpp` (`dequant_pos`,
  `deform_h`, `stitch_h` all mirror the `.cl` exactly). Keeping two
  implementations of 13 kernels bit-compatible is the real risk, and
  **ARCHITECTURE.md invariant #2 (CPU/GPU bit-compat) is currently
  ungated** (goal G17).
- 268 `clSetKernelArg` calls become descriptor-set binding. **This is where
  the real time goes** and it is mechanical, not conceptual.
- The "silent fallback" failure mode gets worse. A Vulkan shader that fails
  to compile must not silently disable the march path — the exact trap that
  bit the OpenCL path twice this session.

### Option B — Vulkan compute for the hot kernels only (hybrid)
Keep OpenCL for the cold kernels; run the march/raymarch/splat kernels
through Vulkan compute.

**Verdict: no.** Two compute APIs on one device means two resource models,
two synchronization domains, and a cross-API fence. The savings are small
because the cold kernels are not where the time is.

### Option C — Keep OpenCL, remove the readback bottleneck
Replace the `clEnqueueReadBuffer` → `SDL_RenderReadPixels` blit with
`clCreateImage` + `GL`/`SDL` interop (or a second OpenCL context sharing
the device, writing into a GL texture).

**Effort:** low-to-moderate. **This captures most of Option A's throughput
benefit at a fraction of the cost**, because it targets the *measured*
bottleneck rather than rewriting the whole surface.

**Note the constraint:** SDL2's OpenGL backend cannot share with an OpenCL
buffer directly; this needs either a Vulkan-SDL path, a GL-CL interop
extension, or a plain `glTexImage2D` upload (which is what happens now).

### Option D — Defer entirely, and revisit at a trigger
**This is the recommended option for now.** Define the trigger conditions in
advance so the decision is not re-litigated every few months.

---

## Recommended decision

**Option D now; Option C as a targeted follow-up if a measured bottleneck
appears; Option A only when a specific trigger fires.**

### Triggers for Option A (any one is sufficient)
1. **Platform reach becomes a requirement** — a target where no OpenCL ICD
   exists or is installable. This is the only trigger that makes the port
   *necessary* rather than merely *better*.
2. **The bones/articulations work (G7–G12) lands and the render cost becomes
   the limiting factor** — because articulated bodies need per-bone
   transforms on the GPU, and the current path pushes a full model matrix
   per body. A scene of articulated bodies may need a different data layout
   (bone-palette storage buffers + per-draw indexed transforms), which is
   *easier to design directly in Vulkan* than to retrofit onto OpenCL.
3. **Subgroup ops are needed for a specific kernel** and the OpenCL path
   cannot express them.

### Triggers for Option C (cheap, do first if any of these appear)
- Measured frame time dominated by GPU→CPU→SDL readback rather than by
  kernel execution.

---

## How this interacts with the goal order

This is the important part: **the Vulkan question is downstream of the
bones work, not parallel to it.**

```
G7 bone vocabulary ──▶ G8 bake skeleton ──▶ G10 coarseness ──▶ G9 inertia
                                                                │
                                                                ▼
                                                   (articulated bodies need
                                                    per-bone GPU transforms)
                                                                │
                                                                ▼
                                              Option A becomes natural, or not
```

Reasoning:

- **Do it after G8.** Once `bone_index` is real on the anchors, the GPU
  needs a *bone palette* (a storage buffer of per-bone transforms, indexed
  by `bone_index`) rather than one model matrix per body. That is a data
  layout change, and **data layout changes are far cheaper to make in a
  backend you are already rewriting** than to retrofit afterwards. Getting
  the layout right *first* (G8) and the API second is the correct order.
- **Do it after G10.** Coarseness decides LOD; Vulkan's real advantage
  (explicit control over work granularity) only pays off if the LOD scheme
  is settled.
- **Do not do it before G6.** A backend rewrite on an uncommitted tree is
  how work gets lost.

### The dependency that argues for waiting

**Option A requires the CPU/GPU bit-compat validator to exist first
(goal G17).** A Vulkan rewrite means 13 kernels reimplemented in SPIR-V
against a CPU mirror that lives in `Patch.cpp`. Without a gate that proves
CPU and GPU agree, that reimplementation is *unverifiable*. The OpenCL path
at least has `psf_validate.py` in the sibling tree (not here) and the
`validate_eval` gate that catches normal-sign flips.

**So: G17 (the CPU/GPU bit-compat validator) is a hard prerequisite for
Option A.** It should move earlier in the order if Option A is chosen —
that is a legitimate order-change under the agreed rule, and it would be
recorded in `docs/GOALS.md`.

---

## Benefits summary (honest version)

| | OpenCL now | Vulkan compute | Full Vulkan |
|---|---|---|---|
| Portability (no ICD needed) | ✗ | ✓ | ✓ |
| Async readback / no `clFinish` stalls | ✗ | ✓ | ✓ |
| Subgroup ops | ✗ (vendor ext) | ✓ | ✓ |
| Compute+graphics interop | ✗ | ✗ | ✓ |
| Rewrite cost | — | high | very high |
| Bit-compat risk | baseline | must re-prove 13 kernels | must re-prove 13 kernels |
| Time to first working frame | — | weeks | weeks |

**The single largest benefit is portability**, and it is a *deployment*
benefit, not a *rendering-quality* one. Vulkan will not make the engine
look better or run the bones/inertia work faster on its own — it changes
where it can run, and removes the readback stall.

---

## What would make the port *wrong* to do now

1. The bones spine (G7–G12) is the goal that changes what the engine
   *can represent*. A backend port does not.
2. The renderer is **not currently the bottleneck** for the work in flight:
   the doc-learner round trip takes 20–280 s, and the render is 43 ms.
3. OpenCL's feature usage here is minimal and maps cleanly to Vulkan compute
   — which means the port has **no technical forcing function**. It is a
   choice, not a necessity. A choice is worth making only when something
   else forces it.
