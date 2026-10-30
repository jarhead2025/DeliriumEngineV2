# M2 SplatRenderer — hit pixel + pattern fill (agent authority)

Read this before editing `engine/src/render/SplatRenderer.cpp` or related kernels.
In-pipeline: `search_agent_docs("splat hit fill")` / `read_agent_doc("cheatsheets/m2_splat_hit_fill")`.

## Design (non-negotiable)

1. Geometry is **equation surfaces**, not triangle meshes.
   - Surface value ≈ Σᵢ (coeffᵢ × basisᵢ(u,v)) with basis ∈ {sphere, plane, corrugation, …}
   - Anchors (XYZ, ≥4 per patch) pin pose / orientation / seams.
2. Appearance is a **parallel series**: pattern coeffs × RGB (`PatternLibrary`), not UV mesh textures.
3. Rendering is **sparse impact → fill**, not full triangle rasterization.

## Algorithm (host or OpenCL)

### Pass A — intersect → one pixel per hit

```
for each sample ray (or grid direction):
  hit = intersect_psf(ray, PSFBody)   // analytic/numeric on coeffs + anchors
  if hit:
    write HitPixel {
      screen_xy or buffer index,
      t, normal,
      patch_id / surface_id,
      surface_params (u,v or equivalent),
      material / pattern_id
    }
```

One surface impact → **one** hit pixel (or surfel record). No triangle soup.

### Pass B — fill by averaging patterns through pixels

```
for each output pixel (or each hit neighborhood):
  neighbors = nearby hits in screen or patch space (k-NN or small window)
  color = average over neighbors of:
            evaluate_pattern_series(pattern_id, surface_params, RGB coeffs)
  write_framebuffer(color)
```

“Average pattern through the pixels” = blend pattern evaluations from sparse hits so the image densifies without meshing.

## Header discipline

- Implement **ONLY** methods declared in `engine/include/delirium/render/SplatRenderer.h`
- Typical names may include `RenderBody` / similar — **NEVER invent**:
  `Initialize`, `BeginFrame`, `EndFrame`, `renderSplats`, `reconstructMesh`
- If unsure: `read_project_file` the header, then write `// FILEPATH:` blocks that match it.

## Intel OpenCL notes

- Use C OpenCL headers (`CL/cl.h`), not `cl.hpp`
- Prefer one kernel for intersect + hit buffer, one for pattern fill
- fp64 may be unavailable on some Intel devices — stay `float` unless probed

## Definition of done for M2

- [ ] `SplatRenderer.cpp` matches header API (compiles)
- [ ] At least one path: synthetic ray/grid → hit buffer → non-black fill from pattern average
- [ ] No triangle mesh generation in this path
- [ ] M0 probe + M1 torus still link

## Not M2 (later milestones)

| Milestone | Focus |
|-----------|--------|
| **M3** | OpenCL intersect + fill kernels on Intel; measurable vs host path |
| **M4** | Multi-patch seams + anchor-stable fill; PatternLibrary wired for real series |
| **M5** | Live coeff / modal edits visible in the fill without remeshing |
| **M6** | Bones → anchors; simple animated body |
| **M7+** | LOD, temporal reuse, editor, multi-body scenes, production quality |

## Roadmap distance (honest)

You already have:

- **M0** — OpenCL device discovery (done)
- **M1** — linkable engine + torus vertical slice (done)
- **M2** — this guide: header-faithful SplatRenderer + hit→fill *vertical slice*

From a **full** Delirium (equation surfaces, pattern color, bones, modal physics, editor):

- **~1 milestone (M2)** to a **demoable** still image: sparse hits + pattern average on a simple body  
- **~2–3 more (M3–M4)** to **GPU path + multi-patch** that looks intentional  
- **~4–5 more (M5–M6)** to **live deformation + animation** on equations  
- **M7+** for production polish (LOD, temporal, tooling)

So: **not “dozens” of M_ steps** if each milestone stays a thin vertical slice; roughly **5–7 milestones total from M0** to a coherent playable prototype, of which **2 are already behind you**.
