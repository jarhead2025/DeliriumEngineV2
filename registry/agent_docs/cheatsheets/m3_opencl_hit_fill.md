---
agent: pipeline
topic: milestone
updated: 2026-09-15T14:20:23+00:00
---

# M3 — OpenCL intersect + pattern fill

Goal: run hit-buffer and/or fill on Intel OpenCL; keep host fallback.
- Headers: CL/cl.h only (no cl.hpp).
- Prefer kernels/*.cl + thin host enqueue in SplatRenderer.
- Probe device once (reuse M0 patterns); float not double if no fp64.
- Do not break delirium_m0_probe or delirium_m1_torus.
- Still: one pixel per surface hit, then average patterns — no tris.
