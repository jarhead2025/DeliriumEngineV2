# Agentic workflow progress (DeliRium)

## Done / stable
- M0 OpenCL probe (Intel Arc)
- M1 torus load: PSF v2 64-byte records, HasBody=yes
- SDL window + interactive Engine::Run
- OpenCL kernels/psf_splat.cl path
- E6 adaptive sample density from projected edges
- Camera WASD + mouse → XYZ drives MVP each frame

## Fill strategy (current direction)
- **Triangular infill**: connect sample dots with triangles; barycentric average color/depth
- Structured grid (per-patch su×sv) preferred
- Delaunay (Bowyer–Watson) optional for unstructured dots
- Host dilate / H-V span = legacy; not primary

## Protect
- PSFFile::Load, PSFBody::FromFile, Reconstructor, GameObject/Scene load path
- No duplicate *FromFile.cpp / SplatRenderer_E5.cpp under engine/src

## Packs
- m1_fix.tgz — Engine, SplatRenderer, TriInfill, tools, docs
- delirium_progress_pack.tgz — slim progress subset

## Agent instructions
- Prefer deterministic scaffolds over whole-registry LLM rewrite
- FORCE_GEN_CREW=0 when validating render quality
- DELIRIUM_TRI=1 for triangular path when wired
