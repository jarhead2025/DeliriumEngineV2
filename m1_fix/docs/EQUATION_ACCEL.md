# EquationAccel — conical frustum + BVH + adaptive UVN

Sticks to PSF math (64B anchors/surfaces, coef[8] deform). Acceleration only
decides *which* patches get equation probes and *how dense* those probes are.

## Pipeline
1. **Build** patch AABBs from anchors + deform margin; binary BVH
2. **FrustumFromMVP** — 6 planes from view-projection (conical view volume)
3. **QueryFrustum** — BVH walk, AABB∩frustum, projected size → adaptive uvn
4. **GPU** `psf_raymarch` loops only `visible[]` with per-entry uvn

## Install
```bash
cp m1_fix/engine/include/delirium/render/EquationAccel.h engine/include/delirium/render/
cp m1_fix/engine/src/render/EquationAccel.cpp engine/src/render/
cp m1_fix/kernels/psf_raymarch.cl kernels/
cp m1_fix/engine/src/render/SplatRenderer.cpp engine/src/render/
# CMake GLOB should pick EquationAccel.cpp
```

## Run
```bash
export DELIRIUM_RAYMARCH=1
export DELIRIUM_RAY_UV_MAX=8   # adaptive upper bound
export DELIRIUM_GLOW=0.3
./build_delirium/delirium_m1_torus
```

Log:
```text
EquationAccel: patches=544 bvh_nodes=...
GPU raymarch: 800x600 vis=400/544 uvn=[2,8] avg=4.2 glow=0.30
```

`vis < total` means frustum culling is active. Adaptive uvn spends detail where
the equation patch is large on screen.
