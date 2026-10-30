# Lateral / view-ray equation march

## Idea
Use PSF raw math (bilinear anchors + deform_h coef[8]) as the surface.
Primary rays from the camera; probe equation domain; light + shadow; glow;
**only final AA** (optional GPU box filter) — no splat dilate as primary fill.

## Enable
Default ON in patched SplatRenderer.
```bash
# explicit
export DELIRIUM_RAYMARCH=1
export DELIRIUM_GLOW=0.6          # emissive strength (torus glow + color shift)
export DELIRIUM_RAY_STRIDE=2      # 2 = faster preview (blocky)
./build_delirium/delirium_m1_torus

# classic GPU splat instead
export DELIRIUM_RAYMARCH=0
```

## Lights
- Headlight (view-aligned)
- Key directional
- Ambient
- Per-patch emissive animated by time + coef energy

## CMake
Add `engine/src/render/LateralRayMarch.cpp` to the target (GLOB usually picks it up).
