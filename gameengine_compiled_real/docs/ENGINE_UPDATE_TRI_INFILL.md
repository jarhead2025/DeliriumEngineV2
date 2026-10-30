# Update engine: triangular infill + Delaunay + camera

## What this adds

1. **Structured triangular infill** – for each patch sample grid `(su,sv)`, two triangles per cell; **barycentric** RGB + depth.
2. **Delaunay (Bowyer–Watson)** – unstructured dots in screen space, then same barycentric fill (capped ~2k points for speed).
3. **Camera** – WASD + mouse still drive `Camera` XYZ each frame so grids and projection track the view.

## Files to copy

```bash
cd ~/Desktop/Tartarus/gameengine_compiled_real

cp ../m1_fix/engine/include/delirium/render/TriInfill.h \
   engine/include/delirium/render/TriInfill.h
cp ../m1_fix/engine/src/render/TriInfill.cpp \
   engine/src/render/TriInfill.cpp

# camera controls
cp ../m1_fix/engine/src/core/Engine.cpp engine/src/core/Engine.cpp

# optional: reference SplatRenderer (adaptive 1.1 sample/px)
# cp attachments/SplatRenderer.cpp engine/src/render/SplatRenderer.cpp

rm -f engine/src/psf/PSFBody_FromFile.cpp engine/src/render/SplatRenderer_E5.cpp
```

## CMake

Ensure `TriInfill.cpp` is on the `delirium_m1_torus` target. If you use `GLOB_RECURSE engine/src/*.cpp`, a reconfigure is enough:

```bash
cmake -S . -B build_delirium -DCMAKE_BUILD_TYPE=Release
cmake --build build_delirium -j"$(nproc)" --target delirium_m1_torus
```

If GLOB does not pick it up, add explicitly:

```cmake
target_sources(delirium_m1_torus PRIVATE engine/src/render/TriInfill.cpp)
```

## Wire into the frame (minimal)

In `SplatRenderer::RenderBody` (host path) or after GPU resolve, build `SampleDot`s from projected equation samples, then:

```cpp
#include "delirium/render/TriInfill.h"
// ...
std::vector<uint32_t> rgba(size_t(width)*height, 0xFF121218u);
std::vector<float> zbuf(size_t(width)*height, 1e30f);
// Structured (preferred when you know su,sv per patch):
TriInfillStructured(dots, su, sv, width, height, rgba, zbuf);
// Or unstructured:
// TriInfillDelaunay(dots, width, height, rgba, zbuf);
// upload rgba to buf_img / SDL texture for Present()
```

Env switches (suggested):

| Env | Meaning |
|-----|---------|
| `DELIRIUM_TRI=1` | enable triangular infill path |
| `DELIRIUM_TRI=delaunay` | use Delaunay instead of structured |
| `DELIRIUM_LATERAL=0` | keep pure CL splat if both exist |

## Run

```bash
cd ~/Desktop/Tartarus/gameengine_compiled_real
./build_delirium/delirium_m1_torus
# WASD move, mouse look, Esc quit
```

## Coordinate interpolation

For a pixel inside triangle ABC:

```text
w0, w1, w2 = barycentric weights (sum 1, all ≥ 0)
color = w0*A + w1*B + w2*C
depth  = w0*Az + w1*Bz + w2*Cz
```

That is the “average the dots” over the triangular section.

## Performance notes

- **Structured** scales with samples; use E6-style density (denser near camera).
- **Delaunay** is O(n²)-ish worst case; subsample if n > ~2500.
- Prefer structured grids from PSF patches; Delaunay for residual / sparse sets.
