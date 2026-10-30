# DeliRium working minimum (do not regress)

## Verified path

1. Load `assets/torus.psf` via `PSFFile::Load` + `PSFBody::FromFile` (64-byte records).
2. `delirium_m1_torus` + SDL window + OpenCL on Intel Arc.
3. `SplatRenderer` builds `kernels/psf_splat.cl`, samples equation patches, presents.

## Protect

- `engine/src/io/PSFFile.cpp`
- `engine/src/psf/PSFBody.cpp` (FromFile / FromRecords)
- `engine/src/psf/Reconstructor.cpp`
- `engine/src/core/GameObject.cpp` / `Scene.cpp`
- Do **not** leave `SplatRenderer_E5.cpp` or `PSFBody_FromFile.cpp` under `engine/src` (duplicate link).

## Knobs

| Env | Effect |
|-----|--------|
| `DELIRIUM_SS=24` | Sample grid side (default 24 in dense build; was 8) |
| `DELIRIUM_INFILL=0` | Disable average-overwrite infill |

## Next layers

- E6 denser grid + host average infill (this pack)
- OBJ→PSF foundation: `tools/obj_to_psf.py`
- Later: LS coef fit, texture→pattern K, multi-chart seams, surface combination ops
