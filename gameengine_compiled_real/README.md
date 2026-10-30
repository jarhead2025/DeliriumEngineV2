# DeliRium Engine Revision — M0

M0 is a clean, portable baseline for the interactive-engine revision. It does **not** attempt to reimplement the experimental renderer or physics stack. It records the decisions earned by the earlier work and makes the next layer buildable from measured evidence.

## M0 outcome

A developer can:

1. configure the project with standard CMake;
2. discover and report a usable OpenCL device without vendor-specific code;
3. record a reproducible benchmark manifest for an external renderer run;
4. validate that the manifest has the data needed to compare later milestones;
5. use the body, rendering, and articulation contracts as the stable boundary for M1.

The canonical future model is:

```text
articulation target → anchor/body solve → surface evaluation
                    → dirty surfel cache → progressive GPU march → screen
```

PSF/cage data is the authoritative body. Surfels are a derived render cache, never the sole editable or physical representation.

## Supported baseline

- Linux-first, including Linux Mint.
- C++17 and CMake 3.20 or newer.
- OpenCL 1.2 baseline; later capabilities are detected rather than assumed.
- Any conformant OpenCL implementation. Intel Arc A770 is a target test device, not a required vendor path.

## Build

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/delirium_m0_probe
```

The probe lists platforms/devices and reports the capabilities that matter to later milestones. It succeeds if at least one OpenCL device is found.

## Record an external renderer benchmark

M0 intentionally does not bundle the previous experimental renderer. Record a benchmark from any compatible renderer with:

```bash
python3 scripts/capture_benchmark.py \
  --out artifacts/spot-a770.json \
  --renderer march_interactive \
  --asset /absolute/path/to/spot.srf \
  --command './march_interactive /absolute/path/to/spot.srf --bench 300' \
  --resolution 700x700 \
  --frames 300 \
  --device 'Intel Arc A770' \
  --driver '<driver version>' \
  --layers 24 --band-steps 24 --fill-passes 4

python3 tests/validate_manifest.py artifacts/spot-a770.json
```

Run profiling separately, because a profiling implementation may synchronize GPU work and distort throughput:

```bash
./march_interactive /absolute/path/to/spot.srf --bench 300 --profile
```

Attach or reference the profile output in the manifest notes rather than treating it as the headline FPS result.

## Layout

- `docs/M0_SCOPE.md` — exact M0 exit criteria and non-goals.
- `docs/EXPERIMENT_DECISIONS.md` — conclusions retained from the previous engine.
- `docs/M1_HANDOFF_CONTRACT.md` — interfaces and invariants M1 must preserve.
- `config/benchmark.schema.json` — benchmark evidence contract.
- `src/opencl_probe.cpp` — generic OpenCL capability probe.
- `scripts/capture_benchmark.py` — converts renderer console output into a structured manifest.
- `tests/validate_manifest.py` — validates captured evidence.

## M0 non-goals

M0 does not add a new geometry format, renderer, physics solver, skeleton runtime, or editor. Those belong to later layers after the baseline is reproducible.
