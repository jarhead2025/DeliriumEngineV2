# Building M1 — DeliRium Engine Revision

## Prerequisites

### Linux Mint (Ubuntu/Debian base)

```bash
sudo apt update
sudo apt install cmake g++ make
```

### Intel Arc A770 (Linux)

Ensure the Intel GPU driver is installed and exposing OpenCL:

```bash
# Verify the GPU is recognized
lspci | grep -i vga

# Check that the Intel OpenCL ICD is present
ls /etc/OpenCL/vendors/

# If missing, install the Intel GPU tools / OpenCL package
# For Ubuntu 22.04+ / Mint 21+:
sudo apt install intel-opencl-icd intel-media-va-driver-non-free
```

The Arc A770 should appear as a generic OpenCL device — no vendor-specific code paths are used.

## Build

```bash
cd delirium-engine-revision-m1
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces:
- `libdelirium_m1_core.a` — the static library
- `delirium_m1_tests` — the test binary

## Run tests

```bash
./delirium_m1_tests
```

Expected output:

```
PASS: topology finalizes with valid anchors/materials
PASS: anchor 0 has one incident patch
PASS: shared anchor has both incident patches
PASS: runtime initializes from rest anchors
PASS: explicit anchor binding accepted    (×6)
PASS: normalized articulation rig binds to body
PASS: fixed articulation step publishes state    (×120)
PASS: generation increases only through completed publishes
PASS: moved anchor is dirty
PASS: unbound-to-moving anchor remains clean
PASS: shared anchor dirties both incident patches
PASS: spring articulation moves influenced anchor
PASS: fixed influence preserves unrelated anchor
PASS: non-normalized influence rejected
PASS: material-bearing patch finalizes
PASS: patch retains explicit material ID for future composition
M1 TESTS PASSED
```

## Build options

| CMake flag | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Build configuration (`Debug` for symbols) |
| `DELIRIUM_M1_WARNINGS_AS_ERRORS` | `OFF` | Treat `-Wall -Wextra -Wpedantic` warnings as errors |

## Cross-platform notes

- **macOS:** Replace `g++` with `clang++`; CMake handles the rest.
- **Windows (MSVC):** CMake generates Visual Studio projects automatically.
- **Other GPUs:** The M1 core library has no GPU dependency — it is pure CPU C++. OpenCL is not required until M2+ adds the surfel cache and marcher.