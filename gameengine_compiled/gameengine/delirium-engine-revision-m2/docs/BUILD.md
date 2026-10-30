# Building M2 — DeliRium Engine Revision

## Prerequisites

### Linux Mint (Ubuntu/Debian base)

```bash
sudo apt update
sudo apt install cmake g++ make
```

**Compiler requirement:** GCC 11+ or Clang 13+ (needed for `std::numbers::pi_v<float>`).

### Intel Arc A770 (Linux) — optional OpenCL

M2 compiles and tests **without OpenCL**. Enable OpenCL only when you want the upload adapter to link against the real SDK:

```bash
# Verify the GPU is recognized
lspci | grep -i vga

# Check that the Intel OpenCL ICD is present
ls /etc/OpenCL/vendors/

# If missing:
sudo apt install intel-opencl-icd intel-media-va-driver-non-free
```

## Build (no OpenCL — default)

```bash
cd delirium-engine-revision-m2
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces:
- `libdelirium_m2_core.a` — the static library
- `delirium_m2_tests` — the test binary

## Build (with OpenCL)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DDELIRIUM_M2_ENABLE_OPENCL=ON
make -j$(nproc)
```

## Run tests

```bash
./delirium_m2_tests
```

Expected output: all tests PASS, ending with `M2 TESTS PASSED`.

## Build options

| CMake flag | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Build configuration (`Debug` for symbols) |
| `DELIRIUM_M2_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings as errors |
| `DELIRIUM_M2_ENABLE_OPENCL` | `OFF` | Link OpenCL SDK into upload adapter |

## Cross-platform notes

- **macOS:** Replace `g++` with `clang++`; CMake handles the rest.
- **Windows (MSVC):** CMake generates Visual Studio projects automatically.
- **Other GPUs:** The M2 core library has no GPU dependency — pure CPU C++. OpenCL is optional.