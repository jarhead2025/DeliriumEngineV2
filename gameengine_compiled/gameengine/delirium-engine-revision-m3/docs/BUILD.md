# Building M3 — DeliRium Engine Revision

## Prerequisites

### Linux Mint (Ubuntu/Debian base)

```bash
sudo apt update
sudo apt install cmake g++ make
```

**Compiler requirement:** GCC 11+ or Clang 13+ for full C++17 support.

## Build

```bash
cd delirium-engine-revision-m3
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces:
- `libdelirium_m3_core.a` — the static library
- `delirium_m3_tests` — the test binary

## Run tests

```bash
./delirium_m3_tests
```

Expected output: all tests PASS, ending with `M3 TESTS PASSED`.

## Build options

| CMake flag | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Build configuration (`Debug` for symbols) |
| `DELIRIUM_M3_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings as errors |

## Cross-platform notes

- **macOS:** Replace `g++` with `clang++`; CMake handles the rest.
- **Windows (MSVC):** CMake generates Visual Studio projects automatically.
- M3 has no GPU dependency — pure CPU C++.