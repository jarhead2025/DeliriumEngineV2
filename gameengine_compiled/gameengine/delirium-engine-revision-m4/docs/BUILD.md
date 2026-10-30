# Build Instructions

## Prerequisites

- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.20+
- Make or Ninja

## Build

```bash
cd delirium-engine-revision-m4

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run tests
./build/delirium_m4_tests
```

## Expected Output

All 15 test groups should pass:

```
PASS: stiff material youngs modulus is correct
PASS: soft material youngs modulus is correct
PASS: shear modulus is positive
PASS: bulk modulus is positive
PASS: unknown material returns default properties
PASS: single-patch topology finalizes
PASS: runtime initializes
PASS: damage accumulation is monotonic across steps
PASS: damage is within valid range [0, 1]
PASS: topology finalizes for modal test
PASS: runtime initializes for modal test
PASS: step with modal system succeeds
PASS: modal displacement is non-zero for at least one anchor
PASS: anchor position or modal displacement reflects modal contribution
PASS: topology finalizes for strain test
PASS: runtime initializes for strain test
PASS: step with strain succeeds
PASS: all patch strain energies are non-negative
PASS: topology finalizes for color test
PASS: runtime initializes for color test
PASS: cache rebuild succeeds for color test
PASS: cache has surfels after rebuild
PASS: surfels have material-driven red color
PASS: surfels have material-driven PBR properties
... (remaining M3-carryover tests)
M4 TESTS PASSED
```

## Warning Flags

Build with warnings as errors:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DELIRIUM_M4_WARNINGS_AS_ERRORS=ON
```