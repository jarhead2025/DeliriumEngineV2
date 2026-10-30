# M0 scope and exit gate

## Why M0 exists

The prior repository validated several ideas, but it also accumulated parallel representations and experimental branches. M0 prevents the revision from repeating that drift. It captures baseline evidence and locks the boundaries that the next layer must obey.

M0 is deliberately small: portable configuration, OpenCL discovery, benchmark evidence, and an architectural handoff. No new renderer is considered successful merely because it produces an image.

## In scope

- Linux-first CMake build using generic OpenCL discovery.
- Capability reporting without Intel-, NVIDIA-, or AMD-specific code paths.
- A machine-readable renderer benchmark manifest.
- A validator for benchmark completeness and internal timing consistency.
- Contracts for the authoritative body, articulation, material regions, surfel cache, and ray-composition behavior.
- A concise record of which conclusions from the experimental engine are retained, deferred, or rejected.

## Out of scope

- Porting the old renderer into this archive.
- Replacing PSF with another geometry format.
- Implementing a physics solver, IK, spring bones, editor, Blender importer, or full scene graph.
- Claiming cross-device performance equivalence.
- Treating a profile run that introduces synchronization as a throughput measurement.

## M0 exit gate

M0 is complete when all conditions hold:

1. `cmake -S . -B build`, `cmake --build build`, and `ctest` work on a supported Linux/OpenCL setup.
2. `delirium_m0_probe` lists at least one OpenCL device and reports relevant optional capabilities.
3. A benchmark manifest exists for at least one known asset and passes `tests/validate_manifest.py`.
4. The manifest identifies exact asset bytes, surfel count, resolution, renderer settings, device/driver, command, and frame-time summary.
5. The M1 team can implement body state and articulation without deciding again whether `.srf`, PSF/cage, or raw mesh is authoritative.

## Performance interpretation

A benchmark is evidence for one asset, camera path, quality configuration, driver, and GPU. It must not be generalized to whole-scene performance. The M0 baseline permits later milestones to make fair comparisons when they add incremental cache updates, multi-surfel cell lists, articulation, and physical state.
