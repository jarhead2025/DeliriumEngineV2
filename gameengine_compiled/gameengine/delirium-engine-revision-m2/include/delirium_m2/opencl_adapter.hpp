#pragma once

#include "delirium_m2/surfel_cache.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// OpenCLUploadAdapter — optional OpenCL-facing upload layer
// ---------------------------------------------------------------------------
//
// This adapter is OPTIONAL. The M2 core compiles and tests without OpenCL.
// When OpenCL is available, it provides a path to upload the surfel cache
// to GPU buffers for the progressive marcher.
//
// The adapter enforces the generation invariant: it refuses to upload a cache
// whose source generation does not match the expected BodyState generation.
// ---------------------------------------------------------------------------

// Opaque handle to an OpenCL buffer (platform-independent)
using CLBufferHandle = std::uintptr_t;

struct OpenCLDeviceInfo {
    std::string platform_name;
    std::string device_name;
    std::string driver_version;
    std::uint32_t compute_units = 0;
    std::size_t global_memory = 0;
    std::string opencl_version;
};

struct UploadResult {
    bool success = false;
    std::string error;

    // Number of surfels uploaded
    std::size_t surfel_count = 0;

    // Generation that was uploaded
    Generation source_generation = 0;

    // Buffer handles (implementation-defined)
    CLBufferHandle position_buffer = 0;
    CLBufferHandle normal_buffer = 0;
    CLBufferHandle color_buffer = 0;
    CLBufferHandle radius_buffer = 0;
    CLBufferHandle material_buffer = 0;
};

class OpenCLUploadAdapter {
public:
    // Construct with OpenCL context handle (platform-specific)
    // If context_handle is 0, the adapter is inactive and all operations
    // return success=false with an appropriate error.
    explicit OpenCLUploadAdapter(std::uintptr_t context_handle = 0);

    ~OpenCLUploadAdapter();

    // Non-copyable
    OpenCLUploadAdapter(const OpenCLUploadAdapter&) = delete;
    OpenCLUploadAdapter& operator=(const OpenCLUploadAdapter&) = delete;

    // Moveable
    OpenCLUploadAdapter(OpenCLUploadAdapter&& other) noexcept;
    OpenCLUploadAdapter& operator=(OpenCLUploadAdapter&& other) noexcept;

    // Check if the adapter is active (has a valid OpenCL context)
    bool is_active() const { return context_handle_ != 0; }

    // Query device info
    OpenCLDeviceInfo query_device() const;

    // Upload surfel cache to GPU buffers
    // Returns error if cache generation does not match expected_body_generation
    UploadResult upload_cache(const SurfelCache& cache,
                              Generation expected_body_generation,
                              TopologyGeneration expected_topology_generation);

    // Release GPU buffers
    void release_buffers();

    // Get the last uploaded generation
    Generation last_uploaded_generation() const { return last_uploaded_generation_; }

private:
    std::uintptr_t context_handle_ = 0;
    Generation last_uploaded_generation_ = 0;

    // Internal buffer handles
    struct Buffers {
        CLBufferHandle positions = 0;
        CLBufferHandle normals = 0;
        CLBufferHandle colors = 0;
        CLBufferHandle radii = 0;
        CLBufferHandle materials = 0;
    } buffers_;
};

} // namespace delirium::m2