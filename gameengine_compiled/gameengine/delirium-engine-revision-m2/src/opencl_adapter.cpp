#include "delirium_m2/opencl_adapter.hpp"

#include <cstring>

namespace delirium::m2 {

// ---------------------------------------------------------------------------
// OpenCLUploadAdapter
// ---------------------------------------------------------------------------
//
// NOTE: This is a stub implementation that does NOT link against OpenCL.
// It provides the full interface for compilation and testing without a GPU.
// When OpenCL is enabled via CMake, the real implementation replaces this
// file with src/opencl_adapter_cl.cpp.
// ---------------------------------------------------------------------------

OpenCLUploadAdapter::OpenCLUploadAdapter(std::uintptr_t context_handle)
    : context_handle_(context_handle) {}

OpenCLUploadAdapter::~OpenCLUploadAdapter() {
    release_buffers();
}

OpenCLUploadAdapter::OpenCLUploadAdapter(OpenCLUploadAdapter&& other) noexcept
    : context_handle_(other.context_handle_),
      last_uploaded_generation_(other.last_uploaded_generation_),
      buffers_(other.buffers_) {
    other.context_handle_ = 0;
    other.last_uploaded_generation_ = 0;
    other.buffers_ = {};
}

OpenCLUploadAdapter& OpenCLUploadAdapter::operator=(OpenCLUploadAdapter&& other) noexcept {
    if (this != &other) {
        release_buffers();
        context_handle_ = other.context_handle_;
        last_uploaded_generation_ = other.last_uploaded_generation_;
        buffers_ = other.buffers_;
        other.context_handle_ = 0;
        other.last_uploaded_generation_ = 0;
        other.buffers_ = {};
    }
    return *this;
}

OpenCLDeviceInfo OpenCLUploadAdapter::query_device() const {
    OpenCLDeviceInfo info;
    if (context_handle_ == 0) {
        info.platform_name = "none";
        info.device_name = "no-context";
        info.opencl_version = "N/A";
    }
    // Real implementation would call clGetDeviceInfo here
    return info;
}

UploadResult OpenCLUploadAdapter::upload_cache(const SurfelCache& cache,
                                                Generation expected_body_generation,
                                                TopologyGeneration expected_topology_generation) {
    UploadResult result;

    if (context_handle_ == 0) {
        result.success = false;
        result.error = "OpenCL context not available";
        return result;
    }

    // Enforce generation invariant: refuse to upload a cache whose source
    // generation does not match the expected BodyState generation
    if (cache.source_generation() != expected_body_generation) {
        result.success = false;
        result.error = "cache generation mismatch: cache=" +
                       std::to_string(cache.source_generation()) +
                       " expected=" + std::to_string(expected_body_generation);
        return result;
    }

    if (cache.topology_generation() != expected_topology_generation) {
        result.success = false;
        result.error = "topology generation mismatch";
        return result;
    }

    // Stub: in the real implementation, this would create cl_mem buffers
    // and call clEnqueueWriteBuffer for each surfel attribute array
    result.success = true;
    result.surfel_count = cache.surfel_count();
    result.source_generation = cache.source_generation();

    // Stub handles (would be real cl_mem pointers)
    result.position_buffer = 1;
    result.normal_buffer = 2;
    result.color_buffer = 3;
    result.radius_buffer = 4;
    result.material_buffer = 5;

    last_uploaded_generation_ = cache.source_generation();

    return result;
}

void OpenCLUploadAdapter::release_buffers() {
    // Stub: would call clReleaseMemObject on each buffer
    buffers_ = {};
}

} // namespace delirium::m2