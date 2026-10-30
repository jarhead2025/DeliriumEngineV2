#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string platform_string(cl_platform_id platform, cl_platform_info field) {
    size_t bytes = 0;
    if (clGetPlatformInfo(platform, field, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0) {
        return "<unavailable>";
    }
    std::string text(bytes, '\0');
    if (clGetPlatformInfo(platform, field, bytes, text.data(), nullptr) != CL_SUCCESS) {
        return "<unavailable>";
    }
    if (!text.empty() && text.back() == '\0') text.pop_back();
    return text;
}

std::string device_string(cl_device_id device, cl_device_info field) {
    size_t bytes = 0;
    if (clGetDeviceInfo(device, field, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0) {
        return "<unavailable>";
    }
    std::string text(bytes, '\0');
    if (clGetDeviceInfo(device, field, bytes, text.data(), nullptr) != CL_SUCCESS) {
        return "<unavailable>";
    }
    if (!text.empty() && text.back() == '\0') text.pop_back();
    return text;
}

template <typename T>
T device_value(cl_device_id device, cl_device_info field, T fallback = {}) {
    T value{};
    if (clGetDeviceInfo(device, field, sizeof(T), &value, nullptr) != CL_SUCCESS) return fallback;
    return value;
}

const char* device_type(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU) return "GPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return "accelerator";
    if (type & CL_DEVICE_TYPE_CPU) return "CPU";
    return "other";
}

bool extension_present(const std::string& extensions, const char* name) {
    const std::string needle(name);
    size_t begin = 0;
    while (begin < extensions.size()) {
        const size_t end = extensions.find(' ', begin);
        const std::string token = extensions.substr(begin, end - begin);
        if (token == needle) return true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    const bool quiet = argc > 1 && std::strcmp(argv[1], "--quiet") == 0;

    cl_uint platform_count = 0;
    const cl_int platform_result = clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform_result != CL_SUCCESS || platform_count == 0) {
        std::cerr << "No OpenCL platform found (clGetPlatformIDs=" << platform_result << ").\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        std::cerr << "Could not enumerate OpenCL platforms.\n";
        return 1;
    }

    cl_uint discovered = 0;
    for (cl_uint pi = 0; pi < platform_count; ++pi) {
        const cl_platform_id platform = platforms[pi];
        if (!quiet) {
            std::cout << "Platform " << pi << ": "
                      << platform_string(platform, CL_PLATFORM_NAME)
                      << " | " << platform_string(platform, CL_PLATFORM_VENDOR)
                      << " | " << platform_string(platform, CL_PLATFORM_VERSION) << "\n";
        }

        cl_uint device_count = 0;
        const cl_int device_result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        if (device_result == CL_DEVICE_NOT_FOUND || device_count == 0) continue;
        if (device_result != CL_SUCCESS) {
            std::cerr << "  Device enumeration failed: " << device_result << "\n";
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) != CL_SUCCESS) {
            std::cerr << "  Could not read device list.\n";
            continue;
        }

        for (cl_uint di = 0; di < device_count; ++di) {
            const cl_device_id device = devices[di];
            ++discovered;
            const std::string extensions = device_string(device, CL_DEVICE_EXTENSIONS);
            const cl_device_type type = device_value<cl_device_type>(device, CL_DEVICE_TYPE);
            const cl_uint compute_units = device_value<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
            const size_t workgroup = device_value<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            const cl_ulong global_mem = device_value<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);

            if (!quiet) {
                std::cout << "  Device " << di << ": " << device_string(device, CL_DEVICE_NAME) << "\n"
                          << "    type: " << device_type(type)
                          << ", OpenCL C: " << device_string(device, CL_DEVICE_OPENCL_C_VERSION) << "\n"
                          << "    vendor: " << device_string(device, CL_DEVICE_VENDOR) << "\n"
                          << "    compute units: " << compute_units
                          << ", max work-group: " << workgroup
                          << ", global memory MiB: " << (global_mem / (1024ULL * 1024ULL)) << "\n"
                          << "    capabilities: fp64="
                          << (extension_present(extensions, "cl_khr_fp64") ? "yes" : "no")
                          << ", int64 atomics="
                          << ((extension_present(extensions, "cl_khr_int64_base_atomics") &&
                               extension_present(extensions, "cl_khr_int64_extended_atomics")) ? "yes" : "no")
                          << ", command-buffer="
                          << (extension_present(extensions, "cl_khr_command_buffer") ? "yes" : "no")
                          << "\n";
            }
        }
    }

    if (discovered == 0) {
        std::cerr << "OpenCL platforms were present, but no usable devices were found.\n";
        return 1;
    }

    if (!quiet) std::cout << "M0 probe passed: " << discovered << " OpenCL device(s) discovered.\n";
    return 0;
}
