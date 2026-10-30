// FILEPATH: src/core/OpenCLManager.cpp
#include "OpenCLManager.h"
#include <stdexcept>

cl_context CLManager::createContext(const std::vector<cl_device_id>& devices) {
    cl_int err;
    cl_platform_id platform;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        throw std::runtime_error("Failed to find a platform");
    }

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
    return clCreateContext(properties, devices.size(), devices.data(), nullptr, nullptr, &err);
}

void CLManager::releaseContext(cl_context context) {
    clReleaseContext(context);
}
