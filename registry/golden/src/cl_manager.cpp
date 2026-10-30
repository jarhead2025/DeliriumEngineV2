// FILEPATH: src/cl_manager.cpp
#include "OpenCLManager.h"
#include <stdexcept>

cl_context CLManager::createContext() {
    cl_int err;
    cl_platform_id platform;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        throw std::runtime_error("Failed to find a platform");
    }

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
    return clCreateContext(properties, 1, &device, nullptr, nullptr, &err);
}

void CLManager::releaseContext(cl_context context) {
    clReleaseContext(context);
}
