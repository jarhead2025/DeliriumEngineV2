// FILEPATH: src/core/GPUManager.cpp

#include "GPUManager.h"
#include <iostream>
#include <CL/cl.h>

void GPUManager::initializeOpenCL() {
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    if (numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return;
    }

    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, nullptr);

    for (cl_uint i = 0; i < numPlatforms; ++i) {
        char platformName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);

        if (strstr(platformName, "Intel") != nullptr) {
            cl_platform_id intelPlatform = platforms[i];

            cl_device_id device;
            cl_int err = clGetDeviceIDs(intelPlatform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            if (err != CL_SUCCESS) {
                err = clGetDeviceIDs(intelPlatform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
                if (err != CL_SUCCESS) {
                    std::cerr << "Failed to get device IDs." << std::endl;
                    delete[] platforms;
                    return;
                }
            }

            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create OpenCL context." << std::endl;
                delete[] platforms;
                return;
            }

            queue = clCreateCommandQueue(context, device, 0, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create command queue." << std::endl;
                clReleaseContext(context);
                delete[] platforms;
                return;
            }

            delete[] platforms;
            break;
        }
    }
}

void GPUManager::initializeOpenGL() {
    // Initialize OpenGL context
    // This part is platform-specific and depends on the windowing system used (e.g., GLFW, SDL)
    // For simplicity, assume an OpenGL context is already created

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        return;
    }
}

void GPUManager::cleanupOpenCL() {
    if (queue) {
        clReleaseCommandQueue(queue);
    }

    if (context) {
        clReleaseContext(context);
    }
}
