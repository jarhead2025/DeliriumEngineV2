// FILEPATH: src/core/OpenCLManager.cpp
#include <CL/cl.h>
#include <iostream>

cl_command_queue create_intel_command_queue(cl_context ctx, cl_device_id dev) {
    cl_int err = 0;
    // Use the appropriate OpenCL function to create a command queue
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << err << std::endl;
        return nullptr;
    }
    return q;
}

int main() {
    // Example usage of the function
    cl_int err = 0;
    // Assume context and device are already created elsewhere
    cl_context ctx = nullptr;
    cl_device_id dev = nullptr;

    cl_command_queue q = create_intel_command_queue(ctx, dev);
    if (q) {
        std::cout << "Command queue created successfully." << std::endl;
        // Clean up the command queue after use
        clReleaseCommandQueue(q);
    }

    return 0;
}
