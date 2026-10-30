// FILEPATH: src/core/OpenCLManager.h
#ifndef OPENCL_MANAGER_H
#define OPENCL_MANAGER_H

#include <CL/cl.h>

namespace OpenCL {

cl_platform_id get_intel_platform();

cl_context create_intel_context();

cl_command_queue create_intel_command_queue(cl_context ctx, cl_device_id dev);

const char* get_cl_error_name(cl_int err);

}

#endif // OPENCL_MANAGER_H
