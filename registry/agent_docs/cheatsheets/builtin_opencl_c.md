---
agent: pipeline
topic: opencl_c
updated: 2026-09-07T01:03:01+00:00
---

# opencl_c

```cpp
// Minimal OpenCL C API (portable; avoid CL/cl.hpp)
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
cl_platform_id plats[8]; cl_uint np = 0;
clGetPlatformIDs(8, plats, &np);
cl_platform_id chosen = plats[0];
for (cl_uint i = 0; i < np; ++i) {
    char name[256] = {};
    clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
    if (std::strstr(name, "Intel") || std::strstr(name, "intel")) { chosen = plats[i]; break; }
}
cl_device_id dev = nullptr;
if (clGetDeviceIDs(chosen, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS)
    clGetDeviceIDs(chosen, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr);
cl_int err = CL_SUCCESS;
cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
```
