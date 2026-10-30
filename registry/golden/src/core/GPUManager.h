// FILEPATH: src/core/GPUManager.h

#ifndef GPUMANAGER_H
#define GPUMANAGER_H

#include <CL/cl.h>
#include <GL/glew.h>

class GPUManager {
public:
    void initializeOpenCL();
    void initializeOpenGL();
    void cleanupOpenCL();

private:
    cl_context context;
    cl_command_queue queue;
};

#endif // GPUMANAGER_H
