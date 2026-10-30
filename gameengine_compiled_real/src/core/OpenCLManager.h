// FILEPATH: src/core/OpenCLManager.h
#ifndef OPENCLMANAGER_H
#define OPENCLMANAGER_H

#include <CL/cl.h>
#include <vector>

class CLManager {
public:
    static cl_context createContext(const std::vector<cl_device_id>& devices);
    static void releaseContext(cl_context context);
};

#endif // OPENCLMANAGER_H
