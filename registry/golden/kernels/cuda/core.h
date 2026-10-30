// FILEPATH: kernels/cuda/core.h
#ifndef CUDA_CORE_H_
#define CUDA_CORE_H_

#include <cuda_runtime.h>
#include <iostream>

// Function to check CUDA error
inline void checkCudaError(const char* fileName, const int lineNumber) {
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error)
                  << " at " << fileName << ":" << lineNumber << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Macro to check CUDA error with file name and line number
#define CHECK_CUDA_ERROR() checkCudaError(__FILE__, __LINE__)

#endif // CUDA_CORE_H_
