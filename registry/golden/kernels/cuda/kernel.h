// FILEPATH: kernels/cuda/kernel.h
#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

__global__ void vectorAdd(const float* a, const float* b, float* c, int n);

#endif // CUDA_KERNEL_H_
