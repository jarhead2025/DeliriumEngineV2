// FILEPATH: kernels/psf_deform.cl
__kernel void psfDeform(__global float* input, __global float* output, const int size) {
    int id = get_global_id(0);
    if (id < size) {
        // Add deformation logic here
        output[id] = input[id] * 1.1; // Example transformation
    }
}
