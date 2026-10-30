// FILEPATH: shaders/psf_splat.comp.glsl

#version 430 core

struct CVBasisTensor {
    vec4 cornerTerms[2];
};

layout (std430) buffer BasisTensorBuffer {
    CVBasisTensor data[];
};

layout (local_size_x = 8, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint gid = gl_GlobalInvocationID.x;

    // Initialize variables
    float accumulatedValue = 0.0f;
    vec4 sumVector = vec4(0.0);

    #define MAX_TERMS 8

    for (int i = 0; i < MAX_TERMS; ++i) {
        uint termIndex = gid * 2 + i / 4; // Access the correct vec4
        int elementIndex = i % 4;
        sumVector += data[termIndex].cornerTerms[elementIndex];
    }

    accumulatedValue = dot(sumVector, sumVector);

    // Store result in global memory (e.g., to an output buffer)
    imageStore(outputImage, ivec2(gid, 0), vec4(accumulatedValue));
}
