// FILEPATH: shaders/psf_splat.comp.glsl

#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(std140, binding = 0) buffer SSBO {
    float data[];
} ssbo;

void main() {
    uint gid = gl_GlobalInvocationID.x + gl_NumWorkGroups.x * gl_WorkGroupID.y * gl_WorkGroupSize.x +
               gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    // Example computation: evaluate 8-term corner-vanishing equations
    float result = ssbo.data[gid] * 3.0f;

    // Store the result back in SSBO
    ssbo.data[gid] = result;
}
