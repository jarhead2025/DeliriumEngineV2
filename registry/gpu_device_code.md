```cl
// FILEPATH: kernels/surfel_march.cl

__kernel void surfelMarch(__global const float4* positions,
                         __global const float3* normals,
                         __global const float2* texCoords,
                         __global float4* outputColors,
                         __constant int width,
                         __constant int height) {
    // Get the global ID
    int gid = get_global_id(0);
    
    // Calculate the 2D position from the global ID
    int x = gid % width;
    int y = gid / width;

    // Ensure we are within bounds
    if (x >= width || y >= height) {
        return;
    }

    // Example processing: just copy the texture coordinates to output colors
    float2 texCoord = texCoords[gid];
    outputColors[gid] = (float4)(texCoord.x, texCoord.y, 0.0f, 1.0f);
}
```