// FILEPATH: kernels/hit_fill_stub.cl
// Placeholder OpenCL kernels for later G3+ GPU path (host fill used until wired).
__kernel void clear_fb(__global uint* fb, const uint color) {
    const size_t i = get_global_id(0);
    fb[i] = color;
}

// Real path: intersect equation surface coeffs → write sparse hits
__kernel void intersect_hits(__global const float* coeffs,
                             const int ncoeff,
                             __global int2* hits,
                             __global int* hit_count,
                             const int max_hits) {
    const int id = get_global_id(0);
    if (id == 0) hit_count[0] = 0;
    // Stub: device path not yet driven from host
    (void)coeffs;
    (void)ncoeff;
    (void)hits;
    (void)max_hits;
}

__kernel void pattern_fill(__global uint* fb,
                           const int width,
                           const int height,
                           __global const int2* hits,
                           const int nhits) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    (void)hits;
    (void)nhits;
    // Neighbor average implemented on host in G3; port here next
}
