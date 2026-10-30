// =============================================================================
// surfel_splat.cl - GPU surfel rasterizer for the DeliRium tree model.
//
// One work-item per SURFEL. Each surfel (pos, normal, color, radius) is
// projected to screen and splatted with a 64-bit atom_min depth test -- the
// same proven trick as psf_splat.cl / spf_splat.cl that already runs on the
// Arc A770. Dense sampling is free here: the GPU processes all surfels in
// parallel, so there are no sampling holes like the CPU/numpy splat had.
//
// The surfel's screen radius comes from its world radius / depth, so close
// surfels cover more pixels and far ones fewer -> LOD for free, cost scales
// with visible surface (the efficiency win over the triangle mesh).
//
// Framebuffer is packed ulong: (as_uint(depth) << 32) | rgba8. atom_min keeps
// the nearest surfel per pixel without races. A resolve kernel unpacks to
// RGBA8 for display.
// =============================================================================

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef struct {
    float px, py, pz;      // world position
    float nx, ny, nz;      // normal
    float r,  g,  b;       // color
    float radius;          // world-space surfel radius
} Surfel;

/* clear packed framebuffer to far depth */
__kernel void surfel_clear(__global ulong* fb, const uint n) {
    uint i = get_global_id(0);
    if (i < n) fb[i] = 0xFFFFFFFFFFFFFFFFUL;
}

/* one work-item per surfel: project, shade, splat a disk via atom_min */
__kernel void surfel_splat(
    __global const Surfel* surfels,
    const uint            n_surfels,
    __global ulong*       fb,
    const uint            width,
    const uint            height,
    // camera basis
    const float3          eye,
    const float3          fwd,
    const float3          right,
    const float3          up,
    const float           focal,
    const float3          light_dir)
{
    uint i = get_global_id(0);
    if (i >= n_surfels) return;
    Surfel s = surfels[i];

    float3 pos = (float3)(s.px, s.py, s.pz);
    float3 rel = pos - eye;
    float z = dot(rel, fwd);
    if (z < 1e-3f) return;                       // behind camera

    float sx = dot(rel, right) / z * focal + width  * 0.5f;
    float sy = height * 0.5f - dot(rel, up) / z * focal;

    // screen radius from world radius / depth (LOD for free)
    int sr = (int)(s.radius / z * focal);
    sr = clamp(sr, 0, 8);

    // lambert shade, kept bright so light colors stay light (no grey wash)
    float3 nrm = (float3)(s.nx, s.ny, s.nz);
    float lam = clamp(dot(nrm, light_dir), 0.0f, 1.0f) * 0.25f + 0.75f;
    uint r = (uint)(clamp(s.r * lam, 0.0f, 1.0f) * 255.0f);
    uint g = (uint)(clamp(s.g * lam, 0.0f, 1.0f) * 255.0f);
    uint b = (uint)(clamp(s.b * lam, 0.0f, 1.0f) * 255.0f);
    uint rgba = (r << 24) | (g << 16) | (b << 8) | 0xFF;
    uint dbits = as_uint(z);
    ulong key = ((ulong)dbits << 32) | (ulong)rgba;

    int cx = (int)sx, cy = (int)sy;
    for (int dy = -sr; dy <= sr; ++dy) {
        for (int dx = -sr; dx <= sr; ++dx) {
            if (dx*dx + dy*dy > sr*sr) continue;
            int X = cx + dx, Y = cy + dy;
            if (X < 0 || X >= (int)width || Y < 0 || Y >= (int)height) continue;
            atom_min(&fb[Y * width + X], key);
        }
    }
}

/* resolve packed fb -> RGBA8 image; empty pixels get the background */
__kernel void surfel_resolve(
    __global const ulong* fb,
    __global uchar4*      img,
    const uint            n,
    const uchar4          background)
{
    uint i = get_global_id(0);
    if (i >= n) return;
    ulong key = fb[i];
    if (key == 0xFFFFFFFFFFFFFFFFUL) { img[i] = background; return; }
    uint rgba = (uint)(key & 0xFFFFFFFFUL);
    img[i] = (uchar4)((rgba >> 24) & 0xFF, (rgba >> 16) & 0xFF,
                      (rgba >> 8) & 0xFF, 0xFF);
}
