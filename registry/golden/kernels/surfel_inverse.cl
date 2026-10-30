// =============================================================================
// surfel_inverse.cl - INVERSE (per-pixel) rasterization for DeliRium.
//
// What the user actually asked for: instead of splatting each surfel as a disc
// onto the screen (forward), we treat EACH PIXEL of the screen as the unit of
// work. One thread per pixel. The pixel "asks the model" what surface is there
// by finding the nearest projected surface sample along its ray. No discs, no
// giant points -- resolution is exactly the screen (W x H). A final pass fills
// genuine holes (pixels that found nothing but sit inside the silhouette) by
// averaging neighbors.
//
// To keep the per-pixel query cheap, surfels are first binned into a screen
// grid (bin_kernel); each pixel then only inspects its own bin + 8 neighbors
// instead of all surfels.
// =============================================================================

typedef struct {
    float px, py, pz;
    float nx, ny, nz;
    float r,  g,  b;
    float radius;
} Surfel;

// project a surfel to screen; returns 0 if behind camera
static int project(const Surfel* s, float3 eye, float3 fwd, float3 right,
                   float3 up, float focal, int W, int H,
                   float* outx, float* outy, float* outz) {
    float3 pos = (float3)(s->px, s->py, s->pz);
    float3 rel = pos - eye;
    float z = dot(rel, fwd);
    if (z < 1e-3f) return 0;
    *outx = dot(rel, right) / z * focal + W * 0.5f;
    *outy = H * 0.5f - dot(rel, up) / z * focal;
    *outz = z;
    return 1;
}

// Pass 1: bin each surfel into the screen cell it projects to. We store, per
// cell, the nearest-depth surfel index (atomic min on a packed depth|index).
__kernel void bin_surfels(
    __global const Surfel* surfels, const uint n_surfels,
    __global ulong* cell,            // packed (depthbits<<32 | index), per cell
    const uint gw, const uint gh, const uint cellpx,
    const float3 eye, const float3 fwd, const float3 right, const float3 up,
    const float focal, const uint W, const uint H)
{
    uint i = get_global_id(0);
    if (i >= n_surfels) return;
    float sx, sy, sz;
    if (!project(&surfels[i], eye, fwd, right, up, focal, W, H, &sx, &sy, &sz))
        return;
    int cxp = (int)(sx / cellpx), cyp = (int)(sy / cellpx);
    if (cxp < 0 || cxp >= (int)gw || cyp < 0 || cyp >= (int)gh) return;
    uint dbits = as_uint(sz);
    ulong key = ((ulong)dbits << 32) | (ulong)i;
    atom_min(&cell[cyp * gw + cxp], key);  // nearest surfel in this cell
}

// Pass 2: one thread per PIXEL. Look in the pixel's cell + 8 neighbors for the
// surfel whose projection is nearest this pixel and nearest in depth; that is
// the surface the pixel's ray hits. Write its shaded color, or mark empty.
__kernel void shade_pixels(
    __global const Surfel* surfels,
    __global const ulong* cell,
    const uint gw, const uint gh, const uint cellpx,
    const float3 eye, const float3 fwd, const float3 right, const float3 up,
    const float focal, const uint W, const uint H,
    const float3 light_dir,
    __global uchar4* img, __global float* depth)
{
    uint x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    uint pidx = y * W + x;
    int cxp = x / cellpx, cyp = y / cellpx;
    float bestd2 = 1e30f, bestz = 1e30f;
    int best = -1;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int gx = cxp + dx, gy = cyp + dy;
        if (gx < 0 || gx >= (int)gw || gy < 0 || gy >= (int)gh) continue;
        ulong key = cell[gy * gw + gx];
        if (key == 0xFFFFFFFFFFFFFFFFUL) continue;
        uint i = (uint)(key & 0xFFFFFFFFUL);
        float sx, sy, sz;
        if (!project(&surfels[i], eye, fwd, right, up, focal, W, H, &sx, &sy, &sz))
            continue;
        float ddx = sx - (x + 0.5f), ddy = sy - (y + 0.5f);
        float d2 = ddx*ddx + ddy*ddy;
        // nearest in screen space, tie-broken by depth
        if (d2 < bestd2 || (d2 == bestd2 && sz < bestz)) {
            bestd2 = d2; bestz = sz; best = (int)i;
        }
    }
    if (best < 0 || bestd2 > (float)(cellpx*cellpx*4)) {
        img[pidx] = (uchar4)(0,0,0,0);   // empty -> hole (alpha 0)
        depth[pidx] = 1e30f;
        return;
    }
    Surfel s = surfels[best];
    float3 nrm = (float3)(s.nx, s.ny, s.nz);
    float lam = clamp(dot(nrm, light_dir), 0.0f, 1.0f) * 0.25f + 0.75f;
    uchar r = (uchar)(clamp(s.r*lam,0.0f,1.0f)*255.0f);
    uchar g = (uchar)(clamp(s.g*lam,0.0f,1.0f)*255.0f);
    uchar b = (uchar)(clamp(s.b*lam,0.0f,1.0f)*255.0f);
    img[pidx] = (uchar4)(r,g,b,255);
    depth[pidx] = bestz;
}

// Pass 3: fill genuine holes (alpha 0) that are surrounded by surface, by
// copying the nearest filled neighbor (no averaging cream+black -> no grey).
__kernel void fill_holes(
    __global uchar4* img, const uint W, const uint H,
    __global const uchar4* src)
{
    uint x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    uint pidx = y * W + x;
    if (src[pidx].w != 0) { img[pidx] = src[pidx]; return; }  // already filled
    // look for nearest filled among 8 neighbors; take the closest one
    int found = 0; uchar4 best = (uchar4)(15,15,18,255);
    for (int dy = -1; dy <= 1 && !found; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx==0 && dy==0) continue;
        int X = (int)x+dx, Y = (int)y+dy;
        if (X<0||X>=(int)W||Y<0||Y>=(int)H) continue;
        uchar4 nb = src[Y*W+X];
        if (nb.w != 0) { best = nb; found = 1; break; }
    }
    img[pidx] = found ? (uchar4)(best.x,best.y,best.z,255)
                      : (uchar4)(15,15,18,255);
}
