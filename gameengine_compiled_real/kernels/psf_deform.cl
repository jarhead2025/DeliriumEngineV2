/* =========================================================================
 * psf_deform.cl - GPU-side live deformation (signatures frozen).
 * Applies modal amplitudes K and seam curves directly onto the 64-byte
 * records in VRAM, so a deforming object never round-trips to the CPU.
 * Same FMA-only philosophy as psf_splat.cl.
 *
 * RECORD LAYOUT (matches Types.h / psf_splat.cl byte-for-byte):
 *   SurfaceRecord: 0: ushort anchor_idx[4] | 8: u32 surface_def
 *                  12: float coef[8]  <- deformation basis weights
 *   AnchorRecord : 32: ushort pos[3]  (quantized over bbox, 0..65535)
 *                  44: uchar  normal[3] (n*0.5+0.5, 0..255)
 *
 * COEF MODE SHAPE LAYOUT (what apply_modes consumes):
 *   mode_shapes = [n_modes][n_surfaces][8]
 *   The CPU's PSM v2 table is [n_modes][n_anchor + n_surf*8]; the host
 *   extracts the coefficient part (offset n_anchor) into this layout
 *   before upload, so the frozen kernel signature stays unchanged.
 *   coef[s] += sum_m K[m] * shape[m][s][i]
 * ========================================================================= */

/* coef[8] += sum_m K[m] * mode_shape[m][surface][8]
 * one work-item = one surface. */
__kernel void apply_modes(
    __global uchar        *surfaces,      /* SurfaceRecord blob, in place  */
    __global const float  *mode_shapes,   /* [n_modes][n_surf][8]          */
    __global const float  *K,             /* [n_modes] amplitudes          */
    const uint n_surfaces, const uint n_modes)
{
    const uint s = get_global_id(0);
    if (s >= n_surfaces) return;

    __global float *coef = (__global float *)(surfaces + s*64u + 12u);
    for (uint m = 0; m < n_modes; ++m) {
        const float km = K[m];
        if (km == 0.0f) continue;
        const __global float *sc =
            mode_shapes + ((size_t)m * n_surfaces + s) * 8u;
        for (uint i = 0; i < 8; ++i)
            coef[i] = fmaf(km, sc[i], coef[i]);
    }
}

/* Displace each anchor along its stored normal by
 *   d[a] = sum_m K[m] * shape[m][a]
 * one work-item = one anchor. Quantization mirrors AnchorPool::SetPosition
 * (clamp to bbox, round to nearest of 65535) so the GPU result matches the
 * CPU path bit-for-bit. shape here is the ANCHOR part of the PSM v2 table:
 * [n_modes][n_anchors] floats (host extracts it from the full table). */
__kernel void apply_anchor_modes(
    __global uchar        *anchors,       /* AnchorRecord blob, in place   */
    __global const float  *mode_shapes,   /* [n_modes][n_anchors]          */
    __global const float  *K,             /* [n_modes] amplitudes          */
    const float3 bbmin, const float3 bbspan,
    const uint n_anchors,
    const uint n_modes)
{
    const uint a = get_global_id(0);
    if (a >= n_anchors) return;

    float d = 0.0f;
    for (uint m = 0; m < n_modes; ++m) {
        const float km = K[m];
        if (km == 0.0f) continue;
        d = fmaf(km, mode_shapes[(size_t)m * n_anchors + a], d);
    }
    if (d == 0.0f) return;

    __global uchar *rec = anchors + a*64u;
    __global ushort *qp = (__global ushort *)(rec + 32u);
    __global uchar *qn  = rec + 44u;
    const float3 p = bbmin + bbspan * (float3)(qp[0], qp[1], qp[2])
                                        * (1.0f/65535.0f);
    const float3 n = fast_normalize((float3)(qn[0], qn[1], qn[2])
                                     * (2.0f/255.0f)
                                     - (float3)(1.0f, 1.0f, 1.0f));
    const float3 np = p + d * n;
    /* clamp to bbox, quantize to 16-bit (mirror of AnchorPool::SetPosition) */
    qp[0] = (ushort)(clamp((np.x - bbmin.x) / bbspan.x, 0.0f, 1.0f)
                     * 65535.0f + 0.5f);
    qp[1] = (ushort)(clamp((np.y - bbmin.y) / bbspan.y, 0.0f, 1.0f)
                     * 65535.0f + 0.5f);
    qp[2] = (ushort)(clamp((np.z - bbmin.z) / bbspan.z, 0.0f, 1.0f)
                     * 65535.0f + 0.5f);
}

/* write seam curves (a,b int8 pairs) coming from the physics step.
 * one work-item = one seam side. */
__kernel void apply_seams(
    __global char         *stitch,        /* 8 int8 per surface            */
    __global const int2   *seam_updates,  /* (surface<<2|edge, packed a,b) */
    const uint n_updates)
{
    const uint i = get_global_id(0);
    if (i >= n_updates) return;
    const int2 u = seam_updates[i];
    const uint surf = (uint)(u.x >> 2);
    const uint edge = (uint)(u.x & 3);
    const char a = (char)(u.y & 0xFF);
    const char b = (char)((u.y >> 8) & 0xFF);
    __global char *e = stitch + surf * 8u + edge * 2u;
    e[0] = a;
    e[1] = b;
}