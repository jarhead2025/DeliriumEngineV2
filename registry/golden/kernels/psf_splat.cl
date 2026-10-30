/* =========================================================================
 * psf_splat.cl  --  PSF v2 surface evaluation + sub-pixel splat renderer
 * Hand-written, dependency-free OpenCL C 1.2. No SDK helpers, no libs.
 *
 * PIPELINE (one dispatch per frame):
 *   clear_fb   : reset packed depth|color buffer to "far, black"
 *   psf_splat  : sample -> evaluate surface equations -> project -> splat
 *   resolve_fb : unpack winning color per pixel into RGBA8 image
 *
 * WHY THIS IS CHEAP ON THE SAME BASE CIRCUITRY:
 *   Every sample is a straight FMA chain: bilinear lerps (position and
 *   normal), the 8-term corner-vanishing polynomial, 4 stitch terms,
 *   pattern color dot products, one 4x4 matrix multiply. No texture
 *   units, no special-function units in the hot path (cospi only in
 *   color patterns), no divergent loops: every thread runs the same
 *   instruction stream. Reads are 64-byte records reused by thousands
 *   of samples (cache-resident); the only scatter is ONE 64-bit atomic
 *   per sample. ~115 FLOPs/sample -> ~36 GFLOP/s at 1080p60 with
 *   2.5 samples/px: about 1.3% of a GTX 1650.
 *
 * DEPTH-TESTED SPLAT WITHOUT A RASTERIZER:
 *   key = (depth_bits << 32) | RGBA8. atom_min keeps the nearest sample
 *   and its color in ONE atomic op. float32 depth bits are monotonic
 *   for depth >= 0, so integer min == nearest. Needs
 *   cl_khr_int64_extended_atomics (define USE_LONG_ATOMICS).
 *   Fallback for ancient GPUs (no 64-bit atomics): two-pass version,
 *   atomic_min on a uint depth buffer, then a color pass that writes
 *   only where its depth won (benign race: equal depth = same surface).
 *
 * RECORD LAYOUTS (byte offsets inside each 64-byte record; matches the
 * Python encoder psf_v2_quadtree.py byte-for-byte):
 *   AnchorPointBuffer:
 *     32: ushort pos[3]   (quantized over bbox)
 *     44: uchar  normal[3] (n*0.5+0.5 quantized)   <- displacement dir
 *     16: uchar  curve_params[16]                  <- stitch coefs home
 *   ProceduralSurfaceBuffer:
 *      0: ushort anchor_idx[4]
 *     12: float  coef[8]      corner-vanishing deformation basis
 *     44: 3 x { uchar idx_hi4|lo4, uchar idx_hi4|lo4, char K[4] }
 *         color = sum_i (K_i/63) * Pattern_{idx_i}(u,v)  per channel
 * ========================================================================= */

#ifdef USE_LONG_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#endif

/* ---------- record accessors (pure pointer math, no structs: immune to
 *            host/device struct-packing disagreements) ------------------ */

inline float3 anchor_pos(__global const uchar *A, uint i,
                         float3 bbmin, float3 bbspan)
{
    __global const ushort *q = (__global const ushort *)(A + i*64u + 32u);
    return bbmin + bbspan * (float3)(q[0], q[1], q[2]) * (1.0f/65535.0f);
}

inline float3 anchor_nrm(__global const uchar *A, uint i)
{
    __global const uchar *n = A + i*64u + 44u;
    return fast_normalize((float3)(n[0], n[1], n[2]) * (2.0f/255.0f)
                          - (float3)(1.0f, 1.0f, 1.0f));
}

/* ---------- the predetermined equations ------------------------------- */

/* corner-vanishing, edge-separable deformation basis.
 * h == 0 at all four corners  -> surface interpolates its anchors
 * each edge owned by exactly 2 coefs -> sewing = adjust 2 numbers   */
inline float deform_h(const float c[8], float u, float v)
{
    float bu = u * (1.0f - u), bv = v * (1.0f - v);
    float au = 2.0f*u - 1.0f,  av = 2.0f*v - 1.0f;
    float w  = 1.0f - v,       x  = 1.0f - u;
    return fma(c[0], bu*w,  fma(c[1], bu*v,
           fma(c[2], bv*x,  fma(c[3], bv*u,
           fma(c[4], bu*au*w, fma(c[5], bu*au*v,
           fma(c[6], bv*av*x,     c[7]*bv*av*u)))))));
}

/* analytic partials of deform_h (matches Patch::HeightGrad on the host).
 * Used for the TRUE surface normal so shading is the geometric normal of
 * the DEFORMED surface, not the base displacement direction Nd. */
inline void deform_dh(const float c[8], float u, float v,
                      float *dhu, float *dhv)
{
    float bu = u*(1.0f-u), bv = v*(1.0f-v);
    float au = 2.0f*u-1.0f, av = 2.0f*v-1.0f;
    float w = 1.0f-v, x = 1.0f-u;
    float dbu = 1.0f-2.0f*u, dbv = 1.0f-2.0f*v;
    float dcu = dbu*au + 2.0f*bu;          /* d(bu*au)/du */
    float dcv = dbv*av + 2.0f*bv;          /* d(bv*av)/dv */
    *dhu = c[0]*dbu*w + c[1]*dbu*v - c[2]*bv + c[3]*bv
         + c[4]*dcu*w + c[5]*dcu*v - c[6]*bv*av + c[7]*bv*av;
    *dhv = -c[0]*bu + c[1]*bu + c[2]*dbv*x + c[3]*dbv*u
         - c[4]*bu*au + c[5]*bu*au + c[6]*dcv*x + c[7]*dcv*u;
}

/* stitch ("sewing"): s(t) = t(1-t)(a + b t), per edge, int8-quantized.
 * weights (1-v), v, (1-u), u blend each edge correction inward.
 * Lives in anchor curve_params in the full format; passed as its own
 * buffer here for clarity. Physics can rewrite 2 bytes per edge to
 * deform BOTH neighbors crack-free. */
inline float stitch_h(__global const char *S, uint surf, float u, float v,
                      float s_scale)
{
    __global const char *e = S + surf * 8u;           /* 4 edges x (a,b) */
    float s = 0.0f;
    float tu = u*(1.0f-u), tv = v*(1.0f-v);
    s += (1.0f-v) * tu * (e[0]*s_scale + e[1]*s_scale*u);   /* edge v=0 */
    s +=        v * tu * (e[2]*s_scale + e[3]*s_scale*u);   /* edge v=1 */
    s += (1.0f-u) * tv * (e[4]*s_scale + e[5]*s_scale*v);   /* edge u=0 */
    s +=        u * tv * (e[6]*s_scale + e[7]*s_scale*v);   /* edge u=1 */
    return s;
}

/* presaved pattern library (engine-side; files store only indices + K) */
inline float pattern_eval(uint idx, float u, float v)
{
    switch (idx & 15u) {
        case 0:  return 1.0f;
        case 1:  return u;
        case 2:  return v;
        case 3:  return u*v;
        case 4:  return u*u;
        case 5:  return v*v;
        case 6:  return u*u*v;
        case 7:  return u*v*v;
        case 8:  return cospi(u);
        case 9:  return cospi(v);
        case 10: return cospi(2.0f*u);
        case 11: return cospi(2.0f*v);
        case 12: return cospi(u)*cospi(v);
        case 13: return cospi(2.0f*u)*cospi(2.0f*v);
        case 14: return cospi(3.0f*u);
        default: return cospi(3.0f*v);
    }
}

inline float channel_color(__global const uchar *srec, uint ch,
                           float u, float v)
{
    __global const uchar *c = srec + 44u + ch*6u;
    uint i0 = c[0] >> 4, i1 = c[0] & 15u, i2 = c[1] >> 4, i3 = c[1] & 15u;
    __global const char *K = (__global const char *)(c + 2);
    float r = (K[0]*(1.0f/63.0f)) * pattern_eval(i0, u, v);
    r = fma((float)K[1]*(1.0f/63.0f), pattern_eval(i1, u, v), r);
    r = fma((float)K[2]*(1.0f/63.0f), pattern_eval(i2, u, v), r);
    r = fma((float)K[3]*(1.0f/63.0f), pattern_eval(i3, u, v), r);
    return clamp(r, 0.0f, 1.0f);
}

/* ---------- kernels ---------------------------------------------------- */

__kernel void clear_fb(__global ulong *fb, uint n)
{
    uint g = get_global_id(0);
    if (g < n) fb[g] = 0xFFFFFFFFFFFFFFFFUL;   /* depth=+inf-ish, color=0 */
}

/* one work-item = one surface sample.
 * sample_off : prefix sums, surface s owns gids [off[s], off[s+1])
 * sample_ss  : per-surface grid side (adaptive: from projected size)   */
__kernel void psf_splat(
    __global const uchar  *anchors,
    __global const float4 *snorm,         /* smooth per-anchor shading normal */
    __global const uchar  *surfaces,
    __global const char   *stitch,        /* 8 int8 per surface */
    __global const uint   *sample_off,
    __global const uint   *sample_ss,    /* (ss_u << 16) | ss_v, per surface */
    const uint  n_surfaces,
    const float3 bbmin, const float3 bbspan,
    const float  s_scale,                  /* stitch dequant scale */
    const float16 mvp,                     /* row-major */
    const float3 light_dir,
    const float3 eye,                      /* camera position (world) */
    const uint  width, const uint height,
    __global volatile ulong *fb
#ifndef USE_LONG_ATOMICS
    , __global volatile uint *depth32     /* fallback path, pass 1 use */
    , const uint pass                     /* 0 = depth, 1 = color */
#endif
)
{
    uint gid = get_global_id(0);
    if (gid >= sample_off[n_surfaces]) return;

    /* binary search: which surface owns this sample id */
    uint lo = 0, hi = n_surfaces;
    while (hi - lo > 1u) {
        uint mid = (lo + hi) >> 1;
        if (sample_off[mid] <= gid) lo = mid; else hi = mid;
    }
    const uint s    = lo;
    const uint k    = gid - sample_off[s];
    const uint ssu  = sample_ss[s] >> 16;          /* anisotropic grid:   */
    const uint ssv  = sample_ss[s] & 0xFFFFu;      /* per-axis density    */
    const float u = (float)(k % ssu) / (float)(ssu - 1u);
    const float v = (float)(k / ssu) / (float)(ssv - 1u);

    __global const uchar *srec = surfaces + s*64u;
    __global const ushort *aidx = (__global const ushort *)srec;

    /* anchors: positions + displacement-direction normals */
    float3 A0 = anchor_pos(anchors, aidx[0], bbmin, bbspan);
    float3 A1 = anchor_pos(anchors, aidx[1], bbmin, bbspan);
    float3 A2 = anchor_pos(anchors, aidx[2], bbmin, bbspan);
    float3 A3 = anchor_pos(anchors, aidx[3], bbmin, bbspan);
    float3 N0 = anchor_nrm(anchors, aidx[0]);
    float3 N1 = anchor_nrm(anchors, aidx[1]);
    float3 N2 = anchor_nrm(anchors, aidx[2]);
    float3 N3 = anchor_nrm(anchors, aidx[3]);

    /* bilinear base (uu along A0->A3 edge, vv along A0->A1 edge) */
    float3 pl = mix(A0, A3, u), pr = mix(A1, A2, u);
    float3 P  = mix(pl, pr, v);
    float3 nl = mix(N0, N3, u), nr = mix(N1, N2, u);
    float3 Nd = fast_normalize(mix(nl, nr, v));

    /* deformation + sewing along the SHARED normal field */
    float c[8];
    {   __global const float *cf = (__global const float *)(srec + 12u);
        for (int i = 0; i < 8; ++i) c[i] = cf[i];
    }
    float h = deform_h(c, u, v) + stitch_h(stitch, s, u, v, s_scale);
    P = fma((float3)(h, h, h), Nd, P);

    /* SMOOTH SHADING NORMAL ("check there once"): the host precomputes one
     * true geometric normal per anchor (PSFBody::ComputeSmoothNormals) --
     * averaged over every patch sharing the anchor -- in the parallel
     * `snorm` buffer (kept separate from the anchor record's normal slot,
     * which is the displacement direction Nd and differs by up to 90 deg).
     * Interpolating these 4 corner normals gives a C0-continuous geometric
     * normal across patch borders: no facets, no per-sample cross product,
     * no shading streaks. */
    float3 S0 = snorm[aidx[0]].xyz, S1 = snorm[aidx[1]].xyz;
    float3 S2 = snorm[aidx[2]].xyz, S3 = snorm[aidx[3]].xyz;
    float3 snl = mix(S0, S3, u), snr = mix(S1, S2, u);
    float3 Nshade = fast_normalize(mix(snl, snr, v));

    /* BACK-FACE CULL ("skip patches that shouldn't draw"): if this point's
     * geometric normal faces away from the camera, it's on the far side of
     * the object and must not be splatted -- otherwise back samples bleed
     * through the front wherever the z-buffer doesn't perfectly occlude
     * them (the tearing on the away-facing side). view = P - eye; a point
     * is back-facing when dot(N, view) > 0 (normal points downstream of the
     * view ray). A small bias keeps silhouette samples that are nearly
     * edge-on. */
    float3 view = fast_normalize(P - eye);
    if (dot(Nshade, view) > 0.10f) return;

    /* project */
    float4 hp = (float4)(
        dot(mvp.s0123, (float4)(P, 1.0f)),
        dot(mvp.s4567, (float4)(P, 1.0f)),
        dot(mvp.s89ab, (float4)(P, 1.0f)),
        dot(mvp.scdef, (float4)(P, 1.0f)));
    if (hp.w <= 1e-6f) return;
    float2 ndc = hp.xy / hp.w;
    if (fabs(ndc.x) > 1.0f || fabs(ndc.y) > 1.0f) return;
    uint px = (uint)((ndc.x * 0.5f + 0.5f) * (float)(width  - 1u) + 0.5f);
    uint py = (uint)((0.5f - ndc.y * 0.5f) * (float)(height - 1u) + 0.5f);
    uint pix = py * width + px;
    float depth = hp.z / hp.w;            /* in [0,1] after projection */
    if (depth < 0.0f) return;

    /* shade: mathematical color * lambert on the smooth geometric normal */
    float lam = clamp(dot(Nshade, -light_dir), 0.0f, 1.0f) * 0.85f + 0.15f;
    uint r = (uint)(channel_color(srec, 0u, u, v) * lam * 255.0f);
    uint g = (uint)(channel_color(srec, 1u, u, v) * lam * 255.0f);
    uint b = (uint)(channel_color(srec, 2u, u, v) * lam * 255.0f);
    uint rgba = (r << 24) | (g << 16) | (b << 8) | 255u;

#ifdef USE_LONG_ATOMICS
    ulong key = ((ulong)as_uint(depth) << 32) | (ulong)rgba;
    atom_min(&fb[pix], key);
#else
    if (pass == 0u) {
        atomic_min(&depth32[pix], as_uint(depth));
    } else {
        if (depth32[pix] == as_uint(depth))    /* my sample won pass 0 */
            fb[pix] = ((ulong)as_uint(depth) << 32) | (ulong)rgba;
    }
#endif
}

__kernel void resolve_fb(__global const ulong *fb, uint n,
                         __global uchar4 *image)
{
    uint g = get_global_id(0);
    if (g >= n) return;
    ulong key = fb[g];
    uint rgba = (uint)(key & 0xFFFFFFFFu);
    image[g] = (key == 0xFFFFFFFFFFFFFFFFUL)
             ? (uchar4)(18, 18, 24, 255)                 /* background */
             : (uchar4)(rgba >> 24, (rgba >> 16) & 255u,
                        (rgba >> 8) & 255u, 255u);
}
