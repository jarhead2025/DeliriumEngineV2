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

/* MODEL MATRIX (world transform): the 64-byte records are FROZEN and hold
 * the body in its OWN local space. A workspace needs to MOVE objects around
 * without re-quantizing (and losing precision on) every anchor, so the
 * placement lives OUTSIDE the records and is applied here, on the GPU.
 * `model` is a row-major 4x4 (local -> world). */
inline float3 xform_point(const float4 m0, const float4 m1,
                           const float4 m2, const float4 m3, float3 p)
{
    return (float3)(dot(m0, (float4)(p, 1.0f)),
                    dot(m1, (float4)(p, 1.0f)),
                    dot(m2, (float4)(p, 1.0f)));
}
inline float3 xform_dir(const float4 m0, const float4 m1,
                         const float4 m2, float3 v)
{
    return (float3)(dot(m0, (float4)(v, 0.0f)),
                    dot(m1, (float4)(v, 0.0f)),
                    dot(m2, (float4)(v, 0.0f)));
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
    const float4 model_r0, const float4 model_r1,
    const float4 model_r2, const float4 model_r3,
    const uint  width, const uint height,
    const float depth_bias,                /* NDC depth added to this body's
                                              samples (fracture fragment: pushes
                                              it behind its coplanar parent cut
                                              edge so atom_min never ties) */
    __global volatile ulong *fb,
    __global volatile uint  *tie_cnt      /* DEPTH-TIE DIAGNOSTIC: counts how
                                              many biased (fragment) samples land
                                              within 2e-5 NDC of the pixel's
                                              current winner. 0 = no coplanar
                                              contention. Only checked when
                                              depth_bias > 0, so normal bodies
                                              pay nothing. */
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

    /* WORLD TRANSFORM (local -> world). The position is transformed as a
     * POINT (translation applies); the shading normal is transformed as a
     * DIRECTION (rotation only) and renormalized (the rotation may carry
     * a scale, which changes the normal's length). The DISPLACEMENT normal
     * Nd is only used to displace along the LOCAL surface, so it must stay
     * in local space -- it is applied to P BEFORE this transform. */
    P = xform_point(model_r0, model_r1, model_r2, model_r3, P);
    Nshade = fast_normalize(xform_dir(model_r0, model_r1, model_r2, Nshade));

    /* BACK-FACE handling: FLIP the shading normal to face the camera instead
     * of culling the sample. The smooth-normal field (ComputeSmoothNormals)
     * is an area-weighted average over patches with mixed (u,v) winding, so
     * on object CENTERS it points INWARD. The old hard cull
     *     if (dot(Nshade, view) > 0.10f) return;
     * then deleted the front-center samples of EVERY object -> the centered
     * dark patches, objects vanishing/flickering as the camera orbits
     * (samples crossing the 0.10 threshold frame-to-frame), and per-object
     * inconsistent lighting. Flipping to face the camera (two-sided normal)
     * keeps the center lit; true back faces are hidden by the depth atom_min
     * anyway (the nearer front sample wins). This matches the march path
     * exactly (surfel_march.cl:300 does `if(dot(n,fwd)>0) n=-n;` then shades,
     * no cull). For a correct outward normal the flip is a no-op on visible
     * front samples, so it only changes the broken (inward) cases. */
    float3 view = fast_normalize(P - eye);
    if (dot(Nshade, view) > 0.f) Nshade = -Nshade;
    /* A view-grazing normal (abs(dot(N, view)) near zero) can flip the
     * two-sided key-light response sharply across neighboring splats.
     * Smooth ONLY the shading direction toward the viewer at grazing angles:
     * s=0 at |N.V|<=0.10, s=1 at >=0.35. Position, projection, depth,
     * visibility and the displacement normal Nd remain untouched. */
    const float facing = clamp(-dot(Nshade, view), 0.0f, 1.0f);
    const float t = clamp((facing - 0.10f) * 4.0f, 0.0f, 1.0f);
    const float grazing = 1.0f - t*t*(3.0f - 2.0f*t);
    Nshade = normalize(Nshade + (-view - Nshade) * (0.25f * grazing));

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
/* DEPTH-TIE DIAGNOSTIC (2026-10-29): a "tie" = a sample whose depth is
     * within 2e-5 NDC of the pixel's current winner. At a fracture cut the
     * parent's cut edge and the fragment's cut edge are COPLANAR -> exact
     * ties -> atom_min flips the winner frame-to-frame = the popping. A
     * healthy scene (no coplanar surfaces) should show ~0 ties. Only runs
     * for biased (fragment) samples, so normal bodies pay nothing. */
    if (depth_bias > 0.0f) {
        uint exd = (uint)(fb[pix] >> 32);
        /* exd is the winner's depth stored as float BITS; as_float() gives the
         * actual NDC depth. A tie = within 2e-5 NDC (~100-300x float epsilon
         * over [0,1]) of this sample's depth. */
        if (exd != 0xFFFFFFFFu && fabs(as_float(exd) - depth) < 2e-5f)
            atomic_add(tie_cnt, 1u);
    }
    depth += depth_bias;                  /* fragment bias: behind its parent */

    /* shade: mathematical color * lambert on the smooth geometric normal.
     * Two lights: the key (light_dir) sculpts form, and a CAMERA FILL that
     * lights whatever faces the viewer. Without the fill, the camera-facing
     * tube wall sits at the emissive floor (dot(N,key)<0) and reads as
     * "transparent" against the near-black background. view = normalize(P-eye)
     * is already computed above for the back-face cull, so the fill is free. */
    float keyL  = clamp(dot(Nshade, -light_dir), 0.0f, 1.0f);
    float fillL = clamp(dot(Nshade, -view),       0.0f, 1.0f);
    float lam   = clamp(keyL * 0.85f + fillL * 0.30f, 0.0f, 1.0f) + 0.15f;
    uint r = (uint)(channel_color(srec, 0u, u, v) * lam * 255.0f);
    uint g = (uint)(channel_color(srec, 1u, u, v) * lam * 255.0f);
    uint b = (uint)(channel_color(srec, 2u, u, v) * lam * 255.0f);
    uint rgba = (r << 24) | (g << 16) | (b << 8) | 255u;

#ifdef USE_LONG_ATOMICS
    ulong key = ((ulong)as_uint(depth) << 32) | (ulong)rgba;
    atom_min(&fb[pix], key);
#else
    if (pass == 0u) {
        /* Interleave with the march layer already in buf_fb (body 0 stores its
         * depth|rgba key there first): the winner is the NEAREST of {existing
         * buf_fb depth, this splat depth}. Without the min against fb, a
         * nearer march hit would be clobbered by this body's plain store. */
        uint exd = (uint)(fb[pix] >> 32);
        uint w   = (exd < as_uint(depth)) ? exd : as_uint(depth);
        atomic_min(&depth32[pix], w);
    } else {
        if (depth32[pix] == as_uint(depth))    /* my sample won pass 0 */
            fb[pix] = ((ulong)as_uint(depth) << 32) | (ulong)rgba;
    }
#endif
}

/* =========================================================================
 * AREA-WEIGHTED SPLAT (the pixel-footprint pyramid)
 * =========================================================================
 *
 * THE GEOMETRY THE USER DESCRIBED:
 *   "in a classical angular per pixel projection, the area of the pixel if
 *    it impacts is increasingly greater since essentially it is a 4 sided
 *    pyramid extending when it comes to the math, that when combined add
 *    up the image correct?"
 *
 * WHAT THAT MEANS, EXACTLY:
 *   A pixel subtends a FIXED ANGLE. At distance d that angle covers a
 *   world-space footprint that GROWS LINEARLY WITH d -- and on a surface
 *   tilted away from the viewer the footprint stretches further still
 *   (by 1/cos(theta)). The correct weight of a sample's contribution to
 *   a pixel is therefore the AREA that pixel covers on the surface, not
 *   a constant 1.
 *
 * WHY THE CLASSIC SPLAT IS WRONG WITHOUT IT:
 *   psf_splat lands every surface sample on EXACTLY ONE pixel
 *   (atom_min(&fb[pix], key)). The surface is sampled at PIXEL
 *   granularity, so two things follow:
 *     (a) a surface seen at a grazing angle -- where one pixel covers
 *         many samples' worth of area -- receives too FEW contributions,
 *         so it shows holes a head-on surface does not;
 *     (b) a surface seen head-on -- where many pixels cover the same
 *         surface area -- receives too MANY, so it looks denser than it
 *         is.
 *   The result is a VIEW-DEPENDENT DENSITY: the same geometry reads
 *   differently from different angles.
 *
 * THE FIX: distribute each sample over the pixel footprint it covers.
 *   The screen-space size of the footprint is
 *       footprint_px = (2*tan(fov/2)*depth / screen_height) * (1/cos)
 *   The footprint is a QUADRANGLE (the user's "4 sided pyramid"); for a
 *   splat it is approximated with a separable BOX -- the sample
 *   contributes to the pixels in a (2*rx+1) x (2*ry+1) neighbourhood
 *   around its projected centre, with a tent (bilinear) falloff.
 *
 * WHY THE WEIGHTS ARE INTEGERS (the user's "simple math operations could
 * help average out where the floats fail, like a floor function"):
 *   The accumulator sums FIXED-POINT INTEGERS and divides with an
 *   INTEGER divide at the end:
 *     - OpenCL C 1.2 has no float atomic_add, so the sum must be an
 *       integer type anyway;
 *     - an integer sum is EXACT -- no accumulation order can perturb it,
 *       so the render stays DETERMINISTIC across runs;
 *     - the integer divide TRUNCATES, biasing dark by < 1/4096
 *       (invisible) while avoiding a float divide that would reintroduce
 *       non-determinism.
 *
 * WHY THE DEPTH STILL GOES IN THE KEY (unchanged):
 *   The area weighting decides HOW MUCH a sample contributes to a pixel.
 *   It does NOT decide WHICH surface is visible there -- that remains the
 *   nearest-wins depth test. Every pixel in a sample's footprint carries
 *   the SAMPLE's depth, so the nearest sample that touched a pixel still
 *   wins it.
 * ========================================================================= */

/* The pixel footprint at a given depth, measured in pixels.
 *
 *   world_per_px  = 2 * tan(fov/2) * depth / screen_height
 *   footprint_px  = world_per_px * (1 / cos_theta)
 *
 * `cos_theta` is the cosine between the surface normal and the view
 * direction. A head-on surface (cos=1) has the smallest footprint; a
 * grazing surface (cos->0) has an unbounded one, which is CLAMPED so a
 * near-tangent sample cannot smear across the whole screen.
 *
 * NOTE: the footprint is expressed in PIXELS, and `world_per_px` already
 * includes the depth, so the result grows linearly with distance -- which
 * is the "increasingly greater" area the user described.
 */
inline float footprint_pixels(float depth, float tan_half, uint screen_h,
                              float cos_theta)
{
    if (depth <= 1e-4f) return 1.0f;
    const float world_per_px = (2.0f * tan_half * depth) / (float)screen_h;
    /* 1/cos stretches the footprint on a tilted surface. The 0.25 floor
     * bounds it to 4x -- generous enough to fill grazing-angle gaps,
     * small enough to keep the splat local. */
    const float stretch = 1.0f / max(cos_theta, 0.25f);
    /* At least 1 pixel (the sample must land somewhere), at most 16. */
    return clamp(world_per_px * stretch, 1.0f, 16.0f);
}

/* ONE WORK-ITEM = ONE SURFACE SAMPLE.
 *
 * Writes into per-pixel AREA accumulators instead of the shared
 * framebuffer:
 *   area_csum[slot*3 + 0..2]  -- the colour sums (fixed point, 4096)
 *   area_wsum[slot]           -- the accumulated area weight
 *   area_key[slot]            -- the NEAREST depth seen (float bits)
 *
 * The AREA accumulators are what the user asked for: the impact is
 * distributed over the pixels its footprint covers, weighted by area, and
 * the pixel's colour is the AREA-WEIGHTED MEAN of everything that
 * touched it.
 *
 * This kernel writes to its OWN buffers; a separate resolve kernel
 * (`area_resolve`) turns them into the image. The classic and layered
 * paths are untouched.
 */
__kernel void psf_splat_area(
    __global const uchar  *anchors,
    __global const float4 *snorm,
    __global const uchar  *surfaces,
    __global const char   *stitch,
    __global const uint   *sample_off,
    __global const uint   *sample_ss,
    const uint  n_surfaces,
    const float3 bbmin, const float3 bbspan,
    const float  s_scale,
    const float16 mvp,
    const float3 light_dir,
    const float3 eye,
    const float3 view_fwd,
    const float4 model_r0, const float4 model_r1,
    const float4 model_r2, const float4 model_r3,
    const uint  width, const uint height,
    const float depth_bias,
    const float tan_half,       /* tan(fov/2) -- the pixel footprint's scale */
    __global uint  *area_csum,  /* 3 uints per pixel: the colour sums */
    __global uint  *area_wsum,  /* 1 uint per pixel: the area weight */
    __global volatile ulong *area_key /* (depth_bits << 32) per pixel */
)
{
    uint gid = get_global_id(0);
    if (gid >= sample_off[n_surfaces]) return;

    uint lo = 0, hi = n_surfaces;
    while (hi - lo > 1u) {
        uint mid = (lo + hi) >> 1;
        if (sample_off[mid] <= gid) lo = mid; else hi = mid;
    }
    const uint s   = lo;
    const uint k   = gid - sample_off[s];
    const uint ssu = sample_ss[s] >> 16;
    const uint ssv = sample_ss[s] & 0xFFFFu;
    const float u = (float)(k % ssu) / (float)(ssu - 1u);
    const float v = (float)(k / ssu) / (float)(ssv - 1u);

    __global const uchar *srec = surfaces + s*64u;
    __global const ushort *aidx = (__global const ushort *)srec;

    float3 A0 = anchor_pos(anchors, aidx[0], bbmin, bbspan);
    float3 A1 = anchor_pos(anchors, aidx[1], bbmin, bbspan);
    float3 A2 = anchor_pos(anchors, aidx[2], bbmin, bbspan);
    float3 A3 = anchor_pos(anchors, aidx[3], bbmin, bbspan);
    float3 N0 = anchor_nrm(anchors, aidx[0]);
    float3 N1 = anchor_nrm(anchors, aidx[1]);
    float3 N2 = anchor_nrm(anchors, aidx[2]);
    float3 N3 = anchor_nrm(anchors, aidx[3]);

    float3 pl = mix(A0, A3, u), pr = mix(A1, A2, u);
    float3 P  = mix(pl, pr, v);
    float3 nl = mix(N0, N3, u), nr = mix(N1, N2, u);
    float3 Nd = fast_normalize(mix(nl, nr, v));

    float c[8];
    {   __global const float *cf = (__global const float *)(srec + 12u);
        for (int i = 0; i < 8; ++i) c[i] = cf[i];
    }
    float h = deform_h(c, u, v) + stitch_h(stitch, s, u, v, s_scale);
    P = fma((float3)(h, h, h), Nd, P);

    float3 S0 = snorm[aidx[0]].xyz, S1 = snorm[aidx[1]].xyz;
    float3 S2 = snorm[aidx[2]].xyz, S3 = snorm[aidx[3]].xyz;
    float3 snl = mix(S0, S3, u), snr = mix(S1, S2, u);
    float3 Nshade = fast_normalize(mix(snl, snr, v));

    P = xform_point(model_r0, model_r1, model_r2, model_r3, P);
    Nshade = fast_normalize(xform_dir(model_r0, model_r1, model_r2, Nshade));

    float3 view = fast_normalize(P - eye);
    if (dot(Nshade, view) > 0.f) Nshade = -Nshade;
    const float facing = clamp(-dot(Nshade, view), 0.0f, 1.0f);
    const float tg = clamp((facing - 0.10f) * 4.0f, 0.0f, 1.0f);
    const float grazing = 1.0f - tg*tg*(3.0f - 2.0f*tg);
    Nshade = normalize(Nshade + (-view - Nshade) * (0.25f * grazing));

    float4 hp = (float4)(
        dot(mvp.s0123, (float4)(P, 1.0f)),
        dot(mvp.s4567, (float4)(P, 1.0f)),
        dot(mvp.s89ab, (float4)(P, 1.0f)),
        dot(mvp.scdef, (float4)(P, 1.0f)));
    if (hp.w <= 1e-6f) return;
    float2 ndc = hp.xy / hp.w;
    if (fabs(ndc.x) > 1.0f || fabs(ndc.y) > 1.0f) return;
    /* The fractional pixel position -- the footprint's centre is
     * SUB-PIXEL accurate, so the tent weights are symmetric around it. */
    const float fx = (ndc.x * 0.5f + 0.5f) * (float)(width  - 1u);
    const float fy = (0.5f - ndc.y * 0.5f) * (float)(height - 1u);
    const uint px = (uint)(fx + 0.5f);
    const uint py = (uint)(fy + 0.5f);
    float depth = hp.z / hp.w;
    if (depth < 0.0f) return;

    /* shade (identical to the classic path) */
    float keyL  = clamp(dot(Nshade, -light_dir), 0.0f, 1.0f);
    float fillL = clamp(dot(Nshade, -view),       0.0f, 1.0f);
    float lam   = clamp(keyL * 0.85f + fillL * 0.30f, 0.0f, 1.0f) + 0.15f;
    const uint cr = (uint)(channel_color(srec, 0u, u, v) * lam * 255.0f);
    const uint cg = (uint)(channel_color(srec, 1u, u, v) * lam * 255.0f);
    const uint cb = (uint)(channel_color(srec, 2u, u, v) * lam * 255.0f);

    /* THE FOOTPRINT. The view-space depth (not the NDC depth) sets the
     * world size of a pixel, so it is computed from the view axis. */
    const float view_depth = dot(P - eye, view_fwd);
    /* The cosine between the surface and the view: a head-on surface has
     * the smallest footprint, a grazing one the largest.
     *
     * NOTE: `fabs` not `abs` -- OpenCL C 1.2's `abs` is ambiguous for a
     * float argument (it resolves to the integer overload), and the
     * compiler rejects it outright. */
    const float cos_theta = clamp(fabs(dot(Nshade, view)), 0.05f, 1.0f);
    const float fp = footprint_pixels(view_depth, tan_half, height, cos_theta);
    /* The footprint's half-extent in pixels. A footprint of 1.0 px still
     * contributes to its own pixel (weight 1); 3.0 px spreads over a
     * 3x3 block with a tent falloff. */
    const int rx = (int)((fp - 1.0f) * 0.5f + 0.5f);
    const int ry = (int)((fp - 1.0f) * 0.5f + 0.5f);

    depth += depth_bias;

    /* DISTRIBUTE over the footprint.
     *
     * The tent weight is (1 - |dx|/(rx+1)) * (1 - |dy|/(ry+1)),
     * quantised to a fixed-point integer in [0, 4096]. The quantisation
     * is what makes the sum exact and the result deterministic; the
     * user's "floor function" instinct applied to the weights.
     */
    for (int dy = -ry; dy <= ry; ++dy) {
        const int qy = (int)py + dy;
        if (qy < 0 || qy >= (int)height) continue;
        const float ty = (ry + 1 > 0) ? (1.0f - fabs((float)dy) / (float)(ry + 1)) : 1.0f;
        for (int dx = -rx; dx <= rx; ++dx) {
            const int qx = (int)px + dx;
            if (qx < 0 || qx >= (int)width) continue;
            const float tx = (rx + 1 > 0) ? (1.0f - fabs((float)dx) / (float)(rx + 1)) : 1.0f;
            const float tent = tx * ty;
            /* the area weight, quantised to a fixed-point integer. The
             * floor is deliberate: it never over-weights, so the sum of
             * the weights is <= the true area. */
            const uint aw = (uint)(tent * 4096.0f);
            if (aw == 0u) continue;
            const uint slot = (uint)qy * width + (uint)qx;

            /* THE DEPTH WINNER: the nearest sample that touched this
             * pixel owns it. The area weights do NOT change this. */
            const ulong key = ((ulong)as_uint(depth) << 32) | 0xFFFFFFFFu;
            atom_min(&area_key[slot], key);

            /* THE AREA-WEIGHTED COLOUR. The colour sums are accumulated
             * with the area weight, so the resolve divides by the total
             * area weight and gets the AREA-WEIGHTED MEAN. */
            atomic_add(&area_csum[slot*3u + 0u], cr * aw);
            atomic_add(&area_csum[slot*3u + 1u], cg * aw);
            atomic_add(&area_csum[slot*3u + 2u], cb * aw);
            atomic_add(&area_wsum[slot], aw);
        }
    }
}

/* Turn the area accumulators into the framebuffer.
 *
 * The colour is (colour_sum / area_weight) -- an INTEGER divide, so it
 * truncates and is deterministic.
 *
 * WHY IT WRITES THE SHARED FRAMEBUFFER (not buf_img directly):
 *   The shared framebuffer is the multi-body compositing surface: every
 *   body writes its pixels there and the nearest wins (atom_min on the
 *   (depth|rgba) key), and the final resolve turns it into the image
 *   once, after every body has contributed. Writing the area result
 *   straight into buf_img would bypass that and make the area path
 *   single-body only -- the LAST body would overwrite the others.
 *
 *   So the area path resolves ONE BODY's area-weighted mean into a
 *   (depth|rgba) key and lets the shared framebuffer do the compositing
 *   exactly as the classic path does. The per-body area accumulators are
 *   reset for each body, which is correct: the area weighting is a
 *   WITHIN-body operation (it answers "how much of THIS surface covers
 *   THIS pixel"), and the depth test answers "which body wins".
 *
 * A pixel with NO area weight is background: it is simply not written,
 * so the framebuffer keeps whatever was there (another body, or the
 * clear state). */
__kernel void area_resolve(
    __global const uint *area_csum,
    __global const uint *area_wsum,
    __global const ulong *area_key,
    const uint W, const uint H,
    __global volatile ulong *fb)
{
    uint g = get_global_id(0);
    if (g >= W * H) return;
    const uint w = area_wsum[g];
    if (w == 0u) return;                  // this body did not touch it
    const ulong key = area_key[g];
    const uint depth_bits = (uint)(key >> 32);
    if (depth_bits == 0xFFFFFFFFu) return; // no depth winner -- impossible
    const uint r  = area_csum[g*3u + 0u] / w;
    const uint gc = area_csum[g*3u + 1u] / w;
    const uint b  = area_csum[g*3u + 2u] / w;
    const uint rgba = (min(r, 255u) << 24) | (min(gc, 255u) << 16)
                    | (min(b, 255u) << 8) | 255u;
    /* The nearest-wins composite, exactly as the classic path does. */
    atom_min(&fb[g], ((ulong)depth_bits << 32) | (ulong)rgba);
}

/* =========================================================================
 * LAYERED DISTANCE-PLANE SPLAT (the user's method, for the SPLAT path).
 *
 * WHAT THE USER DESCRIBED:
 *   "you could average them out when intersecting them with variable
 *    distance planes from the camera's origin, with a normal vector
 *    parallel to the camera's direction, the original image buffer would
 *    be a skybox one and these would stack back to front overwriting each
 *    other in that sequence, after the last one the image acquired would
 *    only then be shown"
 *
 * WHY IT MATTERS FOR THE SPLAT PATH:
 *   The splat path (render path 3) is the one with "smaller but more
 *   numerous pixel impacts": each surface sample is a small splat that
 *   lands on ONE pixel via an atomic_min. Two samples of the SAME surface
 *   often land on ADJACENT pixels (the grid is anisotropic and the
 *   projection is non-uniform), so the surface reads as a dithered,
 *   patchy skin rather than a continuous one.
 *
 *   The fix is to bucket the impacts by DISTANCE PLANE before they are
 *   composited:
 *
 *     1. A plane is the FOV square at depth d from the camera origin.
 *        Its normal is parallel to the camera's direction BY
 *        CONSTRUCTION -- the plane IS the distance shell.
 *     2. Each plane (a "band") accumulates its impacts into its OWN
 *        buffer, so an impact's neighbours are the OTHER impacts of the
 *        SAME distance shell, not a smear across depth.
 *     3. Within a band, the impacts are AVERAGED (the user's "average
 *        them out"): a hole between two impacts takes their mean
 *        colour, so the dither dissolves into a continuous surface.
 *     4. The bands are stacked BACK TO FRONT (far first, near last),
 *        each overwriting what is behind it, so the nearest wins.
 *     5. Only after the LAST band is the image shown.
 *
 *   The stacking uses the SAME (depth|rgba) atom_min key the classic
 *   splat path uses, so a layered body and a classic body interleave by
 *   depth identically -- and the fracture fragment's depth bias keeps
 *   working, because the bias is applied to the band depth exactly where
 *   the classic path applies it to the sample depth.
 *
 *   This is the golden reference's algorithm
 *   (registry/golden/kernels/surfel_march.cl: layer_draw + layer_fill +
 *   layer_compose), re-expressed for the splat's per-sample workload.
 * ========================================================================= */

/* The band a sample belongs to: floor((depth - d_near) / band_span).
 *
 * A sample OUTSIDE [d_near, d_far] is CLAMPED to the nearest band, never
 * dropped. Dropping it (the old `return -1` behaviour) silently deleted
 * every sample whose displaced position fell outside the host's band
 * range -- and because the host derived that range from the ANCHOR
 * positions while the rendered sample is DISPLACED (P + h*Nd), a curved
 * body lost a whole shell of surface at its extremes. That was the
 * layered path's extra interior holes (~450 lost lit pixels on the
 * gallery's classic-splat scene).
 *
 * Clamping keeps the sample: it lands in the nearest shell, and the
 * shell's own depth test still resolves it correctly against everything
 * else, so the geometry is never lost -- only its shell assignment is
 * approximate, and the shell is an averaging group, not a depth test. */
inline int band_of(float depth, const float d_near, const float d_far,
                   const int nbands, const float band_span)
{
    int b = (int)((depth - d_near) / band_span);
    if (b < 0) b = 0;
    if (b >= nbands) b = nbands - 1;
    return b;
}

/* ONE WORK-ITEM = ONE SURFACE SAMPLE (same addressing as psf_splat).
 *
 * Writes into the band buffers instead of the shared framebuffer:
 *   band_color[b*W*H + pix]  -- the accumulated colour of band b
 *   band_wsum[b*W*H + pix]   -- the accumulated weight of band b
 *   band_depth[b*W*H + pix]  -- the NEAREST depth seen in band b
 *
 * The weights are summed with an atomic so several samples landing on
 * the same pixel of the same band combine correctly; the colour is
 * accumulated the same way. */
__kernel void psf_splat_layered(
    __global const uchar  *anchors,
    __global const float4 *snorm,
    __global const uchar  *surfaces,
    __global const char   *stitch,
    __global const uint   *sample_off,
    __global const uint   *sample_ss,
    const uint  n_surfaces,
    const float3 bbmin, const float3 bbspan,
    const float  s_scale,
    const float16 mvp,
    const float3 light_dir,
    const float3 eye,
    const float3 view_fwd,     /* the camera's forward axis (the distance
                                  plane's normal) */
    const float4 model_r0, const float4 model_r1,
    const float4 model_r2, const float4 model_r3,
    const uint  width, const uint height,
    const float depth_bias,
    const float d_near, const float d_far,
    const int   nbands,
    __global volatile ulong *band_key /* (depth_bits << 32) | rgba per slot */
)
{
    uint gid = get_global_id(0);
    if (gid >= sample_off[n_surfaces]) return;

    uint lo = 0, hi = n_surfaces;
    while (hi - lo > 1u) {
        uint mid = (lo + hi) >> 1;
        if (sample_off[mid] <= gid) lo = mid; else hi = mid;
    }
    const uint s   = lo;
    const uint k   = gid - sample_off[s];
    const uint ssu = sample_ss[s] >> 16;
    const uint ssv = sample_ss[s] & 0xFFFFu;
    const float u = (float)(k % ssu) / (float)(ssu - 1u);
    const float v = (float)(k / ssu) / (float)(ssv - 1u);

    __global const uchar *srec = surfaces + s*64u;
    __global const ushort *aidx = (__global const ushort *)srec;

    float3 A0 = anchor_pos(anchors, aidx[0], bbmin, bbspan);
    float3 A1 = anchor_pos(anchors, aidx[1], bbmin, bbspan);
    float3 A2 = anchor_pos(anchors, aidx[2], bbmin, bbspan);
    float3 A3 = anchor_pos(anchors, aidx[3], bbmin, bbspan);
    float3 N0 = anchor_nrm(anchors, aidx[0]);
    float3 N1 = anchor_nrm(anchors, aidx[1]);
    float3 N2 = anchor_nrm(anchors, aidx[2]);
    float3 N3 = anchor_nrm(anchors, aidx[3]);

    float3 pl = mix(A0, A3, u), pr = mix(A1, A2, u);
    float3 P  = mix(pl, pr, v);
    float3 nl = mix(N0, N3, u), nr = mix(N1, N2, u);
    float3 Nd = fast_normalize(mix(nl, nr, v));

    float c[8];
    {   __global const float *cf = (__global const float *)(srec + 12u);
        for (int i = 0; i < 8; ++i) c[i] = cf[i];
    }
    float h = deform_h(c, u, v) + stitch_h(stitch, s, u, v, s_scale);
    P = fma((float3)(h, h, h), Nd, P);

    float3 S0 = snorm[aidx[0]].xyz, S1 = snorm[aidx[1]].xyz;
    float3 S2 = snorm[aidx[2]].xyz, S3 = snorm[aidx[3]].xyz;
    float3 snl = mix(S0, S3, u), snr = mix(S1, S2, u);
    float3 Nshade = fast_normalize(mix(snl, snr, v));

    P = xform_point(model_r0, model_r1, model_r2, model_r3, P);
    Nshade = fast_normalize(xform_dir(model_r0, model_r1, model_r2, Nshade));

    float3 view = fast_normalize(P - eye);
    if (dot(Nshade, view) > 0.f) Nshade = -Nshade;
    const float facing = clamp(-dot(Nshade, view), 0.0f, 1.0f);
    const float tg = clamp((facing - 0.10f) * 4.0f, 0.0f, 1.0f);
    const float grazing = 1.0f - tg*tg*(3.0f - 2.0f*tg);
    Nshade = normalize(Nshade + (-view - Nshade) * (0.25f * grazing));

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
    float depth = hp.z / hp.w;
    if (depth < 0.0f) return;

    /* THE BAND: the impact's distance plane from the camera origin.
     *
     * CRITICAL: the distance plane is the plane PERPENDICULAR to the
     * camera's direction (its normal is PARALLEL to the camera's
     * direction, as the user specified). The signed distance of a point
     * from that plane is its DEPTH ALONG THE VIEW AXIS -- the projection
     * of (P - eye) onto the camera's forward vector -- NOT its radial
     * distance from the eye.
     *
     * Using the radial distance (length(P - eye)) put samples of very
     * different depths into the SAME band: a point 1 unit to the side
     * and a point 1 unit ahead have the same radius but depths that
     * differ by the FOV angle. The band fill then averaged across the
     * whole surface -- every colour mixed into every other (a rainbow
     * torus averaged to white) and the fill smeared the silhouette.
     *
     * The view axis is fwd = normalize(target - eye). The splat kernel
     * does not receive it, so the host passes it: `view_fwd` below.
     */
    const float view_depth = dot(P - eye, view_fwd);
    const float band_span = (d_far - d_near) / (float)nbands;
    if (band_span <= 0.0f) return;
    /* band_of CLAMPS to the nearest band -- it never returns -1, so no
     * sample is ever dropped for being outside the host's range. */
    const int band = band_of(view_depth, d_near, d_far, nbands, band_span);

    depth += depth_bias;

    float keyL  = clamp(dot(Nshade, -light_dir), 0.0f, 1.0f);
    float fillL = clamp(dot(Nshade, -view),       0.0f, 1.0f);
    float lam   = clamp(keyL * 0.85f + fillL * 0.30f, 0.0f, 1.0f) + 0.15f;
    float3 rgb = (float3)(
        channel_color(srec, 0u, u, v) * lam,
        channel_color(srec, 1u, u, v) * lam,
        channel_color(srec, 2u, u, v) * lam);

    /* Accumulate into the band.
     *
     * THE BAND KEEPS THE NEAREST IMPACT'S COLOUR, NOT THE AVERAGE OF ALL
     * THE IMPACTS IN IT.
     *
     * The golden reference (registry/golden/kernels/surfel_march.cl) does
     * exactly this: layer_draw puts the band hit's colour into the
     * layer's buffer, and layer_fill only fills the pixels that have NO
     * hit. Each pixel therefore ends up with ONE surface's colour, and
     * the averaging happens only in the GAPS, between pixels of the SAME
     * surface.
     *
     * Accumulating an AVERAGE was wrong because a band is a DEPTH SHELL,
     * not a surface: at a grazing angle a torus's near rim and far rim
     * project to the same pixel but sit at very different depths, and
     * they land in the same band. Averaging them mixed a bright near rim
     * with a dark far rim -> the radial "spoke" streaks radiating from
     * the silhouette (measured: the streaks are strong at 16 bands and
     * nearly gone at 64, because thinner shells mix fewer depths), and
     * it also darkened pixels below the "lit" threshold, which read as
     * extra holes.
     *
     * So: the colour slot holds the NEAREST impact's colour (the depth
     * is compared atomically and the colour is written only by the
     * winner), and the weight slot counts how many impacts landed there
     * -- used by the fill to weight the neighbour average.
     *
     * The write is a two-step "claim the slot" protocol:
     *   1. atomic_min on the depth slot. If OUR depth is not the
     *      minimum, we lost -> we do not touch the colour.
     *   2. if we won, write our colour. A tie (two samples at exactly
     *      the same depth) is a benign race: both write the same
     *      surface's colour.
     *
     * The colour buffer is 4 uints per slot: r, g, b (fixed point,
     * scale 4096) and the impact count. */
    const uint bslot = (uint)band * (uint)(width * height) + pix;
    /* THE BAND SLOT IS A SINGLE 64-BIT KEY: (depth_bits << 32) | rgba.
     *
     * WHY ONE KEY AND NOT SEPARATE DEPTH + COLOUR SLOTS:
     * the nearest-wins protocol needs the depth comparison AND the colour
     * write to be ATOMIC WITH EACH OTHER. With separate slots the
     * sequence is:
     *     atomic_min(depth_slot, my_depth)   -> "I won"
     *     colour_slot = my_colour             -> (may be preempted here!)
     * A nearer impact can claim the slot and write its colour in
     * between, and the preempted writer then overwrites the NEARER
     * impact's colour with its own (farther) colour. That race makes the
     * render NON-DETERMINISTIC: five runs of the same scene produced five
     * different images.
     *
     * Packing the depth and the colour into ONE 64-bit key makes the
     * claim and the colour a SINGLE atomic operation: atom_min keeps
     * the whole key, so the winner's depth AND colour are installed
     * together. This is the same trick the framebuffer itself uses
     * (key = (depth_bits << 32) | rgba, atom_min, nearest wins).
     *
     * The tie-break is the rgba: two impacts at exactly the same depth
     * are compared by their colour, and the smaller rgba wins -- a
     * total order, so the result is deterministic regardless of the
     * order the impacts arrive in.
     *
     * The empty slot is the sentinel 0xFFFFFFFFFFFFFFFF (depth bits all
     * ones = +inf). Every real key is smaller, so a plain atomic_min
     * against the sentinel works.
     */
    uint r = (uint)(clamp(rgb.x, 0.0f, 1.0f) * 255.0f);
    uint g = (uint)(clamp(rgb.y, 0.0f, 1.0f) * 255.0f);
    uint b = (uint)(clamp(rgb.z, 0.0f, 1.0f) * 255.0f);
    uint rgba = (r << 24) | (g << 16) | (b << 8) | 255u;
    const ulong key = ((ulong)as_uint(depth) << 32) | (ulong)rgba;
    __global volatile ulong *bk = band_key + bslot;
    /* atom_min (NOT atomic_min): OpenCL C 1.2's atomic_min is only
     * defined for 32-bit int/uint, so atomic_min on a ulong is an
     * ambiguous overload. atom_min is the 64-bit variant, enabled by
     * cl_khr_int64_extended_atomics (the same extension the framebuffer
     * already relies on). The device reports 64-bit atomics: yes, and
     * the host passes -DUSE_LONG_ATOMICS when it does. */
    atom_min(bk, key);
}

/* Fill the holes WITHIN one band: an empty pixel takes the average of
 * the filled neighbours IN THAT BAND ONLY. This is the structural
 * difference from the classic path: the fill never smears across depth,
 * because a neighbour belongs to the same distance shell.
 *
 * PING-PONG IS REQUIRED: the fill MUST NOT write into the buffer it is
 * reading from. An in-place fill (cin == cout) is a DATA RACE: a
 * work-item's neighbours may have been overwritten by other work-items
 * in the same dispatch, so the result depends on the GPU's scheduling.
 * Measured: three consecutive renders of the same scene produced three
 * DIFFERENT images (three distinct md5s) with the in-place fill, while
 * the classic path was bit-stable across runs. The host now passes two
 * distinct buffers and swaps them between passes.
 *
 * The read side uses the HIT mask of the SOURCE buffer (hit_in) and
 * writes the hit mask of the DESTINATION (hit_out); a filled pixel's
 * colour is written into the destination only, so the next pass sees it
 * as a hit and propagates further. */
__kernel void layer_fill_band(
    __global const ulong *kin,      /* the band's (depth|rgba) keys */
    __global ulong *kout,
    const uint W, const uint H,
    const uint slot,               /* band base offset into the flat arrays */
    const float INF)
{
    uint x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    uint pid = y*W + x;
    uint idx = slot + pid;
    kout[idx] = kin[idx];
    if (kin[idx] != 0xFFFFFFFFFFFFFFFFUL) return;   // already has an impact
    if (x == 0 || x == W-1 || y == 0 || y == H-1) return;

    /* THE FILL: an empty pixel takes the AVERAGE of its filled
     * neighbours IN THAT BAND ONLY.
     *
     * This is the structural difference from the classic path: the fill
     * never smears across depth, because a neighbour belongs to the same
     * distance shell.
     *
     * The neighbour's depth is the band's own minimum for that pixel
     * (the key's high 32 bits), so the filled pixel inherits the NEAREST
     * neighbour's depth -- it sits at the surface, not at the band's
     * centre. Its colour is the average of the neighbours' colours, but
     * the DEPTH it stores is the nearest one, so the fill never pulls a
     * pixel in front of the real surface.
     *
     * THE NEIGHBOUR COUNT THRESHOLD: the golden reference
     * (registry/golden/kernels/surfel_march.cl, layer_fill) requires
     * cnt>=2 filled neighbours -- "empty pixel with >=3 filled neighbors
     * takes their average color" in the comment, >=2 in the code -- so a
     * pixel with only ONE filled neighbour is NOT filled. That is what
     * keeps the fill from bleeding outward past the true silhouette: a
     * 1-neighbour fill dilates the shape by a pixel in every direction,
     * which showed up as a stair-stepped, thickened rim.
     */
    uint R = 0u, G = 0u, B = 0u;
    uint dmin = 0xFFFFFFFFu;
    int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        uint n = slot + (y+dy)*W + (x+dx);
        ulong nk = kin[n];
        if (nk == 0xFFFFFFFFFFFFFFFFUL) continue;
        uint nrgba = (uint)(nk & 0xFFFFFFFFu);
        R += (nrgba >> 24) & 255u;
        G += (nrgba >> 16) & 255u;
        B += (nrgba >>  8) & 255u;
        uint d = (uint)(nk >> 32);
        if (d < dmin) dmin = d;
        cnt++;
    }
    /* >=2 filled neighbours (the golden's rule): a lone neighbour is not
     * enough to fill, so the fill cannot dilute the silhouette. */
    if (cnt >= 2) {
        uint rgba = (((R / (uint)cnt) & 255u) << 24)
                  | (((G / (uint)cnt) & 255u) << 16)
                  | (((B / (uint)cnt) & 255u) << 8)
                  | 255u;
        kout[idx] = ((ulong)dmin << 32) | (ulong)rgba;
    }
}

/* Stack the bands BACK TO FRONT into the shared framebuffer: the NEAREST
 * band wins per pixel (atom_min on the (depth|rgba) key), exactly the
 * "each plane overwrites the one behind it, far to near" rule. */
__kernel void layer_stack_band(
    __global const ulong *keys,     /* the band's (depth|rgba) keys */
    const uint W, const uint H,
    const uint slot,
    const float INF,
    __global volatile ulong *fb,
    __global volatile uint *depth32)
{
    uint x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    uint pid = y*W + x;
    uint idx = slot + pid;
    ulong key = keys[idx];
    if (key == 0xFFFFFFFFFFFFFFFFUL) return;   // this band had nothing here
    float d = as_float((uint)(key >> 32));
    if (d >= INF) return;
#ifdef USE_LONG_ATOMICS
    atom_min(&fb[pid], key);
#else
    uint exd = (uint)(fb[pid] >> 32);
    uint wmin = (exd < (uint)(key >> 32)) ? exd : (uint)(key >> 32);
    atomic_min(&depth32[pid], wmin);
    if (depth32[pid] == (uint)(key >> 32))
        fb[pid] = key;
#endif
}

__kernel void resolve_fb(__global const ulong *fb, uint n,
                         __global uchar4 *image)
{
    uint g = get_global_id(0);
    if (g >= n) return;
    ulong key = fb[g];
    uint rgba = (uint)(key & 0xFFFFFFFFu);
    /* ALPHA IS THE OBJECT/EMPTY MASK for the post passes (infill +
     * lod-stitch). The old code wrote alpha=255 for EVERY pixel, so the
     * lod-stitch object mask (s.w == oid) saw the background as the same
     * "object" as the surface and blurred colors across object boundaries
     * and into the background -> the "dissolving into noisy dots" grain.
     * Background now carries alpha=0 (empty), hits carry alpha=255. The
     * SDL blit / PPM / PNG readers use RGB only, so alpha is invisible to
     * them; only the GPU post passes read it. */
    image[g] = (key == 0xFFFFFFFFFFFFFFFFUL)
             ? (uchar4)(18, 18, 24, 0)                   /* background: empty */
             : (uchar4)(rgba >> 24, (rgba >> 16) & 255u,
                        (rgba >> 8) & 255u, 255u);       /* hit: opaque */
}
