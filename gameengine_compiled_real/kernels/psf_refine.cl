/* =========================================================================
 * psf_refine.cl - GPU-side live subdivision.
 * Splits flagged patches into 4 children entirely on-device:
 *   1. atom_add on alloc_heads reserves 5 anchors + 3 new surfaces
 *      (the parent slot is REUSED by child 0, so only 3 are appended --
 *       same "blob never has retired holes" rule as the CPU SplitPatch)
 *   2. the 5 hanging anchors (4 edge midpoints + center) are evaluated
 *      ON the parent equation (exact snap, the same rule
 *      tools/psf_encode.py / CPU SplitPatch uses) and quantized into
 *      fresh AnchorRecords (pos uint16 over bbox, normal uint8)
 *   3. each child's 8 coefficients come from a least-squares fit of the
 *      corner-vanishing basis against the parent evaluation on a 9x9
 *      grid inside the quadrant (G=9, same as the CPU SplitPatch; the
 *      basis restriction is exact for bilinear+deform terms, so the fit
 *      recovers them up to LSQ noise)
 *   4. the host rebuilds the seam graph around the 4 child ids
 *      [pid, s0, s0+1, s0+2] (seam restitching is a graph operation
 *      and stays on the CPU)
 *
 * LIMITS vs the CPU SplitPatch (documented, intentional):
 *   - no find_existing_mid: adjacent splits that share an edge each
 *     create their own midpoint anchor (a duplicate at the same point).
 *     The host's seam-graph rebuild + anchor merge pass cleans those.
 *   - the 5 new anchors inherit the parent's material/anim bytes and
 *     get zeroed connectivity/curve_params (hanging anchors).
 *
 * RECORD LAYOUTS (Types.h / psf_splat.cl, byte-for-byte):
 *   SurfaceRecord: 0: ushort anchor_idx[4] | 8: u32 surface_def
 *                  12: float coef[8] | 44: color_pattern[18] | 62: pad[2]
 *   AnchorRecord : 0: u32 bone | 4: u16 conn[4] | 12: u8 ctype[4]
 *                  16: i8 curve_params[16] | 32: u16 pos[3]
 *                  38: i16 delta[3] | 44: u8 normal[3] | 47: u8 tangent[4]
 *                  51: u8 color[3] | 54..63: material/anim/curv/reserved
 *   stitch: 8 int8 per surface, edges v0,v1,u0,u1 x (a,b)
 *
 * BUFFER CAPACITY (host must preallocate):
 *   anchors  >= n_anchors + 5*n_splits
 *   surfaces >= n_surfaces + 3*n_splits
 * alloc_heads[0] = anchor cursor, alloc_heads[1] = surface cursor
 * (both start at the initial counts; the kernel only ever APPENDS).
 * ========================================================================= */

/* ---------- record accessors (pure pointer math, mirrors psf_splat.cl) */

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

/* corner-vanishing deformation basis (identical to psf_splat.cl) */
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

/* stitch ("sewing"), identical to psf_splat.cl */
inline float stitch_h(__global const char *S, uint surf, float u, float v,
                      float s_scale)
{
    __global const char *e = S + surf * 8u;
    float s = 0.0f;
    float tu = u*(1.0f-u), tv = v*(1.0f-v);
    s += (1.0f-v) * tu * (e[0]*s_scale + e[1]*s_scale*u);
    s +=        v * tu * (e[2]*s_scale + e[3]*s_scale*u);
    s += (1.0f-u) * tv * (e[4]*s_scale + e[5]*s_scale*v);
    s +=        u * tv * (e[6]*s_scale + e[7]*s_scale*v);
    return s;
}

inline float3 mix3(float3 a, float3 b, float t)
{ return a + (b - a) * t; }

/* 8x8 Gaussian elimination with partial pivoting (device, no deps) */
inline bool solve8(float M[8][8], float b[8], float out[8])
{
    for (int col = 0; col < 8; ++col) {
        int best = col;
        for (int r = col + 1; r < 8; ++r)
            if (fabs(M[r][col]) > fabs(M[best][col])) best = r;
        if (fabs(M[best][col]) < 1e-14f) return false;
        if (best != col) {
            for (int k = 0; k < 8; ++k) { float t=M[best][k]; M[best][k]=M[col][k]; M[col][k]=t; }
            float t = b[best]; b[best] = b[col]; b[col] = t;
        }
        for (int r = col + 1; r < 8; ++r) {
            const float f = M[r][col] / M[col][col];
            for (int k = col; k < 8; ++k) M[r][k] -= f * M[col][k];
            b[r] -= f * b[col];
        }
    }
    for (int col = 7; col >= 0; --col) {
        float s = b[col];
        for (int k = col + 1; k < 8; ++k) s -= M[col][k] * out[k];
        out[col] = s / M[col][col];
    }
    return true;
}

inline void cv_basis(float u, float v, float B[8])
{
    const float bu = u*(1.0f-u), bv = v*(1.0f-v);
    const float au = 2.0f*u-1.0f, av = 2.0f*v-1.0f;
    B[0]=bu*(1.0f-v); B[1]=bu*v; B[2]=bv*(1.0f-u); B[3]=bv*u;
    B[4]=bu*au*(1.0f-v); B[5]=bu*au*v; B[6]=bv*av*(1.0f-u); B[7]=bv*av*u;
}

/* quantize like AnchorPool::SetPosition / SetNormal */
inline void quantize_anchor(__global uchar *rec, float3 p, float3 n,
                            float3 bbmin, float3 bbspan)
{
    __global ushort *qp = (__global ushort *)(rec + 32u);
    __global uchar *qn  = rec + 44u;
    qp[0] = (ushort)(clamp((p.x - bbmin.x) / bbspan.x, 0.0f, 1.0f)
                     * 65535.0f + 0.5f);
    qp[1] = (ushort)(clamp((p.y - bbmin.y) / bbspan.y, 0.0f, 1.0f)
                     * 65535.0f + 0.5f);
    qp[2] = (ushort)(clamp((p.z - bbmin.z) / bbspan.z, 0.0f, 1.0f)
                     * 65535.0f + 0.5f);
    qn[0] = (uchar)clamp((n.x * 0.5f + 0.5f) * 255.0f + 0.5f, 0.0f, 255.0f);
    qn[1] = (uchar)clamp((n.y * 0.5f + 0.5f) * 255.0f + 0.5f, 0.0f, 255.0f);
    qn[2] = (uchar)clamp((n.z * 0.5f + 0.5f) * 255.0f + 0.5f, 0.0f, 255.0f);
}

__kernel void split_patches(
    __global uchar        *surfaces,   /* grows into preallocated free-list */
    __global uchar        *anchors,    /* grows into preallocated free-list */
    __global const uint   *split_list, /* patch ids flagged by the error
                                          estimator / LocalFEM             */
    __global uint         *alloc_heads,/* [0]=anchor cursor, [1]=surface   */
    const uint n_splits,
    const float3 bbmin, const float3 bbspan,
    __global char         *stitch,     /* 8 int8 per surface; children's
                                          slots are zeroed in-kernel */
    const float s_scale)
{
    const uint sp = get_global_id(0);
    if (sp >= n_splits) return;
    const uint pid = split_list[sp];
    if (pid >= 0xFFFFFFFFu) return;
    __global const uchar *prec = surfaces + (size_t)pid * 64u;
    if (((__global const uint *)(prec + 8u))[0] == 0xFFFFFFFFu) return; /* already retired */

    /* parent record */
    __global const ushort *pai = (__global const ushort *)(prec);
    const uint4 ai = (uint4)(pai[0], pai[1], pai[2], pai[3]);
    float pc[8];
    {
        __global const float *pc8 = (__global const float *)(prec + 12u);
        for (int i = 0; i < 8; ++i) pc[i] = pc8[i];
    }
    __global const char *pst = stitch + (size_t)pid * 8u;

    /* --- 1) reserve 5 anchors + 3 surfaces from the free lists -------- */
    const uint a0 = atomic_add(&alloc_heads[0], 5u);
    const uint s0 = atomic_add(&alloc_heads[1], 3u);

    /* --- 2) evaluate the 5 hanging points ON the parent --------------- */
    /* edge mids: v0->(0.5,0) v1->(0.5,1) u0->(0,0.5) u1->(1,0.5), ctr */
    const float2 MID[5] = { {0.5f,0.0f}, {0.5f,1.0f}, {0.0f,0.5f},
                            {1.0f,0.5f}, {0.5f,0.5f} };
    float3 mp[5]; float3 mn[5];
    for (int e = 0; e < 5; ++e) {
        const float u = MID[e].x, v = MID[e].y;
        float3 P0 = anchor_pos(anchors, ai.x, bbmin, bbspan);
        float3 P1 = anchor_pos(anchors, ai.y, bbmin, bbspan);
        float3 P2 = anchor_pos(anchors, ai.z, bbmin, bbspan);
        float3 P3 = anchor_pos(anchors, ai.w, bbmin, bbspan);
        float3 bil = mix3(mix3(P0, P3, u), mix3(P1, P2, u), v);
        float3 N0 = anchor_nrm(anchors, ai.x);
        float3 N1 = anchor_nrm(anchors, ai.y);
        float3 N2 = anchor_nrm(anchors, ai.z);
        float3 N3 = anchor_nrm(anchors, ai.w);
        float3 n  = fast_normalize(mix3(mix3(N0, N3, u), mix3(N1, N2, u), v));
        const float h = deform_h(pc, u, v) + stitch_h(pst, 0, u, v, s_scale);
        mp[e] = bil + h * n;
        mn[e] = n;
    }

    /* --- 3) write the 5 new AnchorRecords (inherit parent material) --- */
    for (int e = 0; e < 5; ++e) {
        __global uchar *rec = anchors + (size_t)(a0 + e) * 64u;
        /* copy parent record, then zero connectivity + curve params:
         * these are hanging anchors, the CPU seam-graph rebuild fixes
         * connectivity/curve_params after compaction */
        for (int k = 0; k < 64; ++k) rec[k] = prec[k];
        __global ushort *cn = (__global ushort *)(rec + 4u);
        for (int k = 0; k < 4; ++k) cn[k] = 0xFFFFu;
        for (int k = 16; k < 32; ++k) rec[k] = 0;
        quantize_anchor(rec, mp[e], mn[e], bbmin, bbspan);
    }

    /* quadrant layout (same convention as CPU SplitPatch:
     * u along A0->A3, v along A0->A1; slots [A0 A1 A2 A3]) */
    const uint m0 = a0 + 0;   /* v0 mid (0.5,0)  */
    const uint m1 = a0 + 1;   /* v1 mid (0.5,1)  */
    const uint m2 = a0 + 2;   /* u0 mid (0,0.5)  */
    const uint m3 = a0 + 3;   /* u1 mid (1,0.5)  */
    const uint ctr = a0 + 4;

    struct Q { float u0, v0; uint a0, a1, a2, a3; };
    const struct Q quads[4] = {
        { 0.0f, 0.0f, ai.x, m2,   ctr,  m0   },   /* u-,v- */
        { 0.0f, 0.5f, m2,   ai.y, m1,   ctr  },   /* u-,v+ */
        { 0.5f, 0.0f, m0,   ctr,  m3,   ai.w },   /* u+,v- */
        { 0.5f, 0.5f, ctr,  m1,   ai.z, m3   },   /* u+,v+ */
    };

    /* --- 4) fit each child: LSQ of the cv basis vs the parent --------- */
    /* G=9 matches the CPU SplitPatch fit grid exactly */
    const int G = 9;
    for (int q = 0; q < 4; ++q) {
        /* child slot: q=0 reuses the PARENT record, q=1..3 append */
        __global uchar *crec;
        if (q == 0) crec = surfaces + (size_t)pid * 64u;
        else        crec = surfaces + (size_t)(s0 + (q - 1)) * 64u;

        /* inherit color/material (bytes 44..63) from the parent */
        for (int k = 44; k < 64; ++k) crec[k] = prec[k];
        __global ushort *ca = (__global ushort *)crec;
        ca[0] = (ushort)quads[q].a0; ca[1] = (ushort)quads[q].a1;
        ca[2] = (ushort)quads[q].a2; ca[3] = (ushort)quads[q].a3;
        ((__global uint *)(crec + 8u))[0] =
            ((__global const uint *)(prec + 8u))[0];

        /* child corner positions/normals for bilinear + normal field */
        float3 CA[4], CN[4];
        CA[0] = anchor_pos(anchors, quads[q].a0, bbmin, bbspan);
        CA[1] = anchor_pos(anchors, quads[q].a1, bbmin, bbspan);
        CA[2] = anchor_pos(anchors, quads[q].a2, bbmin, bbspan);
        CA[3] = anchor_pos(anchors, quads[q].a3, bbmin, bbspan);
        CN[0] = anchor_nrm(anchors, quads[q].a0);
        CN[1] = anchor_nrm(anchors, quads[q].a1);
        CN[2] = anchor_nrm(anchors, quads[q].a2);
        CN[3] = anchor_nrm(anchors, quads[q].a3);

        float M[8][8]; float r[8];
        for (int x = 0; x < 8; ++x) { r[x] = 0.0f;
            for (int y = 0; y < 8; ++y) M[x][y] = 0.0f; }

        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float cu = (float)i / (float)(G - 1);
                const float cv = (float)j / (float)(G - 1);
                const float pu = quads[q].u0 + 0.5f * cu;
                const float pv = quads[q].v0 + 0.5f * cv;
                /* parent evaluation at (pu,pv) = the target surface */
                float3 P0 = anchor_pos(anchors, ai.x, bbmin, bbspan);
                float3 P1 = anchor_pos(anchors, ai.y, bbmin, bbspan);
                float3 P2 = anchor_pos(anchors, ai.z, bbmin, bbspan);
                float3 P3 = anchor_pos(anchors, ai.w, bbmin, bbspan);
                float3 tgt = mix3(mix3(P0, P3, pu), mix3(P1, P2, pu), pv);
                float3 N0 = anchor_nrm(anchors, ai.x);
                float3 N1 = anchor_nrm(anchors, ai.y);
                float3 N2 = anchor_nrm(anchors, ai.z);
                float3 N3 = anchor_nrm(anchors, ai.w);
                float3 pn = fast_normalize(mix3(mix3(N0, N3, pu),
                                                mix3(N1, N2, pu), pv));
                tgt += (deform_h(pc, pu, pv)
                        + stitch_h(pst, 0, pu, pv, s_scale)) * pn;
                /* child bilinear + normal field at (cu,cv) */
                float3 bil = mix3(mix3(CA[0], CA[3], cu),
                                  mix3(CA[1], CA[2], cu), cv);
                float3 n   = fast_normalize(mix3(mix3(CN[0], CN[3], cu),
                                                 mix3(CN[1], CN[2], cu), cv));
                const float h = (tgt.x - bil.x) * n.x
                               + (tgt.y - bil.y) * n.y
                               + (tgt.z - bil.z) * n.z;
                float B[8]; cv_basis(cu, cv, B);
                for (int x = 0; x < 8; ++x) {
                    r[x] += B[x] * h;
                    for (int y = 0; y < 8; ++y) M[x][y] += B[x] * B[y];
                }
            }

        float coef[8];
        if (!solve8(M, r, coef))
            for (int x = 0; x < 8; ++x) coef[x] = 0.0f;
        __global float *cc = (__global float *)(crec + 12u);
        for (int i = 0; i < 8; ++i) cc[i] = coef[i];
    }

    /* --- 5) reset the 4 children's stitch slots + retire the parent --- */
    {
        __global char *cst = stitch + (size_t)pid * 8u;
        for (int k = 0; k < 8; ++k) cst[k] = 0;
        for (int q = 1; q < 4; ++q) {   /* appended children: zero slots */
            __global char *cst2 = stitch + (size_t)(s0 + (q - 1)) * 8u;
            for (int k = 0; k < 8; ++k) cst2[k] = 0;
        }
    }
    /* parent is now child 0's slot -- nothing to retire (blob stays
     * hole-free); the host rebuilds the seam graph around the 4 ids
     * [pid, s0, s0+1, s0+2] after compaction */
}