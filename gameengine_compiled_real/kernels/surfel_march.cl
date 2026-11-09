// =============================================================================
// surfel_march.cl  -  Progressive per-pixel lateral ray march, DeliRium engine.
//
// USER'S METHOD, GPU-optimized for real-time:
//   Each pixel marches its view ray NEAR->FAR through the 3D uniform grid of
//   surfels (baked from the live deformed PSF) for O(1) lookup.
//   accumulate=1 (default, 2026-10-28): FRONTMOST-DOMINANT ACCUMULATION --
//   every hit layer along the ray is ADDED UP (weighted 1/d so nearer layers
//   dominate), the march stops when the ray crosses a surface (two
//   consecutive hit steps, or a dense cluster), and the pixel is written
//   ONCE after the loop. Layers behind the front face never contribute.
//   accumulate=0: legacy early-exit (first hit wins) for A/B.
//
//   SHADING: every surfel within the hit radius is accumulated, weighted
//   1/(1+d2), and the normals+colors AVERAGED (not one closest) -> smooth
//   normal field, no "shaky between the seams". The closest surfel sets the
//   exact hit depth (closest point on the ray to it).
//
//   FILL: after the march, 2 passes of "groups of 3" hole-filling — an empty
//   pixel with >=FILL_MIN filled neighbors takes their AVERAGE color +
//   average depth. Closes residual sub-surfel gaps without cross-layer
//   interference (single buffer, so no interference at all).
//
//   COMPOSE: the single filled buffer IS the image (min-depth is trivially
//   the frontmost hit) -> copy to buf_img.
//
//   COMPOSE (multi-body): MARCH IS THE PRIMARY RENDERER -- every body's
//   marched layer is written into the SHARED framebuffer buf_fb (depth|rgba
//   64-bit key, same scale as psf_splat.cl) by march_compose_fb, which uses
//   atom_min so the FRONTMOST hit across all bodies wins (two-pass depth32
//   fallback when the device lacks 64-bit atomics, like psf_splat). Then
//   resolve_fb produces buf_img.
// =============================================================================

typedef struct {
    float px,py,pz, nx,ny,nz, r,g,b, radius;
} Surfel;

// --- 3D grid kernels ---------------------------------------------------------
__kernel void clear_grid(__global int* grid, const uint n){
    uint i=get_global_id(0); if(i<n) grid[i]=-1;
}
__kernel void build_grid(
    __global const Surfel* S, const uint n,
    __global int* grid,
    const int gx,const int gy,const int gz,
    const float3 org, const float cell)
{
    uint i=get_global_id(0); if(i>=n) return;
    int cx=(int)((S[i].px-org.x)/cell);
    int cy=(int)((S[i].py-org.y)/cell);
    int cz=(int)((S[i].pz-org.z)/cell);
    if(cx<0||cx>=gx||cy<0||cy>=gy||cz<0||cz>=gz) return;
    // DETERMINISTIC cell ownership.
    //
    // A plain store (`grid[cell] = i`) is a RACE when several surfels land
    // in the same cell: the winner is whichever work-item happens to write
    // last, and that order is not defined. The symptom is a render that
    // differs between two runs of the SAME binary on the SAME input:
    // measured 1.09% of pixels differing (7,534 of 691,200), with only 9
    // pixels differing by more than 16/255 -- i.e. visible noise that
    // changes every frame, not a stable artefact.
    //
    // atomic_max makes the winner a FUNCTION OF THE DATA rather than of the
    // scheduling: the same surfel wins every time.
    //
    // WHY atomic_max and NOT atomic_min: the grid is cleared to -1 (see
    // clear_grid), and -1 is the "empty cell" sentinel. atomic_min would
    // keep -1 forever, because -1 < every valid index -- the grid would
    // never receive a surfel and the march would find nothing. atomic_max
    // against a -1 sentinel correctly replaces -1 with any valid index.
    // (Which of several colliding surfels wins is arbitrary; that it is
    // the SAME one every run is the property that matters.)
    int cell_idx = (cz*gy+cy)*gx+cx;
    atomic_max(&grid[cell_idx], (int)i);
}

// --- GPU-side surfel bake (kills the CPU bake + 2.6 MB upload) -------------
// One work-item per (surface, iu, iv). Evaluates the live deformed surface
// ON-GPU from the raw 64-byte records (same math as the CPU BakeSurfels:
// bilinear anchors + deform_h + bilinear smooth normal + hue albedo). The
// host uploads only the small raw records (cached, re-uploaded on dirty);
// the 54,400-surfel array is produced here, on the GPU.
//
// MODEL MATRIX (world transform): the 64-byte records are FROZEN and hold
// the body in its OWN local space. A workspace needs to MOVE objects around
// without re-quantizing (and losing precision on) every anchor, so the
// placement lives OUTSIDE the records and is applied here, on the GPU.
// `model` is a row-major 4x4 (local -> world). `model_nrm` is the rotation
// part only (normals must not be translated; with uniform scale they need
// no renormalization beyond the normalize we already do).
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
/* THE COLOUR PATTERN (ported from kernels/psf_splat.cl).
 *
 * WHY IT IS HERE: the surfel bake used to invent a PROCEDURAL HUE per
 * patch (`fmod(s * 0.07f, 1.0f)`) -- a leftover from an early debugging
 * session that was never reverted. The result was a rainbow-striped
 * render: every patch a different hue, and the patch grid showing up as
 * horizontal banding. The body's ACTUAL colour (the frozen 64-byte
 * color_pattern field) was ignored entirely.
 *
 * The helpers are byte-identical to the splat kernel's, so both render
 * paths now read the same colour from the same records. */
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

__kernel void bake_surfels(
    __global const uchar  *anchors,
    __global const uchar  *surfaces,
    __global const float4 *snorm,
    const uint n_surfaces,
    const int  uvn,
    const float3 bbmin, const float3 bbspan,
    const float  foot,
    const float4 model_r0, const float4 model_r1,
    const float4 model_r2, const float4 model_r3,
    __global Surfel* out)
{
    uint u = (uint)uvn;
    uint gid = get_global_id(0);
    uint total = n_surfaces * u * u;
    if (gid >= total) return;
    uint s   = gid / (u*u);
    uint rem = gid % (u*u);
    uint iu  = rem % u;
    uint iv  = rem / u;
    float uu = (float)(iu + 0.5f) / (float)u;
    float vv = (float)(iv + 0.5f) / (float)u;

    __global const uchar  *srec = surfaces + s*64u;
    __global const ushort *aidx = (__global const ushort *)srec;
    uint a0=aidx[0], a1=aidx[1], a2=aidx[2], a3=aidx[3];

    float3 A0=anchor_pos(anchors,a0,bbmin,bbspan);
    float3 A1=anchor_pos(anchors,a1,bbmin,bbspan);
    float3 A2=anchor_pos(anchors,a2,bbmin,bbspan);
    float3 A3=anchor_pos(anchors,a3,bbmin,bbspan);
    float3 P = mix(mix(A0,A1,uu), mix(A3,A2,uu), vv);   // eval_patch convention

    float3 N0=anchor_nrm(anchors,a0), N1=anchor_nrm(anchors,a1);
    float3 N2=anchor_nrm(anchors,a2), N3=anchor_nrm(anchors,a3);
    float3 Nd = fast_normalize(mix(mix(N0,N1,uu), mix(N3,N2,uu), vv));

    float c[8];
    __global const float *cf = (__global const float *)(srec + 12u);
    for (int i = 0; i < 8; ++i) c[i] = cf[i];
    float h = deform_h(c, uu, vv);
    float3 pos = P + Nd * h;

    float3 S0=snorm[a0].xyz, S1=snorm[a1].xyz, S2=snorm[a2].xyz, S3=snorm[a3].xyz;
    float3 N = fast_normalize(mix(mix(S0,S1,uu), mix(S3,S2,uu), vv));

    // WORLD TRANSFORM: local -> world. The position is transformed as a
    // POINT (translation applies); the normal is transformed as a DIRECTION
    // (rotation only, no translation) and renormalized (the rotation may
    // carry a scale, which changes the normal's length).
    float3 wpos = xform_point(model_r0, model_r1, model_r2, model_r3, pos);
    float3 wnrm = fast_normalize(xform_dir(model_r0, model_r1, model_r2, N));

    /* THE BODY'S REAL COLOUR, read from the frozen 64-byte
     * color_pattern field -- the SAME bytes the splat path reads, so both
     * paths render the same body in the same colour.
     *
     * (This replaces a procedural per-patch hue that made every patch a
     * different colour and turned the patch grid into visible horizontal
     * banding.) */
    Surfel sf;
    sf.px=wpos.x; sf.py=wpos.y; sf.pz=wpos.z;
    sf.nx=wnrm.x;   sf.ny=wnrm.y;   sf.nz=wnrm.z;
    sf.r=channel_color(srec, 0u, uu, vv);
    sf.g=channel_color(srec, 1u, uu, vv);
    sf.b=channel_color(srec, 2u, uu, vv);
    sf.radius=foot;
    out[gid]=sf;
}

// --- clear the single march buffer ------------------------------------------
__kernel void layer_clear(
    __global uchar4* color, __global float* depth,
    const uint n, const float INF)
{
    uint i=get_global_id(0);
    if(i>=n) return;
    color[i]=(uchar4)(0,0,0,0);
    depth[i]=INF;
}

// --- single full-range march: NEAR->FAR, first (nearest) hit wins -----------
__kernel void march_draw(
    __global const Surfel* S,
    __global const int*    grid,
    const int gx,const int gy,const int gz,
    const float3 org, const float cell,
    const float3 eye, const float3 fwd,
    const float3 right, const float3 up,
    const float tanfov,
    const uint W, const uint H,
    const float d_near, const float d_far,
    const int steps,
    const float foot,             // hit radius (world units)
    const float3 light,
    const float4 mvp_s0123, const float4 mvp_s4567,
    const float4 mvp_s89ab, const float4 mvp_scdef,
    __global uchar4* color, __global float* depth,
    const int use_slab,
    const int accumulate)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    float ndcx=((x+0.5f)/W-0.5f)*2.0f;
    float ndcy=(0.5f-(y+0.5f)/H)*2.0f;
    float step=(d_far-d_near)/steps;
    // Neighborhood radius: a surfel within `foot` of p (p in cell cx) can be at
    // most floor(foot/cell) cells away in any axis (plus cx's own cell and its
    // immediate neighbor) -> nr = 1 + floor(foot/cell) covers every in-reach
    // surfel. Default foot ~= 0.5*cell -> nr=1 -> 3x3x3 (27 cell loads) instead
    // of the old fixed 5x5x5 (125): 4.6x fewer random grid+Surfel loads, and the
    // SAME set of surfels is found (fidelity-preserving -- identical output).
    // If foot is raised past cell (DELIRIUM_MARCH_FOOT>~2), nr grows to 2
    // (5x5x5) automatically, so no hit is ever missed.
    //
    // HARD INVARIANT: the search radius must cover the hit radius, i.e.
    //   nr * cell >= foot
    // The host guarantees this by sizing foot = max(2*cell, 1.6*spacing) and
    // capping nr at 2. If foot ever exceeded 2*cell, the 5x5x5 search would
    // be NARROWER than the hit test -> hits silently missed (measured: the
    // FOOT=8 env multiplier, foot/cell=4.4, produced 845 holes and 52% edge
    // density -- the same broken appearance as the old undersized foot).
    // The clamp below is therefore a SAFETY net, not a tuning knob: if it
    // would clamp, the hit radius must be reduced to match.
    int nr = 1 + (int)(foot / cell);
    if (nr < 1) nr = 1;
    // `foot` is a const kernel argument -> clamp into a LOCAL so the search
    // radius can never be narrower than the hit test (see the invariant above).
    float foot_eff = foot;
    if (nr > 2) { nr = 2; foot_eff = 2.0f * cell; }
    float foot2=foot_eff*foot_eff;               // AFTER the clamp

    // ---- per-ray slab intersection (the "preallocated t-range") ---------
    // The view ray is a straight line p(d) = eye + d*L where
    // L = fwd + right*(ndcx*tanfov) + up*(ndcy*tanfov). Instead of marching
    // every pixel from d_near to d_far (the full bbox range), compute the
    // t-range [t0,t1] where THIS ray crosses the grid AABB (org ..
    // org+cell*grid) and march only within it. Off-center pixels that
    // skim the body (or miss it) skip most of the range -> 2-4x fewer
    // steps on a scene where the object doesn't fill the frame. If the ray
    // misses the grid AABB entirely, there are no surfels to hit -> return.
    // use_slab=0: march the full [d_near, d_far] range (the pre-2026-10-28
    // behavior). use_slab=1: march only the per-ray slab-intersection range.
    // The env DELIRIUM_MARCH_SLAB=0 disables the slab cull for A/B testing.
    int s0 = 0, s1 = steps - 1;
    if(use_slab){
        float3 L = fwd + right*(ndcx*tanfov) + up*(ndcy*tanfov);
        float3 gmin = org;
        float3 gmax = org + cell*(float3)(gx, gy, gz);
        float t0 = 0.0f, t1 = 1e30f;
        bool slab_ok = true;
        if(fabs(L.x) > 1e-8f){
            float a=(gmin.x-eye.x)/L.x, b=(gmax.x-eye.x)/L.x;
            if(a>b){float t=a;a=b;b=t;}
            t0=max(t0,a); t1=min(t1,b);
        } else if(eye.x<gmin.x || eye.x>gmax.x) slab_ok=false;
        if(fabs(L.y) > 1e-8f){
            float a=(gmin.y-eye.y)/L.y, b=(gmax.y-eye.y)/L.y;
            if(a>b){float t=a;a=b;b=t;}
            t0=max(t0,a); t1=min(t1,b);
        } else if(eye.y<gmin.y || eye.y>gmax.y) slab_ok=false;
        if(fabs(L.z) > 1e-8f){
            float a=(gmin.z-eye.z)/L.z, b=(gmax.z-eye.z)/L.z;
            if(a>b){float t=a;a=b;b=t;}
            t0=max(t0,a); t1=min(t1,b);
        } else if(eye.z<gmin.z || eye.z>gmax.z) slab_ok=false;
        if(!slab_ok || t1 < t0) return;   // ray misses the grid AABB -> no hit
        float d_lo = max(d_near, t0);
        float d_hi = min(d_far, t1);
        if(d_hi <= d_lo) return;
        s0 = (int)ceil((d_lo - d_near)/step);
        s1 = (int)floor((d_hi - d_near)/step);
        if(s0 < 0) s0 = 0;
        if(s1 >= steps) s1 = steps - 1;
        if(s0 > s1) return;
    }
    // FRONTMOST-DOMINANT ACCUMULATION (accumulate=1, the user's method):
    // march near->far and ADD UP every hit layer (each layer = the weighted
    // 5x5x5 neighborhood at that step), then write the pixel ONCE after the
    // loop. Layers are weighted 1/d so NEARER layers dominate (frontmost-
    // dominant). Occlusion: stop as soon as the ray has crossed a surface --
    // two CONSECUTIVE hit steps (a surface crossing is a chord of ~2*foot,
    // and with step<=foot that always yields >=2 consecutive hit steps), or a
    // DENSE cluster (>=2 surfels within foot in one step). Everything behind
    // the front face never contributes -> no back-face bleed, no zebra
    // normal-averaging in holes. The same-object check is FREE: each body
    // marches its OWN grid (built only from its own surfels), so every hit is
    // by definition the same object -- no object-ID lookup.
    // accumulate=0: the old early-exit (first hit written immediately) for
    // A/B via DELIRIUM_MARCH_ACCUM=0.
    float awsum=0.f, anx=0.f, any=0.f, anz=0.f;
    float aR=0.f, aG=0.f, aB=0.f;
    int prev_hit=0;
    int front_si=-1;
    for(int s=s0;s<=s1;++s){
        float d=d_near+s*step;
        float hsize=d*tanfov;
        float3 p=eye+fwd*d+right*(ndcx*hsize)+up*(ndcy*hsize);
        int cx=(int)((p.x-org.x)/cell);
        int cy=(int)((p.y-org.y)/cell);
        int cz=(int)((p.z-org.z)/cell);
        if(cx<0||cx>=gx||cy<0||cy>=gy||cz<0||cz>=gz){ prev_hit=0; continue; }

        // (2*nr+1)^3 neighborhood: ACCUMULATE every surfel within `foot`,
        // weighted 1/(1+d2). The CLOSEST surfel sets this step's hit depth.
        // nr is the adaptive radius (1 -> 3x3x3 at default foot, 2 -> 5x5x5
        // if foot is raised past cell) -- see the nr computation above.
        float wsum=0.f, nx=0.f, ny=0.f, nz=0.f;
        float R=0.f, G=0.f, B=0.f;
        float best_d2=foot2; int best_si=-1; int cnt=0;
        for(int dz=-nr;dz<=nr;++dz){
            int ncz=cz+dz; if(ncz<0||ncz>=gz) continue;
            for(int dy=-nr;dy<=nr;++dy){
                int ncy=cy+dy; if(ncy<0||ncy>=gy) continue;
                for(int dx=-nr;dx<=nr;++dx){
                    int ncx=cx+dx; if(ncx<0||ncx>=gx) continue;
                    int si=grid[(ncz*gy+ncy)*gx+ncx]; if(si<0) continue;
                    Surfel ss=S[si];
                    float ddx=ss.px-p.x, ddy=ss.py-p.y, ddz=ss.pz-p.z;
                    float d2=ddx*ddx+ddy*ddy+ddz*ddz;
                    if(d2>=foot2) continue;
                    float w=1.f/(1.f+d2);
                    wsum+=w; nx+=w*ss.nx; ny+=w*ss.ny; nz+=w*ss.nz;
                    R+=w*ss.r; G+=w*ss.g; B+=w*ss.b;
                    if(d2<best_d2){ best_d2=d2; best_si=si; }
                    cnt++;
                }
            }
        }
        if(best_si<0){ prev_hit=0; continue; }

        if(accumulate==0){
            // ---- legacy early-exit: first hit wins, write immediately ----
            float inv=1.f/wsum;
            float3 n=normalize((float3)(nx*inv, ny*inv, nz*inv));
            // ensure the normal faces the camera (fwd points eye->surface);
            // matches the splat back-face cull so shading is stable.
            if(dot(n, fwd) > 0.f) n = -n;
            // key + camera fill + floor (EXACT splat lighting model).
            float keyL  = clamp(dot(n, light), 0.f, 1.f);
            float fillL = clamp(dot(n, -fwd),  0.f, 1.f);
            float lam   = clamp(keyL*0.85f + fillL*0.30f, 0.f, 1.f) + 0.15f;
            float cr=R*inv, cg=G*inv, cb=B*inv;
            color[pid]=(uchar4)((uchar)(clamp(cr*lam,0.f,1.f)*255),
                                (uchar)(clamp(cg*lam,0.f,1.f)*255),
                                (uchar)(clamp(cb*lam,0.f,1.f)*255),255);
            Surfel bs=S[best_si];
            float3 hpf=(float3)(bs.px,bs.py,bs.pz);
            float4 hp = (float4)(
                dot(mvp_s0123, (float4)(hpf, 1.0f)),
                dot(mvp_s4567, (float4)(hpf, 1.0f)),
                dot(mvp_s89ab, (float4)(hpf, 1.0f)),
                dot(mvp_scdef, (float4)(hpf, 1.0f)));
            if (hp.w <= 1e-6f) continue;   // behind camera: keep marching
            float ndcz = hp.z / hp.w;
            if (ndcz < 0.0f) continue;     // behind near plane
            depth[pid] = ndcz;             // exact splat-path scale (no clamp)
            return;
        }

        // ---- frontmost-dominant: add this layer, then test for occlusion --
        if(front_si<0) front_si=best_si;
        float wstep=1.f/max(d,1e-4f);
        awsum+=wstep*wsum; anx+=wstep*nx; any+=wstep*ny; anz+=wstep*nz;
        aR+=wstep*R; aG+=wstep*G; aB+=wstep*B;
        // Surface crossed: consecutive hit steps, or a dense cluster.
        if(prev_hit==1 || cnt>=2) break;
        prev_hit=1;
    }
    if(front_si<0) return;   // no hit: leave the pixel empty (fill bridges)

    // ---- write the accumulated result ONCE (the user's "buffer at the end")
    float inv=1.f/awsum;
    float3 n=normalize((float3)(anx*inv, any*inv, anz*inv));
    if(dot(n, fwd) > 0.f) n = -n;
    // key + camera fill + floor (EXACT splat lighting model).
    float keyL  = clamp(dot(n, light), 0.f, 1.f);
    float fillL = clamp(dot(n, -fwd),  0.f, 1.f);
    float lam   = clamp(keyL*0.85f + fillL*0.30f, 0.f, 1.f) + 0.15f;
    float cr=aR*inv, cg=aG*inv, cb=aB*inv;
    color[pid]=(uchar4)((uchar)(clamp(cr*lam,0.f,1.f)*255),
                        (uchar)(clamp(cg*lam,0.f,1.f)*255),
                        (uchar)(clamp(cb*lam,0.f,1.f)*255),255);
    // NDC depth of the FRONTMOST hit surfel (SAME scale as psf_splat.cl) so
    // the compose interleaves by depth correctly with the other bodies.
    Surfel bs=S[front_si];
    float3 hpf=(float3)(bs.px,bs.py,bs.pz);
    float4 hp = (float4)(
        dot(mvp_s0123, (float4)(hpf, 1.0f)),
        dot(mvp_s4567, (float4)(hpf, 1.0f)),
        dot(mvp_s89ab, (float4)(hpf, 1.0f)),
        dot(mvp_scdef, (float4)(hpf, 1.0f)));
    if (hp.w <= 1e-6f) return;
    float ndcz = hp.z / hp.w;
    if (ndcz < 0.0f) return;
    depth[pid] = ndcz;
}

// --- fill holes: empty pixel with >=FILL_MIN filled neighbors takes their
// --- AVERAGE color + average depth (the "groups of 3, averaged" rule). ------
__kernel void layer_fill(
    __global const uchar4* cin, __global const float* din,
    __global uchar4* cout,      __global float* dout,
    const uint W, const uint H,
    const float INF, const float fill_depth, const int fill_min)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    cout[pid]=cin[pid]; dout[pid]=din[pid];   // copy through
    if(din[pid]<INF) return;                  // already filled
    if(x==0||x==W-1||y==0||y==H-1) return;
    int R=0,G=0,B=0,cnt=0; float dsum=0.f;
    for(int dy=-1;dy<=1;++dy)
    for(int dx=-1;dx<=1;++dx){
        if(dx==0&&dy==0) continue;
        uint n=(y+dy)*W+(x+dx);
        if(din[n]<INF){ uchar4 c=cin[n]; R+=c.x;G+=c.y;B+=c.z; dsum+=din[n]; cnt++; }
    }
    if(cnt>=fill_min){
        cout[pid]=(uchar4)((uchar)(R/cnt),(uchar)(G/cnt),(uchar)(B/cnt),255);
        dout[pid]=dsum/cnt;   // average neighbor depth -> depth-continuous
    }
}

// --- compose: single buffer -> buf_img (RGBA8888), bg where empty -----------
__kernel void layer_compose(
    __global const uchar4* color,
    __global const float*  depth,
    const uint W, const uint H,
    const float INF, const uchar4 bg,
    __global uchar4* out)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    out[pid] = (depth[pid]<INF) ? color[pid] : bg;
}

// =============================================================================
// THE LAYERED BACK-TO-FRONT PATH (the user's method, restored).
//
// WHAT THE USER DESCRIBED:
//   "you could average them out when intersecting them with variable
//    distance planes from the camera's origin, with a normal vector
//    parallel to the camera's direction, the original image buffer would
//    be a skybox one and these would stack back to front overwriting each
//    other in that sequence, after the last one the image acquired would
//    only then be shown"
//
// WHY IT MATTERS: the single-pass march_draw (above) resolves ONE hit per
// pixel with a frontmost-wins depth test. Every pixel that misses is a
// HOLE, and the only repair is a screen-space fill -- which can only ever
// copy a neighbour's colour, never recover the surface that was actually
// there. The layered method removes that class of hole entirely:
//
//   1. The ray is swept through DISTANCE PLANES from the camera origin.
//      Each plane is the FOV square at depth d: half-size d*tan(FOV/2),
//      mapped 1:1 to the screen. The plane's normal is parallel to the
//      camera's direction by construction.
//   2. Each plane (a "band" between d_far and d_near) is drawn into its
//      OWN buffer. A band never sees another band's pixels, so a band
//      whose hits are sparse can fill its own holes from ITS OWN hits --
//      the fill is a genuine average of impacts that share the same
//      distance plane, not a smear across depth.
//   3. The bands are stacked BACK TO FRONT (far first, near last). Each
//      overwrites what is behind it, so the nearest band wins.
//   4. Only after the last band is the image shown.
//
// This is the golden reference's algorithm
// (/home/josias/Desktop/Tartarus/registry/golden/kernels/surfel_march.cl:
// layer_draw + layer_fill + layer_compose), re-expressed with this tree's
// surfel struct, the 3x3x3 grid-neighbourhood search, the weighted
// normal averaging, and the MVP-corrected depth.
//
// The difference that matters: in the golden version a band wrote ONE
// colour and ONE depth per pixel. Here a band writes the WEIGHTED AVERAGE
// of all impacts it found (the "average them out" rule) and the exact
// intersection depth (not the step depth), which is what makes the
// composited result free of the squiggly contour lines that a quantized
// depth produces.
// =============================================================================

// Draw the impacts that fall inside ONE distance band into that band's own
// buffer.
//
// The band is [d_far, d_near]. The ray is sampled at BSTEPS depths inside
// it; at each depth the FOV square at that distance is mapped 1:1 to the
// screen, so the ray point p(d) is:
//
//   p = eye + fwd*d + right*(ndcx*hsize) + up*(ndcy*hsize)
//   hsize = d*tanfov
//
// Impacts are found by the same 3x3x3 grid-neighbourhood search the
// single-pass path uses (a surfel near a cell boundary can be quantized
// into a neighbouring cell), and all impacts within `foot` of the sample
// point are ACCUMERATED with a 1/(1+d^2) weight -- so the band's colour is
// the average of its impacts, not the nearest one's colour.
__kernel void layer_draw_band(
    __global const Surfel* S,
    __global const int*    grid,
    const int gx,const int gy,const int gz,
    const float3 org, const float cell,
    const float3 eye, const float3 fwd,
    const float3 right, const float3 up,
    const float tanfov,
    const float aspect,
    const uint W, const uint H,
    const float d_far, const float d_near,
    const int bsteps,
    const float foot,
    const float3 light,
    const float4 mvp_s0123, const float4 mvp_s4567,
    const float4 mvp_s89ab, const float4 mvp_scdef,
    __global uchar4* color, __global float* depth)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    float ndcx=((x+0.5f)/W-0.5f)*2.0f;
    float ndcy=(0.5f-(y+0.5f)/H)*2.0f;
    float step=(d_far-d_near)/bsteps;

    // The band's accumulator.
    float aw=0.f, anx=0.f, any_=0.f, anz=0.f;
    float aR=0.f, aG=0.f, aB=0.f;
    float best_t=1e30f, best_d=1e30f;

    // The neighbourhood radius that covers the hit radius.
    //
    // HARD INVARIANT (the same one march_draw obeys): the search radius
    // must COVER the hit radius, i.e. nr * cell >= foot. With
    // foot = max(2*cell, 1.6*spacing) the ratio foot/cell is at least 2,
    // so nr must be 2 (a 5x5x5 search). Searching only 3x3x3 (nr=1) with a
    // hit radius of 2 cells MISSES every surfel that is not in the exact
    // cell -- which is why the band path lit only the pixels whose ray
    // happened to land near a surfel's own cell.
    int nr = 1 + (int)(foot / cell);
    if (nr < 1) nr = 1;
    if (nr > 2) nr = 2;
    // `foot` is const -> clamp into a LOCAL, so the search radius can never
    // be narrower than the hit test.
    float foot_eff = foot;
    if (nr * cell < foot_eff) foot_eff = (float)nr * cell;
    float foot2 = foot_eff * foot_eff;

    for(int s=0; s<bsteps; ++s){
        float d = d_far - s*step;          // FAR -> NEAR within the band
        // THE FOV SQUARE AT DEPTH d: half-size d*tanfov vertically, and
        // d*tanfov*ASPECT horizontally.
        //
        // WHY THE ASPECT FACTOR IS REQUIRED: Camera::BuildMVP divides the
        // X row by the aspect ratio (P[0][0] = t/aspect), so the
        // projection's horizontal half-extent at depth d is d*tanfov*aspect.
        // Without it the band path samples a SQUARE where the projection is
        // a RECTANGLE: on a non-square viewport (the workspace's docked
        // viewport is 762x511, the gallery's is 640x480) the geometry is
        // searched in the wrong place and is pushed off the side of the
        // frame. The single-pass march_draw has the same omission, but its
        // slab cull masks it.
        float hsize = d*tanfov;
        float hsize_x = hsize*aspect;
        float3 p = eye + fwd*d + right*(ndcx*hsize_x) + up*(ndcy*hsize);
        int cx=(int)((p.x-org.x)/cell);
        int cy=(int)((p.y-org.y)/cell);
        int cz=(int)((p.z-org.z)/cell);
        if(cx<0||cx>=gx||cy<0||cy>=gy||cz<0||cz>=gz) continue;

        // 3x3x3 (or 5x5x5) neighbourhood search: accumulate EVERY surfel
        // within `foot` of p, weighted by 1/(1+d^2).
        for(int dz=-nr;dz<=nr;++dz){
            int ncz=cz+dz; if(ncz<0||ncz>=gz) continue;
            for(int dy=-nr;dy<=nr;++dy){
                int ncy=cy+dy; if(ncy<0||ncy>=gy) continue;
                for(int dx=-nr;dx<=nr;++dx){
                    int ncx=cx+dx; if(ncx<0||ncx>=gx) continue;
                    int si=grid[(ncz*gy+ncy)*gx+ncx]; if(si<0) continue;
                    float ddx=S[si].px-p.x, ddy=S[si].py-p.y, ddz=S[si].pz-p.z;
                    float d2=ddx*ddx+ddy*ddy+ddz*ddz;
                    if(d2>foot2) continue;
                    // The EXACT intersection depth, using the SAME MVP
                    // convention as march_draw (row-vector x matrix):
                    //   hp = (dot(m0,(p,1)), dot(m1,(p,1)), ...)
                    // The matrix x row-vector form transposes the MVP and
                    // yields a wrong depth off the screen's centre axis.
                    float4 hp = (float4)(
                        dot(mvp_s0123, (float4)(S[si].px, S[si].py, S[si].pz, 1.0f)),
                        dot(mvp_s4567, (float4)(S[si].px, S[si].py, S[si].pz, 1.0f)),
                        dot(mvp_s89ab, (float4)(S[si].px, S[si].py, S[si].pz, 1.0f)),
                        dot(mvp_scdef, (float4)(S[si].px, S[si].py, S[si].pz, 1.0f)));
                    if(hp.w <= 1e-6f) continue;   // behind the camera
                    float t = hp.z/hp.w;
                    if(t < 0.0f) continue;        // behind the near plane
                    float wgt = 1.0f/(1.0f+d2);
                    aw += wgt;
                    anx += wgt*S[si].nx; any_ += wgt*S[si].ny; anz += wgt*S[si].nz;
                    aR += wgt*S[si].r; aG += wgt*S[si].g; aB += wgt*S[si].b;
                    // The NEAREST impact sets the band's depth.
                    if(t < best_t){ best_t = t; }
                }
            }
        }
    }

    if(aw <= 1e-9f) return;               // this band hit nothing

    // The band's colour is the WEIGHTED AVERAGE of its impacts.
    float inv = 1.f/aw;
    anx*=inv; any_*=inv; anz*=inv;
    aR*=inv; aG*=inv; aB*=inv;
    float3 n = normalize((float3)(anx, any_, anz));
    // Ensure the normal faces the camera (fwd points eye->surface), the
    // same convention march_draw and the splat path use. Without this the
    // off-axis parts of the surface are lit from behind and go black.
    if(dot(n, fwd) > 0.f) n = -n;

    // The EXACT splat lighting model (key + camera fill + emissive floor),
    // so the band path and the splat path agree on brightness.
    float keyL  = clamp(dot(n, light), 0.f, 1.f);
    float fillL = clamp(dot(n, -fwd),  0.f, 1.f);
    float lam   = clamp(keyL*0.85f + fillL*0.30f, 0.f, 1.f) + 0.15f;
    color[pid]=(uchar4)((uchar)(clamp(aR*lam,0.f,1.f)*255),
                        (uchar)(clamp(aG*lam,0.f,1.f)*255),
                        (uchar)(clamp(aB*lam,0.f,1.f)*255),255);
    depth[pid]=best_t;
}

// --- stack the bands BACK TO FRONT into the shared framebuffer -------------
//
// The stacking itself is done by `march_compose_fb` (below), which packs
// (depth|rgba) into a 64-bit key and does atom_min into the shared
// framebuffer -- so the NEAREST band wins, exactly the "stack back to
// front, each overwriting the one behind" rule, and the march path
// interleaves by depth with the splat path just like psf_splat.
//
// There is deliberately NO separate stacking kernel: march_compose_fb
// already does the atom_min, already has the 32-bit two-pass fallback for
// devices without 64-bit atomics, and already carries the fracture
// fragment's depth bias. A second implementation would have to duplicate
// all of that -- and would be one more thing to keep in sync.

// --- compose marched layer into the SHARED framebuffer (buf_fb) --------------
// MARCH IS THE PRIMARY RENDERER: every body (not just body 0) marches and
// composes here. FRONTMOST WINS: the key is (depth_bits << 32) | rgba, and
// atom_min keeps the NEAREST hit across ALL bodies (the high 32 bits are the
// float depth bits, monotonic with distance; the low 32 bits carry the winner's
// color). This is exactly what psf_splat does, so a marched body and a
// splat-fallback body interleave by depth identically. The host launches this
// TWICE when the device lacks 64-bit atomics (pass 0 claims the depth with a
// 32-bit atomic_min, pass 1 stores the color only if this body won) -- the
// same two-pass fallback as psf_splat.
__kernel void march_compose_fb(
    __global const uchar4* color,
    __global const float*  depth,
    const uint W, const uint H,
    const float INF,
    const float depth_bias,   /* NDC depth added to this body (fracture
                                 fragment: behind its coplanar parent cut edge,
                                 same as psf_splat's depth_bias) */
    const uint pass,
    __global ulong* fb,
    __global uint* depth32)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    if(depth[pid]>=INF) return;  // no hit: leave buf_fb untouched
    float d = depth[pid] + depth_bias;   /* fragment bias: behind its parent */
    uint rgba = ((uint)color[pid].x << 24) | ((uint)color[pid].y << 16) |
                ((uint)color[pid].z << 8) | 255u;
#ifdef USE_LONG_ATOMICS
    ulong key = ((ulong)as_uint(d) << 32) | (ulong)rgba;
    atom_min(&fb[pid], key);
#else
    if (pass == 0u) {
        uint exd = (uint)(fb[pid] >> 32);
        uint w   = (exd < as_uint(d)) ? exd : as_uint(d);
        atomic_min(&depth32[pid], w);
    } else {
        if (depth32[pid] == as_uint(d))
            fb[pid] = ((ulong)as_uint(d) << 32) | (ulong)rgba;
    }
#endif
}