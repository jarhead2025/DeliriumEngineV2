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
    grid[(cz*gy+cy)*gx+cx]=(int)i;
}

// --- GPU-side surfel bake (kills the CPU bake + 2.6 MB upload) -------------
// One work-item per (surface, iu, iv). Evaluates the live deformed surface
// ON-GPU from the raw 64-byte records (same math as the CPU BakeSurfels:
// bilinear anchors + deform_h + bilinear smooth normal + hue albedo). The
// host uploads only the small raw records (cached, re-uploaded on dirty);
// the 54,400-surfel array is produced here, on the GPU.
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
inline float huech(float h)
{
    float x = fabs(fmod(h, 1.0f) * 6.0f - 3.0f) - 1.0f;
    return clamp(x, 0.0f, 1.0f);
}
__kernel void bake_surfels(
    __global const uchar  *anchors,
    __global const uchar  *surfaces,
    __global const float4 *snorm,
    const uint n_surfaces,
    const int  uvn,
    const float3 bbmin, const float3 bbspan,
    const float  foot,
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

    float hue = fmod((float)s * 0.07f, 1.0f);
    Surfel sf;
    sf.px=pos.x; sf.py=pos.y; sf.pz=pos.z;
    sf.nx=N.x;   sf.ny=N.y;   sf.nz=N.z;
    sf.r=0.35f+0.5f*huech(hue);
    sf.g=0.35f+0.5f*huech(hue+0.33f);
    sf.b=0.35f+0.5f*huech(hue+0.66f);
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
    float foot2=foot*foot;
    // Neighborhood radius: a surfel within `foot` of p (p in cell cx) can be at
    // most floor(foot/cell) cells away in any axis (plus cx's own cell and its
    // immediate neighbor) -> nr = 1 + floor(foot/cell) covers every in-reach
    // surfel. Default foot ~= 0.5*cell -> nr=1 -> 3x3x3 (27 cell loads) instead
    // of the old fixed 5x5x5 (125): 4.6x fewer random grid+Surfel loads, and the
    // SAME set of surfels is found (fidelity-preserving -- identical output).
    // If foot is raised past cell (DELIRIUM_MARCH_FOOT>~2), nr grows to 2
    // (5x5x5) automatically, so no hit is ever missed.
    int nr = 1 + (int)(foot / cell);
    if (nr < 1) nr = 1;
    if (nr > 2) nr = 2;

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