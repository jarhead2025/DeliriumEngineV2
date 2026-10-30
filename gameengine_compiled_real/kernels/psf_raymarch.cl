// GPU-owned equation raymarch: CPU only supplies camera + frustum planes.
// Each pixel ray; each surface does AABB∩frustum then adaptive uvn from depth.
// Alpha 0 = empty; alpha = object_id for object-masked stitch.

inline float deform_h(const float c[8], float u, float v) {
    const float bu = u*(1.f-u), bv = v*(1.f-v);
    const float au = 2.f*u-1.f, av = 2.f*v-1.f;
    const float w = 1.f-v, x = 1.f-u;
    return mad(c[0], bu*w, mad(c[1], bu*v, mad(c[2], bv*x, mad(c[3], bv*u,
           mad(c[4], bu*au*w, mad(c[5], bu*au*v, mad(c[6], bv*av*x, c[7]*bv*av*u)))))));
}

inline float3 eval_pos(const float3 P[4], const float3 Nd[4], const float c[8], float u, float v) {
    float um=1.f-u, vm=1.f-v;
    float3 bil = um*vm*P[0] + u*vm*P[1] + u*v*P[2] + um*v*P[3];
    float3 Nd_uv = um*vm*Nd[0] + u*vm*Nd[1] + u*v*Nd[2] + um*v*Nd[3];
    float nlen = length(Nd_uv);
    if (nlen > 1e-8f) Nd_uv /= nlen; else Nd_uv = (float3)(0,1,0);
    return bil + Nd_uv * deform_h(c, u, v);
}

inline float3 eval_n(const float3 P[4], const float3 Nd[4], const float c[8], float u, float v) {
    float um=1.f-u, vm=1.f-v;
    float3 Nd_uv = um*vm*Nd[0] + u*vm*Nd[1] + u*v*Nd[2] + um*v*Nd[3];
    float nlen = length(Nd_uv);
    if (nlen > 1e-8f) Nd_uv /= nlen; else Nd_uv = (float3)(0,1,0);
    float3 dPdu = (P[1]-P[0])*vm + (P[2]-P[3])*v;
    float3 dPdv = (P[3]-P[0])*um + (P[2]-P[1])*u;
    const float eps = 1e-3f;
    float up = clamp(u+eps,0.f,1.f), um2 = clamp(u-eps,0.f,1.f);
    float vp = clamp(v+eps,0.f,1.f), vm2 = clamp(v-eps,0.f,1.f);
    float dhu = (deform_h(c,up,v)-deform_h(c,um2,v))/max(up-um2,1e-6f);
    float dhv = (deform_h(c,u,vp)-deform_h(c,u,vm2))/max(vp-vm2,1e-6f);
    dPdu += Nd_uv * dhu; dPdv += Nd_uv * dhv;
    float3 n = cross(dPdu, dPdv);
    float nl = length(n);
    return nl > 1e-8f ? n/nl : Nd_uv;
}

inline float3 anchor_pos(__global const uchar *A, uint i, float3 bbmin, float3 bbspan) {
    __global const uchar *a = A + i*64;
    ushort px = *(__global const ushort*)(a+32);
    ushort py = *(__global const ushort*)(a+34);
    ushort pz = *(__global const ushort*)(a+36);
    return bbmin + (float3)(px,py,pz) * (1.f/65535.f) * bbspan;
}

inline float3 anchor_nd(__global const uchar *A, uint i) {
    __global const uchar *a = A + i*64;
    float3 n = (float3)(a[44],a[45],a[46]) * (2.f/255.f) - 1.f;
    float L = length(n);
    return L > 1e-6f ? n/L : (float3)(0,1,0);
}

// planes: 6 * float4 (nx,ny,nz,d), inside if dot(n,p)+d >= 0
inline int aabb_outside_frustum(float3 bmin, float3 bmax, __global const float4 *planes) {
    for (int i = 0; i < 6; ++i) {
        float4 pl = planes[i];
        float3 p = (float3)(
            pl.x >= 0.f ? bmax.x : bmin.x,
            pl.y >= 0.f ? bmax.y : bmin.y,
            pl.z >= 0.f ? bmax.z : bmin.z);
        if (dot(pl.xyz, p) + pl.w < 0.f) return 1;
    }
    return 0;
}

__kernel void psf_raymarch(
    __global const uchar *anchors,
    __global const uchar *surfaces,
    __global const float *patch_aabb,  // 6 floats per surface: bmin.xyz bmax.xyz
    const uint n_surfaces,
    __global const float4 *frustum_planes, // 6 planes
    const float3 bbmin,
    const float3 bbspan,
    const float3 eye,
    const float3 fwd,
    const float3 right,
    const float3 upv,
    const float tan_half,
    const float aspect,
    const float time_sec,
    const float emissive_strength,
    const float hit_radius,
    const int uvn_min,
    const int uvn_max,
    const uint object_id,
    const uint W,
    const uint H,
    __global uchar4 *out_rgba)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= W || y >= H) return;

    float ndc_x = (2.f*(x+0.5f)/W - 1.f) * aspect * tan_half;
    float ndc_y = (1.f - 2.f*(y+0.5f)/H) * tan_half;
    float3 rd = normalize(fwd + right*ndc_x + upv*ndc_y);

    float best_t = 1e30f, best_d = 1e30f;
    float3 best_n = (float3)(0,1,0);
    float3 best_alb = (float3)(0.5f), best_em = (float3)(0);

    for (uint si = 0; si < n_surfaces; ++si) {
        float3 pmin = (float3)(patch_aabb[si*6+0], patch_aabb[si*6+1], patch_aabb[si*6+2]);
        float3 pmax = (float3)(patch_aabb[si*6+3], patch_aabb[si*6+4], patch_aabb[si*6+5]);
        if (aabb_outside_frustum(pmin, pmax, frustum_planes)) continue;

        // Adaptive uvn on GPU: nearer + larger AABB → more equation probes
        float3 ctr = 0.5f * (pmin + pmax);
        float dist = max(length(ctr - eye), 0.15f);
        float extent = length(pmax - pmin);
        float screen_proxy = extent / dist; // perspective-ish
        int uv = uvn_min + (int)(screen_proxy * 3.5f);
        if (uv > uvn_max) uv = uvn_max;
        if (uv < uvn_min) uv = uvn_min;

        __global const uchar *s = surfaces + si*64;
        ushort a0 = *(__global const ushort*)(s+0);
        ushort a1 = *(__global const ushort*)(s+2);
        ushort a2 = *(__global const ushort*)(s+4);
        ushort a3 = *(__global const ushort*)(s+6);
        float3 P[4], Nd[4];
        P[0]=anchor_pos(anchors,a0,bbmin,bbspan); Nd[0]=anchor_nd(anchors,a0);
        P[1]=anchor_pos(anchors,a1,bbmin,bbspan); Nd[1]=anchor_nd(anchors,a1);
        P[2]=anchor_pos(anchors,a2,bbmin,bbspan); Nd[2]=anchor_nd(anchors,a2);
        P[3]=anchor_pos(anchors,a3,bbmin,bbspan); Nd[3]=anchor_nd(anchors,a3);
        float c[8];
        for (int k=0;k<8;++k) c[k] = *(__global const float*)(s+12+k*4);

        float hue = fmod((float)si * 0.07f, 1.f);
        float3 alb = (float3)(
            0.35f + 0.5f * clamp(fabs(fmod(hue,1.f)*6.f-3.f)-1.f, 0.f, 1.f),
            0.35f + 0.5f * clamp(fabs(fmod(hue+0.33f,1.f)*6.f-3.f)-1.f, 0.f, 1.f),
            0.35f + 0.5f * clamp(fabs(fmod(hue+0.66f,1.f)*6.f-3.f)-1.f, 0.f, 1.f));
        float energy = 0.f;
        for (int k=0;k<8;++k) energy += fabs(c[k]);
        float pulse = 0.5f + 0.5f * sin(time_sec*1.7f + (float)si*0.11f);
        float3 em = alb * (emissive_strength * (0.4f+0.6f*pulse) * (0.15f+energy));

        for (int iu=0; iu<uv; ++iu) {
            for (int iv=0; iv<uv; ++iv) {
                float uu = (iu+0.5f)/(float)uv;
                float vv = (iv+0.5f)/(float)uv;
                float3 Sp = eval_pos(P, Nd, c, uu, vv);
                float3 rel = Sp - eye;
                float t = dot(rel, rd);
                if (t < 1e-3f) continue;
                float3 closest = eye + rd * t;
                float d = length(Sp - closest);
                float rad = hit_radius * (0.65f + 0.35f * clamp(t * 0.35f, 0.f, 2.f));
                if (d > rad) continue;
                if (t < best_t - 1e-4f || (fabs(t-best_t)<1e-4f && d < best_d)) {
                    best_t = t; best_d = d;
                    best_n = eval_n(P, Nd, c, uu, vv);
                    if (dot(best_n, rd) > 0.f) best_n = -best_n;
                    best_alb = alb; best_em = em;
                }
            }
        }
    }

    if (best_t >= 1e29f) {
        out_rgba[y*W+x] = (uchar4)(10, 10, 16, 0);
        return;
    }
    float3 V = -rd;
    float3 col = best_em + best_alb * 0.07f;
    col += best_alb * (0.75f * max(0.f, dot(best_n, V)));
    // NOTE: half-vector renamed Hv (was `H`) -- `H` is the kernel's height
    // parameter (line 89); the old `float3 H` collided with it and broke
    // compilation, silently disabling the whole raymarch path.
    float3 Hv = normalize(V+V);
    col += (float3)(1.f,0.95f,0.9f) * (0.25f * pow(max(0.f, dot(best_n, Hv)), 32.f));
    float3 L = normalize((float3)(0.35f, 0.8f, 0.4f));
    col += best_alb * (0.45f * max(0.f, dot(best_n, L)));
    col = clamp(col, 0.f, 1.f);
    uchar oid = (uchar)(object_id == 0u ? 1u : (object_id > 255u ? 255u : object_id));
    out_rgba[y*W+x] = (uchar4)((uchar)(col.x*255.f+0.5f),(uchar)(col.y*255.f+0.5f),
                               (uchar)(col.z*255.f+0.5f), oid);
}
