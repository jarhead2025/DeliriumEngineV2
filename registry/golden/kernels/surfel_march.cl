// =============================================================================
// surfel_march.cl  -  Progressive per-pixel rasterization, DeliRium engine.
//
// USER'S METHOD: each pixel marches its ray far->near through FOV-planes
// (a square of half-size d*tan(FOV/2) mapped 1:1 to screen at depth d).
// The model's surface is stored in a 3D uniform grid for O(1) lookup.
//
// LAYER PATTERN (what the user asked for):
//   for each depth band (layer), far->near:
//     1. layer_draw  - draw that band's hits into the layer's OWN buffer
//     2. layer_fill  - fill holes AGGRESSIVELY in that layer's own buffer
//                      (multiple passes, no other layer interferes)
//     3. save the filled layer
//   at the end, layer_compose picks the nearest layer per pixel (min depth).
//
// FIX (grid-quantization holes): layer_draw used to look up ONLY the exact
// grid cell a sample point falls into. build_grid stores a single surfel
// index per cell with no atomic "nearest" resolution, so a surfel sitting
// near a cell boundary can be recorded in the cell NEXT to a sample point
// that is geometrically very close to it. The single-cell lookup then
// misses it outright -> isolated black speckle holes on smooth/curved
// regions, even after the aggressive same-layer neighbor fill (which only
// helps when >=3 neighboring PIXELS are already filled). The fix mirrors
// what kernels/surfel_inverse.cl already does for its screen-space grid:
// search the 3x3x3 neighborhood of cells and keep the closest surfel
// within the pixel's footprint, instead of trusting one cell blindly.
// =============================================================================
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

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

// --- per-layer kernels -------------------------------------------------------

// clear one layer's color+depth buffers
__kernel void layer_clear(
    __global uchar4* color, __global float* depth,
    const uint n, const float INF)
{
    uint i=get_global_id(0);
    if(i>=n) return;
    color[i]=(uchar4)(0,0,0,0);
    depth[i]=INF;
}

// draw this band [d_far..d_near] into the layer's OWN clean buffer
__kernel void layer_draw(
    __global const Surfel* S,
    __global const int*    grid,
    const int gx,const int gy,const int gz,
    const float3 org, const float cell,
    const float3 eye, const float3 fwd,
    const float3 right, const float3 up,
    const float tanfov,
    const uint W, const uint H,
    const float d_far, const float d_near,
    const int bsteps,
    const float band_depth,          // depth assigned to filled pixels
    const float3 light,
    __global uchar4* color, __global float* depth)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    float ndcx=((x+0.5f)/W-0.5f)*2.0f;
    float ndcy=(0.5f-(y+0.5f)/H)*2.0f;
    float step=(d_far-d_near)/bsteps;
    for(int s=0;s<bsteps;++s){
        float d=d_far-s*step;
        if(d>=depth[pid]) continue;       // already have closer hit
        float hsize=d*tanfov;
        float3 p=eye+fwd*d+right*(ndcx*hsize)+up*(ndcy*hsize);
        int cx=(int)((p.x-org.x)/cell);
        int cy=(int)((p.y-org.y)/cell);
        int cz=(int)((p.z-org.z)/cell);
        if(cx<0||cx>=gx||cy<0||cy>=gy||cz<0||cz>=gz) continue;

        // search the 3x3x3 neighborhood: a surfel geometrically close to
        // p can be quantized into a NEIGHBOR cell, not just this one.
        // Keep the closest surfel found within the pixel footprint.
        float foot=(2.0f*hsize/W)*1.5f;
        float best_d2=foot*foot;
        int best_si=-1;
        for(int dz=-1;dz<=1;++dz){
            int ncz=cz+dz; if(ncz<0||ncz>=gz) continue;
            for(int dy=-1;dy<=1;++dy){
                int ncy=cy+dy; if(ncy<0||ncy>=gy) continue;
                for(int dx=-1;dx<=1;++dx){
                    int ncx=cx+dx; if(ncx<0||ncx>=gx) continue;
                    int si=grid[(ncz*gy+ncy)*gx+ncx]; if(si<0) continue;
                    float ddx=S[si].px-p.x, ddy=S[si].py-p.y, ddz=S[si].pz-p.z;
                    float d2=ddx*ddx+ddy*ddy+ddz*ddz;
                    if(d2<best_d2){ best_d2=d2; best_si=si; }
                }
            }
        }
        if(best_si>=0){
            Surfel ss=S[best_si];
            float3 n=(float3)(ss.nx,ss.ny,ss.nz);
            float lam=clamp(dot(n,light),0.f,1.f)*0.25f+0.75f;
            color[pid]=(uchar4)((uchar)(clamp(ss.r*lam,0.f,1.f)*255),
                                (uchar)(clamp(ss.g*lam,0.f,1.f)*255),
                                (uchar)(clamp(ss.b*lam,0.f,1.f)*255),255);
            depth[pid]=d;
            return;
        }
    }
}

// fill holes aggressively in THIS layer's own buffer (ping-pong).
// empty pixel with >=3 filled neighbors takes their average color.
// depth of filled pixel = band_depth (so compose knows this layer's depth).
__kernel void layer_fill(
    __global const uchar4* cin, __global const float* din,
    __global uchar4* cout,      __global float* dout,
    const uint W, const uint H,
    const float INF, const float band_depth)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    cout[pid]=cin[pid]; dout[pid]=din[pid];   // copy through
    if(din[pid]<INF) return;                  // already filled
    if(x==0||x==W-1||y==0||y==H-1) return;
    int R=0,G=0,B=0,cnt=0;
    for(int dy=-1;dy<=1;++dy)
    for(int dx=-1;dx<=1;++dx){
        if(dx==0&&dy==0) continue;
        uint n=(y+dy)*W+(x+dx);
        if(din[n]<INF){ uchar4 c=cin[n]; R+=c.x;G+=c.y;B+=c.z; cnt++; }
    }
    // Umbral bajado de >=3 a >=2 (verificado: no hay otro uso del campo
    // 'radius' en este kernel que pueda explicar/arreglar huecos chicos
    // residuales -- este es el unico lugar que realmente cierra huecos
    // aislados). TRADE-OFF HONESTO: mas permisivo tambien puede engordar
    // 1px el contorno de la silueta contra el fondo en bordes reales. Si
    // eso se nota, subir de nuevo a 3 y buscar la causa en la densidad de
    // muestreo del bake en vez de en el fill.
    if(cnt>=2){
        cout[pid]=(uchar4)((uchar)(R/cnt),(uchar)(G/cnt),(uchar)(B/cnt),255);
        dout[pid]=band_depth;
    }
}

// compose all layers: per pixel, the nearest layer (min depth) wins.
__kernel void layer_compose(
    __global const uchar4* layers_color,   // all layers packed: layer i at i*W*H
    __global const float*  layers_depth,
    const uint nlayers, const uint W, const uint H,
    const float INF, const uchar4 bg,
    __global uchar4* out)
{
    uint x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H) return;
    uint pid=y*W+x;
    float best=INF; uchar4 bestc=bg;
    for(uint li=0;li<nlayers;++li){
        uint idx=li*W*H+pid;
        float d=layers_depth[idx];
        if(d<best){ best=d; bestc=layers_color[idx]; }
    }
    out[pid]=bestc;
}
