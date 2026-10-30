// =============================================================================
// march_gpu_interactive.cpp -- Interactive SDL viewer for the progressive
// per-pixel surfel renderer (kernels/surfel_march.cl).
//
// This is march_gpu_render.cpp (the validated one-shot host) restructured
// into persistent-setup + per-frame-render, exactly mirroring the pattern
// already working in legacy/examples/04_spectral_live/main.cpp:
//   - context/queue/program/kernels/static buffers (surfels + 3D grid):
//     built ONCE (they don't change unless you load a different .srf)
//   - RenderFrame(yaw, pitch, dist): the camera + NLAYERS loop + compose +
//     readback, called every frame
//   - SDL2 window + streaming texture + event loop (drag=orbit, wheel=zoom)
//
// HONEST PERFORMANCE NOTE (flagged before writing a line of this): the
// per-frame algorithm still does NLAYERS=24 * (clear+draw+FILL_PASSES=4
// fills+2 copies) = ~192 kernel launches/frame, written for a single
// offline render, not 60fps. This file measures and PRINTS real ms/frame
// in the window title so you can see whether it's actually interactive on
// your hardware, instead of assuming. If it's too slow, the two documented
// levers (NEXT_STEPS_PLAN.md, frente 1) are: (a) drop NLAYERS/FILL_PASSES
// while dragging and restore them once the camera settles, or (b) make
// layer_compose incremental instead of storing all 24 layers.
//
//   g++ -std=c++17 tools/march_gpu_interactive.cpp -lOpenCL `sdl2-config --cflags --libs` -o march_interactive
//   ./march_interactive assets/spot/spot.srf
//
//   mouse drag = orbit    wheel = zoom    R = reset    ESC = quit
// =============================================================================
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <SDL2/SDL.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

static const int W=700, H=700;
static const int NLAYERS=24;
static const int BAND_STEPS=24;
static const int FILL_PASSES=4;

struct Surfel{ float px,py,pz,nx,ny,nz,r,g,b,radius; };
struct V3{ float x,y,z; };
static V3 sub(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 vcross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static float vdot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 vnorm(V3 v){float l=std::sqrt(vdot(v,v));return l>1e-9f?V3{v.x/l,v.y/l,v.z/l}:v;}

static std::string slurp(const char* p){
    std::ifstream f(p,std::ios::binary);
    if(!f){fprintf(stderr,"cannot open %s\n",p);exit(1);}
    return {(std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()};
}
#define CK(x) do{cl_int _e=(x);if(_e){printf("CL err %d at line %d\n",_e,__LINE__);exit(1);}}while(0)

// ---- persistent state (built once in Init, reused every frame) -----------
struct GpuState {
    cl_context ctx; cl_command_queue q; cl_program prog;
    cl_kernel kclear_grid, kbuild_grid, klayer_clear, klayer_draw, klayer_fill, kcompose;
    cl_mem bS, bgrid, ball_color, ball_depth, bcolA, bcolB, bdepA, bdepB, bout;
    int gx, gy, gz; V3 org; float cell;
    V3 cen; uint32_t n_surfels;
    cl_float3 gorg;
};

static GpuState Init(const char* srf_path) {
    GpuState S{};
    std::string buf = slurp(srf_path);
    if (memcmp(buf.data(), "SRF\0", 4)) { printf("bad .srf\n"); exit(1); }
    uint32_t n; memcpy(&n, buf.data()+4, 4);
    const Surfel* surf = reinterpret_cast<const Surfel*>(buf.data()+8);
    S.n_surfels = n;
    printf("loaded %u surfels from %s\n", n, srf_path);

    V3 lo{1e30f,1e30f,1e30f}, hi{-1e30f,-1e30f,-1e30f}, cen{0,0,0};
    for (uint32_t i=0;i<n;++i){
        lo.x=std::min(lo.x,surf[i].px); lo.y=std::min(lo.y,surf[i].py); lo.z=std::min(lo.z,surf[i].pz);
        hi.x=std::max(hi.x,surf[i].px); hi.y=std::max(hi.y,surf[i].py); hi.z=std::max(hi.z,surf[i].pz);
        cen.x+=surf[i].px; cen.y+=surf[i].py; cen.z+=surf[i].pz;
    }
    cen.x/=n; cen.y/=n; cen.z/=n; S.cen=cen;
    S.org = {lo.x-0.05f, lo.y-0.05f, lo.z-0.05f};
    S.cell = 0.008f;
    S.gx=(int)std::ceil((hi.x-lo.x+0.1f)/S.cell);
    S.gy=(int)std::ceil((hi.y-lo.y+0.1f)/S.cell);
    S.gz=(int)std::ceil((hi.z-lo.z+0.1f)/S.cell);
    printf("grid %dx%dx%d cell %.3f\n", S.gx, S.gy, S.gz, S.cell);

    cl_platform_id plat; CK(clGetPlatformIDs(1,&plat,nullptr));
    cl_device_id dev; CK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,nullptr));
    char dn[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,256,dn,nullptr);
    printf("GPU: %s\n", dn);
    cl_int err;
    S.ctx = clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err); CK(err);
#ifdef CL_VERSION_2_0
    cl_queue_properties qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    S.q = clCreateCommandQueueWithProperties(S.ctx,dev,qprops,&err); CK(err);
#else
    S.q = clCreateCommandQueue(S.ctx,dev,CL_QUEUE_PROFILING_ENABLE,&err); CK(err);
#endif
    std::string ksrc = slurp("kernels/surfel_march.cl");
    const char* kp = ksrc.c_str(); size_t kl = ksrc.size();
    S.prog = clCreateProgramWithSource(S.ctx,1,&kp,&kl,&err); CK(err);
    if (clBuildProgram(S.prog,1,&dev,"",nullptr,nullptr)!=CL_SUCCESS) {
        char log[16384]; clGetProgramBuildInfo(S.prog,dev,CL_PROGRAM_BUILD_LOG,16384,log,nullptr);
        printf("BUILD LOG:\n%s\n", log); exit(1);
    }
    S.kclear_grid  = clCreateKernel(S.prog,"clear_grid",&err);  CK(err);
    S.kbuild_grid  = clCreateKernel(S.prog,"build_grid",&err);  CK(err);
    S.klayer_clear = clCreateKernel(S.prog,"layer_clear",&err); CK(err);
    S.klayer_draw  = clCreateKernel(S.prog,"layer_draw",&err);  CK(err);
    S.klayer_fill  = clCreateKernel(S.prog,"layer_fill",&err);  CK(err);
    S.kcompose     = clCreateKernel(S.prog,"layer_compose",&err); CK(err);

    S.bS = clCreateBuffer(S.ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                          sizeof(Surfel)*n,(void*)surf,&err); CK(err);
    size_t ncell=(size_t)S.gx*S.gy*S.gz;
    S.bgrid = clCreateBuffer(S.ctx,CL_MEM_READ_WRITE,sizeof(cl_int)*ncell,nullptr,&err); CK(err);
    size_t npix=(size_t)W*H;
    S.ball_color = clCreateBuffer(S.ctx,CL_MEM_READ_WRITE,4*npix*NLAYERS,nullptr,&err); CK(err);
    S.ball_depth = clCreateBuffer(S.ctx,CL_MEM_READ_WRITE,sizeof(cl_float)*npix*NLAYERS,nullptr,&err); CK(err);
    S.bcolA = clCreateBuffer(S.ctx,CL_MEM_READ_WRITE,4*npix,nullptr,&err); CK(err);
    S.bcolB = clCreateBuffer(S.ctx,CL_MEM_READ_WRITE,4*npix,nullptr,&err); CK(err);
    S.bdepA = clCreateBuffer(S.ctx,CL_MEM_READ_WRITE,sizeof(cl_float)*npix,nullptr,&err); CK(err);
    S.bdepB = clCreateBuffer(S.ctx,CL_MEM_READ_WRITE,sizeof(cl_float)*npix,nullptr,&err); CK(err);
    S.bout  = clCreateBuffer(S.ctx,CL_MEM_WRITE_ONLY,4*npix,nullptr,&err); CK(err);

    cl_uint nc=(cl_uint)ncell;
    CK(clSetKernelArg(S.kclear_grid,0,sizeof(S.bgrid),&S.bgrid));
    CK(clSetKernelArg(S.kclear_grid,1,sizeof(nc),&nc));
    size_t gc=ncell; CK(clEnqueueNDRangeKernel(S.q,S.kclear_grid,1,nullptr,&gc,nullptr,0,nullptr,nullptr));
    S.gorg = {{S.org.x,S.org.y,S.org.z}};
    int a=0;
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(S.bS),&S.bS));
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(n),&n));
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(S.bgrid),&S.bgrid));
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(S.gx),&S.gx));
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(S.gy),&S.gy));
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(S.gz),&S.gz));
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(S.gorg),&S.gorg));
    CK(clSetKernelArg(S.kbuild_grid,a++,sizeof(S.cell),&S.cell));
    size_t gn=n; CK(clEnqueueNDRangeKernel(S.q,S.kbuild_grid,1,nullptr,&gn,nullptr,0,nullptr,nullptr));
    return S;
}

// per-frame device-time breakdown (OpenCL event profiling), separate from
// the host-side wall clock already measured around RenderFrame() -- this
// is what lets us tell apart "the GPU kernel itself took longer this
// frame" (real device work variance) from "something stalled the host
// between enqueues" (driver/OS/CPU scheduling, not this algorithm).
struct FrameStats {
    double draw_us = 0, fill_us = 0, other_us = 0, readback_us = 0;
};

static double EventUs(cl_event ev) {
    cl_ulong t0, t1;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
    return (t1 - t0) / 1000.0;   // ns -> us
}

// ---- per-frame render: camera from orbit state -> RGBA8 image ------------
static std::vector<unsigned char> RenderFrame(GpuState& S, float yaw, float pitch, float dist,
                                              FrameStats* stats = nullptr) {
    V3 eye{S.cen.x+dist*std::cos(pitch)*std::sin(yaw),
           S.cen.y-dist*std::cos(pitch)*std::cos(yaw),
           S.cen.z+dist*std::sin(pitch)};
    V3 fwd=vnorm(sub(S.cen,eye));
    V3 right=vnorm(vcross(fwd,V3{0,0,1}));
    V3 up=vcross(right,fwd);
    float tanf=std::tan(50.0f*3.14159265f/360.0f);
    float dmin=dist-1.2f, dmax=dist+1.2f;
    float layer_thick=(dmax-dmin)/NLAYERS;
    V3 L=vnorm(V3{0.4f,-0.5f,0.6f});
    cl_float3 ceye={{eye.x,eye.y,eye.z}}, cfwd={{fwd.x,fwd.y,fwd.z}},
              cright={{right.x,right.y,right.z}}, cup={{up.x,up.y,up.z}},
              clight={{L.x,L.y,L.z}};
    cl_uint w=W,h=H; const float INF=1e30f;
    size_t gws2[2]={(size_t)W,(size_t)H}; size_t npix=(size_t)W*H;

    for (int li=0; li<NLAYERS; ++li) {
        float d_far = dmax - li*layer_thick;
        float d_near = d_far - layer_thick;
        float band_depth = (d_far+d_near)*0.5f;
        cl_uint np2=(cl_uint)npix;
        CK(clSetKernelArg(S.klayer_clear,0,sizeof(S.bcolA),&S.bcolA));
        CK(clSetKernelArg(S.klayer_clear,1,sizeof(S.bdepA),&S.bdepA));
        CK(clSetKernelArg(S.klayer_clear,2,sizeof(np2),&np2));
        CK(clSetKernelArg(S.klayer_clear,3,sizeof(INF),&INF));
        cl_event ev_clear;
        CK(clEnqueueNDRangeKernel(S.q,S.klayer_clear,1,nullptr,&npix,nullptr,0,nullptr,&ev_clear));
        if (stats) { clWaitForEvents(1,&ev_clear); stats->other_us += EventUs(ev_clear); }
        clReleaseEvent(ev_clear);

        int a=0;
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.bS),&S.bS));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.bgrid),&S.bgrid));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.gx),&S.gx));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.gy),&S.gy));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.gz),&S.gz));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.gorg),&S.gorg));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.cell),&S.cell));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(ceye),&ceye));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(cfwd),&cfwd));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(cright),&cright));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(cup),&cup));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(tanf),&tanf));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(w),&w));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(h),&h));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(d_far),&d_far));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(d_near),&d_near));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(BAND_STEPS),&BAND_STEPS));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(band_depth),&band_depth));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(clight),&clight));  // fix del hilo anterior
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.bcolA),&S.bcolA));
        CK(clSetKernelArg(S.klayer_draw,a++,sizeof(S.bdepA),&S.bdepA));
        cl_event ev_draw;
        CK(clEnqueueNDRangeKernel(S.q,S.klayer_draw,2,nullptr,gws2,nullptr,0,nullptr,&ev_draw));
        if (stats) { clWaitForEvents(1,&ev_draw); stats->draw_us += EventUs(ev_draw); }
        clReleaseEvent(ev_draw);

        cl_mem curC=S.bcolA,curD=S.bdepA,scrC=S.bcolB,scrD=S.bdepB;
        for (int fp=0; fp<FILL_PASSES; ++fp) {
            a=0;
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(curC),&curC));
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(curD),&curD));
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(scrC),&scrC));
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(scrD),&scrD));
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(w),&w));
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(h),&h));
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(INF),&INF));
            CK(clSetKernelArg(S.klayer_fill,a++,sizeof(band_depth),&band_depth));
            cl_event ev_fill;
            CK(clEnqueueNDRangeKernel(S.q,S.klayer_fill,2,nullptr,gws2,nullptr,0,nullptr,&ev_fill));
            if (stats) { clWaitForEvents(1,&ev_fill); stats->fill_us += EventUs(ev_fill); }
            clReleaseEvent(ev_fill);
            std::swap(curC,scrC); std::swap(curD,scrD);
        }
        size_t col_off=(size_t)li*npix*4, dep_off=(size_t)li*npix*sizeof(cl_float);
        CK(clEnqueueCopyBuffer(S.q,curC,S.ball_color,0,col_off,npix*4,0,nullptr,nullptr));
        CK(clEnqueueCopyBuffer(S.q,curD,S.ball_depth,0,dep_off,npix*sizeof(cl_float),0,nullptr,nullptr));
    }

    cl_uchar4 bg={{15,15,18,255}}; cl_uint nl=NLAYERS;
    int a=0;
    CK(clSetKernelArg(S.kcompose,a++,sizeof(S.ball_color),&S.ball_color));
    CK(clSetKernelArg(S.kcompose,a++,sizeof(S.ball_depth),&S.ball_depth));
    CK(clSetKernelArg(S.kcompose,a++,sizeof(nl),&nl));
    CK(clSetKernelArg(S.kcompose,a++,sizeof(w),&w));
    CK(clSetKernelArg(S.kcompose,a++,sizeof(h),&h));
    CK(clSetKernelArg(S.kcompose,a++,sizeof(INF),&INF));
    CK(clSetKernelArg(S.kcompose,a++,sizeof(bg),&bg));
    CK(clSetKernelArg(S.kcompose,a++,sizeof(S.bout),&S.bout));
    cl_event ev_compose;
    CK(clEnqueueNDRangeKernel(S.q,S.kcompose,2,nullptr,gws2,nullptr,0,nullptr,&ev_compose));
    if (stats) { clWaitForEvents(1,&ev_compose); stats->other_us += EventUs(ev_compose); }
    clReleaseEvent(ev_compose);

    std::vector<unsigned char> img(4*npix);
    auto rb0 = std::chrono::high_resolution_clock::now();
    CK(clEnqueueReadBuffer(S.q,S.bout,CL_TRUE,0,4*npix,img.data(),0,nullptr,nullptr));
    auto rb1 = std::chrono::high_resolution_clock::now();
    if (stats) stats->readback_us += std::chrono::duration<double, std::micro>(rb1-rb0).count();
    return img;
}

int main(int argc, char** argv) {
    const char* srf = argc>1 ? argv[1] : "assets/spot/spot.srf";
    // benchmark mode: ./march_interactive model.srf --bench 60
    // runs N frames headless (no window), prints per-frame ms + summary,
    // so you can just paste the terminal output back instead of reading
    // fps off a window title in a screenshot.
    bool bench = false; int bench_frames = 60; bool profile = false;
    for (int i=2;i<argc;++i) {
        if (std::string(argv[i])=="--bench") {
            bench = true;
            if (i+1<argc) bench_frames = std::atoi(argv[i+1]);
        }
        if (std::string(argv[i])=="--profile") profile = true;
    }
    GpuState S = Init(srf);

    if (bench) {
        printf("=== BENCH MODE: %d frames, fixed camera path, no window ===\n",
               bench_frames);
        if (profile)
            printf("(--profile activo: agrega clWaitForEvents por kernel para medir "
                   "tiempo de DEVICE por separado -- esto mismo SINCRONIZA el pipeline "
                   "y va a inflar el ms/frame total vs una corrida sin --profile. Sirve "
                   "para ver DONDE se va el tiempo en un frame lento, no para medir fps "
                   "real -- para eso, corré sin --profile.)\n");
        std::vector<double> ms_all;
        std::vector<FrameStats> st_all;
        for (int f=0; f<bench_frames; ++f) {
            float yaw = 0.7f + 0.02f*f, pitch = 0.32f, dist = 2.6f;
            FrameStats st;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto img = RenderFrame(S, yaw, pitch, dist, profile ? &st : nullptr);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
            ms_all.push_back(ms); st_all.push_back(st);
            if (profile) {
                double accounted = (st.draw_us+st.fill_us+st.other_us+st.readback_us)/1000.0;
                double host_overhead = ms - accounted;
                printf("frame %2d: %7.2f ms total | device: draw %6.2f ms  fill %6.2f ms  "
                       "other %5.2f ms  readback %5.2f ms | host_overhead %6.2f ms\n",
                       f, ms, st.draw_us/1000.0, st.fill_us/1000.0, st.other_us/1000.0,
                       st.readback_us/1000.0, host_overhead);
            } else {
                printf("frame %2d: %7.2f ms  (%.1f fps instantaneo)\n", f, ms, 1000.0/ms);
            }
        }
        double sum=0, worst=0, best=1e9;
        for (double m : ms_all) { sum+=m; worst=std::max(worst,m); best=std::min(best,m); }
        double avg = sum/ms_all.size();
        printf("\n=== RESUMEN (pegame esto) ===\n");
        printf("frames: %d | avg %.2f ms (%.1f fps) | best %.2f ms | worst %.2f ms | "
               "surfels %u\n", (int)ms_all.size(), avg, 1000.0/avg, best, worst, S.n_surfels);
        return 0;

    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL init: %s\n", SDL_GetError()); return 1; }
    SDL_Window* win = SDL_CreateWindow("DeliRium Surfel March (live)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_SHOWN);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING, W, H);

    float yaw=0.7f, pitch=0.32f, dist=2.6f;
    const float base_yaw=0.7f, base_pitch=0.32f, base_dist=2.6f;
    bool dragging=false; int lastx=0, lasty=0;

    bool running=true;
    Uint32 last=SDL_GetTicks(); int frames=0; float fps=0; double last_ms=0;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type==SDL_QUIT) running=false;
            else if (e.type==SDL_KEYDOWN) {
                SDL_Keycode k=e.key.keysym.sym;
                if (k==SDLK_ESCAPE) running=false;
                else if (k==SDLK_r) { yaw=base_yaw; pitch=base_pitch; dist=base_dist; }
            } else if (e.type==SDL_MOUSEBUTTONDOWN) {
                dragging=true; lastx=e.button.x; lasty=e.button.y;
            } else if (e.type==SDL_MOUSEBUTTONUP) dragging=false;
            else if (e.type==SDL_MOUSEMOTION && dragging) {
                yaw   += (e.motion.x-lastx)*0.01f;
                pitch += (e.motion.y-lasty)*0.01f;
                pitch = std::fmax(-1.5f, std::fmin(1.5f, pitch));
                lastx=e.motion.x; lasty=e.motion.y;
            } else if (e.type==SDL_MOUSEWHEEL)
                dist = std::fmax(0.4f, dist - e.wheel.y*0.2f);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<unsigned char> img = RenderFrame(S, yaw, pitch, dist);
        auto t1 = std::chrono::high_resolution_clock::now();
        last_ms = std::chrono::duration<double, std::milli>(t1-t0).count();

        SDL_UpdateTexture(tex, nullptr, img.data(), W*4);
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, nullptr, nullptr);
        SDL_RenderPresent(ren);

        if (++frames >= 10) {
            Uint32 now = SDL_GetTicks();
            fps = frames*1000.0f/(now-last);
            last = now; frames = 0;
            char title[256];
            std::snprintf(title, sizeof title,
                "DeliRium Surfel March | %.1f fps | %.1f ms/frame | "
                "dist %.2f | %u surfels",
                fps, last_ms, dist, S.n_surfels);
            SDL_SetWindowTitle(win, title);
        }
    }

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
