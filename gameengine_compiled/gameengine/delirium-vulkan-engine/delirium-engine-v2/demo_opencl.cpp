// demo_opencl.cpp — v2 BodyRuntime → bake surfels → proven OpenCL renderer
// Integrates v2's body generation with the proven surfel_march.cl pipeline
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
#include <algorithm>

#include "core/types.h"
#include "core/body_topology.h"
#include "core/body_state.h"
#include "core/basis_functions.h"
#include "core/body_runtime.h"
#include "core/ai/generator.h"
#include "core/ai/domain_classifier.h"

using namespace delirium;

// ── Surfel for .srf format (matches proven pipeline exactly) ─────────
struct SrfSurfel { float px,py,pz,nx,ny,nz,r,g,b,radius; };
struct V3 { float x,y,z; };
static V3 sub(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 vcross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static float vdot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 vnorm(V3 v){float l=std::sqrt(vdot(v,v));return l>1e-9f?V3{v.x/l,v.y/l,v.z/l}:v;}
#define CK(x) do{cl_int _e=(x);if(_e){printf("CL err %d at line %d\n",_e,__LINE__);exit(1);}}while(0)

// ── Bake surfels from v2 BodyRuntime ─────────────────────────────────
std::vector<SrfSurfel> bake_surfels(BodyRuntime& body, int grid=24) {
    std::vector<SrfSurfel> surfels;
    auto& state = body.read_state();
    auto& topology = body.topology();
    auto& materials = body.materials();

    // Bounding box for radius scaling
    V3 lo{1e30f,1e30f,1e30f}, hi{-1e30f,-1e30f,-1e30f};
    for (AnchorId ai = 0; ai < topology.anchor_count; ++ai) {
        auto& ar = state.anchors[ai];
        lo.x = std::min(lo.x, ar.pos[0]); lo.y = std::min(lo.y, ar.pos[1]); lo.z = std::min(lo.z, ar.pos[2]);
        hi.x = std::max(hi.x, ar.pos[0]); hi.y = std::max(hi.y, ar.pos[1]); hi.z = std::max(hi.z, ar.pos[2]);
    }
    float diag = vnorm(sub(hi, lo)).x; // placeholder, compute properly
    {
        V3 d = sub(hi, lo);
        diag = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    }

    for (PatchId pid = 0; pid < topology.patch_count; ++pid) {
        auto& ps = state.patches[pid];
        auto& pt = topology.patches[pid];

        // Patch extent for adaptive grid
        V3 plo{1e30f,1e30f,1e30f}, phi{-1e30f,-1e30f,-1e30f};
        for (int c = 0; c < 4; ++c) {
            auto& ar = state.anchors[pt.anchor_indices[c]];
            plo.x = std::min(plo.x, ar.pos[0]); plo.y = std::min(plo.y, ar.pos[1]); plo.z = std::min(plo.z, ar.pos[2]);
            phi.x = std::max(phi.x, ar.pos[0]); phi.y = std::max(phi.y, ar.pos[1]); phi.z = std::max(phi.z, ar.pos[2]);
        }
        V3 pd = sub(phi, plo);
        float extent = std::sqrt(pd.x*pd.x + pd.y*pd.y + pd.z*pd.z);
        int g = std::max(4, std::min(grid, (int)(extent / diag * 40)));

        // Material color
        uint8_t mr = 200, mg = 200, mb = 200;
        if (ps.material_id < materials.size()) {
            mr = materials[ps.material_id].r;
            mg = materials[ps.material_id].g;
            mb = materials[ps.material_id].b;
        }

        for (int i = 0; i < g; ++i) {
            for (int j = 0; j < g; ++j) {
                float u = (float)i / (g - 1);
                float v = (float)j / (g - 1);
                SurfacePoint sp = body.evaluate_surface(pid, u, v);

                // Radius = spacing * 0.9
                float spacing = extent / (g - 1);
                float radius = spacing * 0.9f;

                surfels.push_back({
                    sp.position.x, sp.position.y, sp.position.z,
                    sp.normal.x, sp.normal.y, sp.normal.z,
                    (float)mr / 255.0f, (float)mg / 255.0f, (float)mb / 255.0f,
                    radius
                });
            }
        }
    }
    return surfels;
}

// ── OpenCL renderer (proven pipeline from march_gpu_interactive.cpp) ──
static const int W = 700, H = 700;
static const int NLAYERS = 24;
static const int BAND_STEPS = 24;
static const int FILL_PASSES = 4;

struct GpuState {
    cl_context ctx; cl_command_queue q; cl_program prog;
    cl_kernel kclear_grid, kbuild_grid, klayer_clear, klayer_draw, klayer_fill, kcompose;
    cl_mem bS, bgrid, ball_color, ball_depth, bcolA, bcolB, bdepA, bdepB, bout;
    int gx, gy, gz; V3 org; float cell;
    V3 cen; uint32_t n_surfels;
    cl_float3 gorg;
};

static GpuState InitGpu(const std::vector<SrfSurfel>& surfels) {
    GpuState S{};
    S.n_surfels = (uint32_t)surfels.size();
    printf("[GPU] %u surfels\n", S.n_surfels);

    V3 lo{1e30f,1e30f,1e30f}, hi{-1e30f,-1e30f,-1e30f}, cen{0,0,0};
    for (uint32_t i = 0; i < S.n_surfels; ++i) {
        lo.x = std::min(lo.x, surfels[i].px); lo.y = std::min(lo.y, surfels[i].py); lo.z = std::min(lo.z, surfels[i].pz);
        hi.x = std::max(hi.x, surfels[i].px); hi.y = std::max(hi.y, surfels[i].py); hi.z = std::max(hi.z, surfels[i].pz);
        cen.x += surfels[i].px; cen.y += surfels[i].py; cen.z += surfels[i].pz;
    }
    cen.x /= S.n_surfels; cen.y /= S.n_surfels; cen.z /= S.n_surfels;
    S.cen = cen;
    S.org = {lo.x - 0.05f, lo.y - 0.05f, lo.z - 0.05f};
    S.cell = 0.008f;
    S.gx = (int)std::ceil((hi.x - lo.x + 0.1f) / S.cell);
    S.gy = (int)std::ceil((hi.y - lo.y + 0.1f) / S.cell);
    S.gz = (int)std::ceil((hi.z - lo.z + 0.1f) / S.cell);
    printf("[GPU] grid %dx%dx%d cell %.3f\n", S.gx, S.gy, S.gz, S.cell);

    cl_platform_id plat; CK(clGetPlatformIDs(1, &plat, nullptr));
    cl_device_id dev; CK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, 256, dn, nullptr);
    printf("[GPU] %s\n", dn);
    cl_int err;
    S.ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
#ifdef CL_VERSION_2_0
    S.q = clCreateCommandQueueWithProperties(S.ctx, dev, nullptr, &err); CK(err);
#else
    S.q = clCreateCommandQueue(S.ctx, dev, 0, &err); CK(err);
#endif

    // Load kernel source from proven delirium-engine
    std::ifstream kf("kernels/surfel_march.cl", std::ios::binary);
    if (!kf) { printf("[ERROR] Cannot open kernels/surfel_march.cl\n"); exit(1); }
    std::string ksrc((std::istreambuf_iterator<char>(kf)), std::istreambuf_iterator<char>());
    const char* kp = ksrc.c_str(); size_t kl = ksrc.size();
    S.prog = clCreateProgramWithSource(S.ctx, 1, &kp, &kl, &err); CK(err);
    if (clBuildProgram(S.prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
        char log[16384]; clGetProgramBuildInfo(S.prog, dev, CL_PROGRAM_BUILD_LOG, 16384, log, nullptr);
        printf("[BUILD LOG]\n%s\n", log); exit(1);
    }
    S.kclear_grid  = clCreateKernel(S.prog, "clear_grid", &err);  CK(err);
    S.kbuild_grid  = clCreateKernel(S.prog, "build_grid", &err);  CK(err);
    S.klayer_clear = clCreateKernel(S.prog, "layer_clear", &err); CK(err);
    S.klayer_draw  = clCreateKernel(S.prog, "layer_draw", &err);  CK(err);
    S.klayer_fill  = clCreateKernel(S.prog, "layer_fill", &err);  CK(err);
    S.kcompose     = clCreateKernel(S.prog, "layer_compose", &err); CK(err);

    S.bS = clCreateBuffer(S.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(SrfSurfel) * S.n_surfels, (void*)surfels.data(), &err); CK(err);
    size_t ncell = (size_t)S.gx * S.gy * S.gz;
    S.bgrid = clCreateBuffer(S.ctx, CL_MEM_READ_WRITE, sizeof(cl_int) * ncell, nullptr, &err); CK(err);
    size_t npix = (size_t)W * H;
    S.ball_color = clCreateBuffer(S.ctx, CL_MEM_READ_WRITE, 4 * npix * NLAYERS, nullptr, &err); CK(err);
    S.ball_depth = clCreateBuffer(S.ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * npix * NLAYERS, nullptr, &err); CK(err);
    S.bcolA = clCreateBuffer(S.ctx, CL_MEM_READ_WRITE, 4 * npix, nullptr, &err); CK(err);
    S.bcolB = clCreateBuffer(S.ctx, CL_MEM_READ_WRITE, 4 * npix, nullptr, &err); CK(err);
    S.bdepA = clCreateBuffer(S.ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * npix, nullptr, &err); CK(err);
    S.bdepB = clCreateBuffer(S.ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * npix, nullptr, &err); CK(err);
    S.bout  = clCreateBuffer(S.ctx, CL_MEM_WRITE_ONLY, 4 * npix, nullptr, &err); CK(err);

    cl_uint nc = (cl_uint)ncell;
    CK(clSetKernelArg(S.kclear_grid, 0, sizeof(S.bgrid), &S.bgrid));
    CK(clSetKernelArg(S.kclear_grid, 1, sizeof(nc), &nc));
    size_t gc = ncell; CK(clEnqueueNDRangeKernel(S.q, S.kclear_grid, 1, nullptr, &gc, nullptr, 0, nullptr, nullptr));
    S.gorg = {{S.org.x, S.org.y, S.org.z}};
    int a = 0;
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.bS), &S.bS));
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.n_surfels), &S.n_surfels));
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.bgrid), &S.bgrid));
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.gx), &S.gx));
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.gy), &S.gy));
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.gz), &S.gz));
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.gorg), &S.gorg));
    CK(clSetKernelArg(S.kbuild_grid, a++, sizeof(S.cell), &S.cell));
    size_t gn = S.n_surfels; CK(clEnqueueNDRangeKernel(S.q, S.kbuild_grid, 1, nullptr, &gn, nullptr, 0, nullptr, nullptr));
    return S;
}

static std::vector<unsigned char> RenderFrame(GpuState& S, float yaw, float pitch, float dist) {
    V3 eye{S.cen.x + dist * std::cos(pitch) * std::sin(yaw),
           S.cen.y - dist * std::cos(pitch) * std::cos(yaw),
           S.cen.z + dist * std::sin(pitch)};
    V3 fwd = vnorm(sub(S.cen, eye));
    V3 right = vnorm(vcross(fwd, V3{0, 0, 1}));
    V3 up = vcross(right, fwd);
    float tanf = std::tan(50.0f * 3.14159265f / 360.0f);
    float dmin = dist - 1.2f, dmax = dist + 1.2f;
    float layer_thick = (dmax - dmin) / NLAYERS;
    V3 L = vnorm(V3{0.4f, -0.5f, 0.6f});
    cl_float3 ceye = {{eye.x, eye.y, eye.z}}, cfwd = {{fwd.x, fwd.y, fwd.z}},
              cright = {{right.x, right.y, right.z}}, cup = {{up.x, up.y, up.z}},
              clight = {{L.x, L.y, L.z}};
    cl_uint w = W, h = H; const float INF = 1e30f;
    size_t gws2[2] = {(size_t)W, (size_t)H}; size_t npix = (size_t)W * H;

    for (int li = 0; li < NLAYERS; ++li) {
        float d_far = dmax - li * layer_thick;
        float d_near = d_far - layer_thick;
        float band_depth = (d_far + d_near) * 0.5f;
        cl_uint np2 = (cl_uint)npix;
        CK(clSetKernelArg(S.klayer_clear, 0, sizeof(S.bcolA), &S.bcolA));
        CK(clSetKernelArg(S.klayer_clear, 1, sizeof(S.bdepA), &S.bdepA));
        CK(clSetKernelArg(S.klayer_clear, 2, sizeof(np2), &np2));
        CK(clSetKernelArg(S.klayer_clear, 3, sizeof(INF), &INF));
        CK(clEnqueueNDRangeKernel(S.q, S.klayer_clear, 1, nullptr, &npix, nullptr, 0, nullptr, nullptr));

        int a = 0;
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.bS), &S.bS));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.bgrid), &S.bgrid));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.gx), &S.gx));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.gy), &S.gy));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.gz), &S.gz));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.gorg), &S.gorg));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.cell), &S.cell));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(ceye), &ceye));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(cfwd), &cfwd));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(cright), &cright));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(cup), &cup));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(tanf), &tanf));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(w), &w));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(h), &h));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(d_far), &d_far));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(d_near), &d_near));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(BAND_STEPS), &BAND_STEPS));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(band_depth), &band_depth));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(clight), &clight));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.bcolA), &S.bcolA));
        CK(clSetKernelArg(S.klayer_draw, a++, sizeof(S.bdepA), &S.bdepA));
        CK(clEnqueueNDRangeKernel(S.q, S.klayer_draw, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr));

        cl_mem curC = S.bcolA, curD = S.bdepA, scrC = S.bcolB, scrD = S.bdepB;
        for (int fp = 0; fp < FILL_PASSES; ++fp) {
            a = 0;
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(curC), &curC));
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(curD), &curD));
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(scrC), &scrC));
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(scrD), &scrD));
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(w), &w));
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(h), &h));
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(INF), &INF));
            CK(clSetKernelArg(S.klayer_fill, a++, sizeof(band_depth), &band_depth));
            CK(clEnqueueNDRangeKernel(S.q, S.klayer_fill, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr));
            std::swap(curC, scrC); std::swap(curD, scrD);
        }
        size_t col_off = (size_t)li * npix * 4, dep_off = (size_t)li * npix * sizeof(cl_float);
        CK(clEnqueueCopyBuffer(S.q, curC, S.ball_color, 0, col_off, npix * 4, 0, nullptr, nullptr));
        CK(clEnqueueCopyBuffer(S.q, curD, S.ball_depth, 0, dep_off, npix * sizeof(cl_float), 0, nullptr, nullptr));
    }

    cl_uchar4 bg = {{15, 15, 18, 255}}; cl_uint nl = NLAYERS;
    int a = 0;
    CK(clSetKernelArg(S.kcompose, a++, sizeof(S.ball_color), &S.ball_color));
    CK(clSetKernelArg(S.kcompose, a++, sizeof(S.ball_depth), &S.ball_depth));
    CK(clSetKernelArg(S.kcompose, a++, sizeof(nl), &nl));
    CK(clSetKernelArg(S.kcompose, a++, sizeof(w), &w));
    CK(clSetKernelArg(S.kcompose, a++, sizeof(h), &h));
    CK(clSetKernelArg(S.kcompose, a++, sizeof(INF), &INF));
    CK(clSetKernelArg(S.kcompose, a++, sizeof(bg), &bg));
    CK(clSetKernelArg(S.kcompose, a++, sizeof(S.bout), &S.bout));
    CK(clEnqueueNDRangeKernel(S.q, S.kcompose, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr));

    std::vector<unsigned char> img(4 * npix);
    CK(clEnqueueReadBuffer(S.q, S.bout, CL_TRUE, 0, 4 * npix, img.data(), 0, nullptr, nullptr));
    return img;
}

// ── Main ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    printf("=== Delirium v2 + OpenCL (proven pipeline) ===\n");

    // 1. Generate body via v2 AI
    DomainClassifier classifier;
    Generator gen;
    BodyTopology topology = gen.generate_body("organic", "creature");
    printf("[AI] %zu anchors, %zu patches\n", topology.anchor_count, topology.patch_count);

    // 2. Initialize BodyRuntime
    BodyRuntime body(topology);
    auto rest_positions = gen.generate_rest_positions(topology);
    if (!body.initialize(rest_positions)) {
        printf("[ERROR] Body init failed\n"); return 1;
    }

    // 3. Bake surfels from v2 body
    auto surfels = bake_surfels(body, 24);
    if (surfels.empty()) {
        printf("[ERROR] No surfels baked\n"); return 1;
    }
    printf("[Bake] %zu surfels\n", surfels.size());

    // 4. Init OpenCL GPU (proven pipeline)
    GpuState gpu = InitGpu(surfels);

    // 5. SDL2 display
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL init: %s\n", SDL_GetError()); return 1;
    }
    SDL_Window* win = SDL_CreateWindow("Delirium v2 — OpenCL Render",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_SHOWN);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING, W, H);

    float yaw = 0.7f, pitch = 0.32f, dist = 2.6f;
    bool dragging = false; int lastx = 0, lasty = 0;
    bool running = true;
    Uint32 last = SDL_GetTicks(); int frames = 0; float fps = 0; double last_ms = 0;

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) running = false;
                else if (e.key.keysym.sym == SDLK_g) {
                    // Regenerate body on 'G'
                    topology = gen.generate_body("organic", "creature");
                    body = BodyRuntime(topology);
                    rest_positions = gen.generate_rest_positions(topology);
                    body.initialize(rest_positions);
                    surfels = bake_surfels(body, 24);
                    gpu = InitGpu(surfels);
                }
            }
            else if (e.type == SDL_MOUSEBUTTONDOWN) { dragging = true; lastx = e.button.x; lasty = e.button.y; }
            else if (e.type == SDL_MOUSEBUTTONUP) dragging = false;
            else if (e.type == SDL_MOUSEMOTION && dragging) {
                yaw += (e.motion.x - lastx) * 0.01f;
                pitch += (e.motion.y - lasty) * 0.01f;
                pitch = std::fmax(-1.5f, std::fmin(1.5f, pitch));
                lastx = e.motion.x; lasty = e.motion.y;
            }
            else if (e.type == SDL_MOUSEWHEEL) dist = std::fmax(0.4f, dist - e.wheel.y * 0.2f);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<unsigned char> img = RenderFrame(gpu, yaw, pitch, dist);
        auto t1 = std::chrono::high_resolution_clock::now();
        last_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        SDL_UpdateTexture(tex, nullptr, img.data(), W * 4);
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, nullptr, nullptr);
        SDL_RenderPresent(ren);

        if (++frames >= 10) {
            Uint32 now = SDL_GetTicks();
            fps = frames * 1000.0f / (now - last);
            last = now; frames = 0;
            char title[256];
            std::snprintf(title, sizeof title,
                "Delirium v2 | %.1f fps | %.1f ms/frame | dist %.2f | %u surfels",
                fps, last_ms, dist, gpu.n_surfels);
            SDL_SetWindowTitle(win, title);
        }
    }

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    printf("[OpenCL] Clean exit\n");
    return 0;
}