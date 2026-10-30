// =============================================================================
// march_gpu_render.cpp  -  C++ OpenCL host for progressive per-pixel render.
//
// USER'S METHOD: each depth layer is drawn and filled INDEPENDENTLY (its own
// buffer), then all layers are composed at the end by min-depth per pixel.
// The aggressive per-layer fill closes holes without interference from other
// layers, giving much better coverage than a single z-buffered march.
//
//   g++ -std=c++17 tools/march_gpu_render.cpp -lOpenCL -o march_gpu
//   ./march_gpu assets/spot/spot_march.srf out.ppm 0.7
// =============================================================================
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

static const int W=700, H=700;
static const int NLAYERS=24;        // depth bands
static const int BAND_STEPS=24;     // march steps per band
static const int FILL_PASSES=4;     // aggressive fill passes per layer

struct Surfel{ float px,py,pz,nx,ny,nz,r,g,b,radius; };
struct V3{ float x,y,z; };
static V3 sub(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 norm(V3 v){float l=std::sqrt(dot(v,v));return l>1e-9f?V3{v.x/l,v.y/l,v.z/l}:v;}

static std::string slurp(const char* p){
    std::ifstream f(p,std::ios::binary);
    if(!f){fprintf(stderr,"cannot open %s\n",p);exit(1);}
    return {(std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()};
}
#define CK(x) do{cl_int _e=(x);if(_e){printf("CL err %d at line %d\n",_e,__LINE__);exit(1);}}while(0)

int main(int argc,char** argv){
    const char* srf  = argc>1?argv[1]:"assets/spot/spot_march.srf";
    const char* out  = argc>2?argv[2]:"out.ppm";
    float yaw        = argc>3?atof(argv[3]):0.7f;

    // load surface samples
    std::string buf=slurp(srf);
    if(memcmp(buf.data(),"SRF\0",4)){printf("bad .srf\n");return 1;}
    uint32_t n; memcpy(&n,buf.data()+4,4);
    const Surfel* S=reinterpret_cast<const Surfel*>(buf.data()+8);
    printf("loaded %u samples\n",n);

    // bounds + grid dims
    V3 lo{1e30f,1e30f,1e30f},hi{-1e30f,-1e30f,-1e30f},cen{0,0,0};
    for(uint32_t i=0;i<n;++i){
        lo.x=std::min(lo.x,S[i].px);lo.y=std::min(lo.y,S[i].py);lo.z=std::min(lo.z,S[i].pz);
        hi.x=std::max(hi.x,S[i].px);hi.y=std::max(hi.y,S[i].py);hi.z=std::max(hi.z,S[i].pz);
        cen.x+=S[i].px;cen.y+=S[i].py;cen.z+=S[i].pz;
    }
    cen.x/=n;cen.y/=n;cen.z/=n;
    V3 org{lo.x-0.05f,lo.y-0.05f,lo.z-0.05f};
    float cell=0.008f;
    int gx=(int)std::ceil((hi.x-lo.x+0.1f)/cell);
    int gy=(int)std::ceil((hi.y-lo.y+0.1f)/cell);
    int gz=(int)std::ceil((hi.z-lo.z+0.1f)/cell);
    printf("grid %dx%dx%d cell %.3f\n",gx,gy,gz,cell);

    // OpenCL init
    cl_platform_id plat; CK(clGetPlatformIDs(1,&plat,nullptr));
    cl_device_id dev; CK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,nullptr));
    char dn[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,256,dn,nullptr);
    printf("GPU: %s\n",dn);
    cl_int err;
    cl_context ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err); CK(err);
#ifdef CL_VERSION_2_0
    cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,nullptr,&err); CK(err);
#else
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&err); CK(err);
#endif
    std::string ksrc=slurp("kernels/surfel_march.cl");
    const char* kp=ksrc.c_str(); size_t kl=ksrc.size();
    cl_program prog=clCreateProgramWithSource(ctx,1,&kp,&kl,&err); CK(err);
    if(clBuildProgram(prog,1,&dev,"",nullptr,nullptr)!=CL_SUCCESS){
        char log[16384]; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,16384,log,nullptr);
        printf("BUILD LOG:\n%s\n",log); return 1;
    }
    cl_kernel kclear_grid =clCreateKernel(prog,"clear_grid",&err);  CK(err);
    cl_kernel kbuild_grid =clCreateKernel(prog,"build_grid",&err);  CK(err);
    cl_kernel klayer_clear=clCreateKernel(prog,"layer_clear",&err); CK(err);
    cl_kernel klayer_draw =clCreateKernel(prog,"layer_draw",&err);  CK(err);
    cl_kernel klayer_fill =clCreateKernel(prog,"layer_fill",&err);  CK(err);
    cl_kernel kcompose    =clCreateKernel(prog,"layer_compose",&err);CK(err);

    // buffers
    cl_mem bS=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                             sizeof(Surfel)*n,(void*)S,&err); CK(err);
    size_t ncell=(size_t)gx*gy*gz;
    cl_mem bgrid=clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(cl_int)*ncell,nullptr,&err); CK(err);
    // all layers packed: color[layer*W*H + pid], depth[layer*W*H + pid]
    size_t npix=(size_t)W*H;
    cl_mem ball_color=clCreateBuffer(ctx,CL_MEM_READ_WRITE,4*npix*NLAYERS,nullptr,&err); CK(err);
    cl_mem ball_depth=clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(cl_float)*npix*NLAYERS,nullptr,&err); CK(err);
    // ping-pong scratch for fill
    cl_mem bcolA=clCreateBuffer(ctx,CL_MEM_READ_WRITE,4*npix,nullptr,&err); CK(err);
    cl_mem bcolB=clCreateBuffer(ctx,CL_MEM_READ_WRITE,4*npix,nullptr,&err); CK(err);
    cl_mem bdepA=clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(cl_float)*npix,nullptr,&err); CK(err);
    cl_mem bdepB=clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(cl_float)*npix,nullptr,&err); CK(err);
    cl_mem bout =clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,4*npix,nullptr,&err); CK(err);

    // build 3D grid
    cl_uint nc=(cl_uint)ncell;
    CK(clSetKernelArg(kclear_grid,0,sizeof(bgrid),&bgrid));
    CK(clSetKernelArg(kclear_grid,1,sizeof(nc),&nc));
    size_t gc=ncell; CK(clEnqueueNDRangeKernel(q,kclear_grid,1,nullptr,&gc,nullptr,0,nullptr,nullptr));
    cl_float3 gorg={{org.x,org.y,org.z}};
    int a=0;
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(bS),&bS));
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(n),&n));
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(bgrid),&bgrid));
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(gx),&gx));
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(gy),&gy));
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(gz),&gz));
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(gorg),&gorg));
    CK(clSetKernelArg(kbuild_grid,a++,sizeof(cell),&cell));
    size_t gn=n; CK(clEnqueueNDRangeKernel(q,kbuild_grid,1,nullptr,&gn,nullptr,0,nullptr,nullptr));

    // camera
    float dist=2.6f;
    V3 eye{cen.x+dist*std::cos(0.32f)*std::sin(yaw),
           cen.y-dist*std::cos(0.32f)*std::cos(yaw),
           cen.z+dist*std::sin(0.32f)};
    V3 fwd=norm(sub(cen,eye));
    V3 right=norm(cross(fwd,V3{0,0,1}));
    V3 up=cross(right,fwd);
    float tanf=std::tan(50.0f*3.14159265f/360.0f);
    float d0=std::sqrt(dot(sub(eye,cen),sub(eye,cen)));
    float dmin=d0-1.2f,dmax=d0+1.2f;
    float layer_thick=(dmax-dmin)/NLAYERS;
    V3 L=norm(V3{0.4f,-0.5f,0.6f});
    cl_float3 ceye={{eye.x,eye.y,eye.z}};
    cl_float3 cfwd={{fwd.x,fwd.y,fwd.z}};
    cl_float3 cright={{right.x,right.y,right.z}};
    cl_float3 cup={{up.x,up.y,up.z}};
    cl_float3 clight={{L.x,L.y,L.z}};
    cl_uint w=W,h=H;
    const float INF=1e30f;
    size_t gws2[2]={(size_t)W,(size_t)H};
    size_t gpix=npix;

    // process each layer independently
    for(int li=0;li<NLAYERS;++li){
        float d_far =dmax - li*layer_thick;
        float d_near=d_far - layer_thick;
        float band_depth=(d_far+d_near)*0.5f;

        // 1. clear this layer's working buffers (A)
        cl_uint np2=(cl_uint)npix;
        CK(clSetKernelArg(klayer_clear,0,sizeof(bcolA),&bcolA));
        CK(clSetKernelArg(klayer_clear,1,sizeof(bdepA),&bdepA));
        CK(clSetKernelArg(klayer_clear,2,sizeof(np2),&np2));
        CK(clSetKernelArg(klayer_clear,3,sizeof(INF),&INF));
        CK(clEnqueueNDRangeKernel(q,klayer_clear,1,nullptr,&gpix,nullptr,0,nullptr,nullptr));

        // 2. draw this band into layer buffer A
        a=0;
        CK(clSetKernelArg(klayer_draw,a++,sizeof(bS),&bS));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(bgrid),&bgrid));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(gx),&gx));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(gy),&gy));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(gz),&gz));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(gorg),&gorg));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(cell),&cell));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(ceye),&ceye));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(cfwd),&cfwd));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(cright),&cright));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(cup),&cup));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(tanf),&tanf));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(w),&w));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(h),&h));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(d_far),&d_far));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(d_near),&d_near));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(BAND_STEPS),&BAND_STEPS));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(band_depth),&band_depth));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(clight),&clight));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(bcolA),&bcolA));
        CK(clSetKernelArg(klayer_draw,a++,sizeof(bdepA),&bdepA));
        CK(clEnqueueNDRangeKernel(q,klayer_draw,2,nullptr,gws2,nullptr,0,nullptr,nullptr));

        // 3. fill aggressively (FILL_PASSES passes, ping-pong A<->B)
        cl_mem curC=bcolA,curD=bdepA,scrC=bcolB,scrD=bdepB;
        for(int fp=0;fp<FILL_PASSES;++fp){
            a=0;
            CK(clSetKernelArg(klayer_fill,a++,sizeof(curC),&curC));
            CK(clSetKernelArg(klayer_fill,a++,sizeof(curD),&curD));
            CK(clSetKernelArg(klayer_fill,a++,sizeof(scrC),&scrC));
            CK(clSetKernelArg(klayer_fill,a++,sizeof(scrD),&scrD));
            CK(clSetKernelArg(klayer_fill,a++,sizeof(w),&w));
            CK(clSetKernelArg(klayer_fill,a++,sizeof(h),&h));
            CK(clSetKernelArg(klayer_fill,a++,sizeof(INF),&INF));
            CK(clSetKernelArg(klayer_fill,a++,sizeof(band_depth),&band_depth));
            CK(clEnqueueNDRangeKernel(q,klayer_fill,2,nullptr,gws2,nullptr,0,nullptr,nullptr));
            cl_mem t=curC;curC=scrC;scrC=t;
            t=curD;curD=scrD;scrD=t;
        }

        // 4. copy filled layer into the packed all-layers buffer at slot li
        size_t col_off=(size_t)li*npix*4;
        size_t dep_off=(size_t)li*npix*sizeof(cl_float);
        CK(clEnqueueCopyBuffer(q,curC,ball_color,0,col_off,npix*4,0,nullptr,nullptr));
        CK(clEnqueueCopyBuffer(q,curD,ball_depth,0,dep_off,npix*sizeof(cl_float),0,nullptr,nullptr));
    }

    // compose all layers: nearest depth wins per pixel
    cl_uchar4 bg={{15,15,18,255}};
    cl_uint nl=NLAYERS;
    a=0;
    CK(clSetKernelArg(kcompose,a++,sizeof(ball_color),&ball_color));
    CK(clSetKernelArg(kcompose,a++,sizeof(ball_depth),&ball_depth));
    CK(clSetKernelArg(kcompose,a++,sizeof(nl),&nl));
    CK(clSetKernelArg(kcompose,a++,sizeof(w),&w));
    CK(clSetKernelArg(kcompose,a++,sizeof(h),&h));
    CK(clSetKernelArg(kcompose,a++,sizeof(INF),&INF));
    CK(clSetKernelArg(kcompose,a++,sizeof(bg),&bg));
    CK(clSetKernelArg(kcompose,a++,sizeof(bout),&bout));
    CK(clEnqueueNDRangeKernel(q,kcompose,2,nullptr,gws2,nullptr,0,nullptr,nullptr));

    std::vector<unsigned char> img(4*npix);
    CK(clEnqueueReadBuffer(q,bout,CL_TRUE,0,4*npix,img.data(),0,nullptr,nullptr));
    clFinish(q);

    std::ofstream fo(out,std::ios::binary);
    fo<<"P6\n"<<W<<" "<<H<<"\n255\n";
    for(size_t i=0;i<npix;++i) fo.write((char*)&img[i*4],3);
    printf("wrote %s\n",out);
    return 0;
}
