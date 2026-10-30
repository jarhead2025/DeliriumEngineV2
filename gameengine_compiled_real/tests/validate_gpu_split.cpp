// =============================================================================
// validate_gpu_split.cpp - GPU live-subdivision gate. Runs the new
// kernels/psf_refine.cl::split_patches on a real OpenCL device and proves it
// matches the CPU SplitPatch (the reference implementation) by:
//  A) FIDELITY: each GPU child must reproduce the parent surface; max
//     deviation sampled at 17x17 over the parent domain < 0.15% of diag
//     (same gate as validate_split.cpp).
//  B) BOOKKEEPING: +5 anchors, +3 surfaces (parent slot reused by child 0),
//     stitch blob stays 8 bytes per surface, the 4 children's stitch slots
//     are zeroed, and every child has 4 distinct in-range anchors.
//  C) CPU/GPU PARITY: run the CPU SplitPatch on an identical fresh copy of
//     the same patch; the GPU children's corner anchors must sit within
//     0.05% diag of the CPU children's, and the fitted coefficients must
//     agree to < 1e-3 absolute (fast-math tolerance).
//  D) SEAMS: after rebuilding the seam graph around the GPU children, the
//     max gap stays under the 0.05% subpixel budget.
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

using namespace DeliRium;

namespace {

cl_device_id pick_device() {
    cl_platform_id plats[8]; cl_uint np = 0;
    if (clGetPlatformIDs(8, plats, &np) != CL_SUCCESS || !np) return nullptr;
    cl_device_id dev = nullptr;
    for (cl_uint p = 0; p < np && !dev; ++p) {
        cl_device_id d; cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &d, &nd)
            == CL_SUCCESS && nd) dev = d;
    }
    if (!dev)
        for (cl_uint p = 0; p < np && !dev; ++p) {
            cl_device_id d; cl_uint nd = 0;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 1, &d, &nd)
                == CL_SUCCESS && nd) dev = d;
        }
    return dev;
}

bool read_kernel(const char* path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf();
    out = ss.str();
    return !out.empty();
}

} // namespace

int main(int argc, char** argv) {
    const char* psf = argc > 1 ? argv[1] : "assets/torus.psf";
    const char* kdir = argc > 2 ? argv[2] : "kernels";
    const uint32_t pid = 42;   // same patch validate_dent.cpp uses

    auto body = PSFBody::FromFile(psf);
    if (!body) { std::fprintf(stderr, "psf load failed\n"); return 1; }
    const Vec3 sp = body->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    const Vec3 bbmin = body->BBoxMin();
    const float s_scale = body->StitchScale();
    const size_t na0 = body->AnchorBlob().size();
    const size_t ns0 = body->SurfaceBlob().size();
    std::printf("loaded: %zu anchors, %zu surfaces, patch %u\n", na0, ns0, pid);

    // parent surface snapshot BEFORE the split (17x17, same grid as
    // validate_split.cpp)
    const int G = 17;
    std::vector<Vec3> parent_pts(G * G);
    for (int i = 0; i < G; ++i)
        for (int j = 0; j < G; ++j)
            parent_pts[i*G + j] = body->Patches()[pid].Evaluate(i/float(G-1),
                                                                j/float(G-1));

    cl_device_id dev = pick_device();
    if (!dev) { std::fprintf(stderr, "no OpenCL device\n"); return 1; }
    char dname[128] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    std::printf("OpenCL device: %s\n", dname);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "ctx %d\n", err); return 1; }
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "queue %d\n", err); return 1; }

    std::string src;
    if (!read_kernel((std::string(kdir) + "/psf_refine.cl").c_str(), src)) {
        std::fprintf(stderr, "kernel file not found: %s\n", kdir);
        return 1;
    }
    const char* psrc = src.c_str(); const size_t nsrc = src.size();
    cl_program pr = clCreateProgramWithSource(ctx, 1, &psrc, &nsrc, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "program %d\n", err); return 1; }
    if (clBuildProgram(pr, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr)
        != CL_SUCCESS) {
        size_t len = 0;
        clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::vector<char> log(len + 1);
        clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::fprintf(stderr, "build failed:\n%s\n", log.data());
        return 1;
    }
    cl_kernel k = clCreateKernel(pr, "split_patches", &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "kernel %d\n", err); return 1; }

    // device buffers: capacity for ONE split (5 anchors + 3 surfaces)
    const size_t na1 = na0 + 5, ns1 = ns0 + 3;
    std::vector<char> d_srf(ns1 * 64), d_anc(na1 * 64);
    std::memcpy(d_srf.data(), body->SurfaceBlob().data(), ns0 * 64);
    std::memcpy(d_anc.data(), body->AnchorBlob().data(), na0 * 64);
    std::vector<char> d_stitch(ns1 * 8);
    std::memcpy(d_stitch.data(), body->StitchBlob().data(), ns0 * 8);

    cl_mem b_srf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, d_srf.size(), nullptr, &err);
    cl_mem b_anc = clCreateBuffer(ctx, CL_MEM_READ_WRITE, d_anc.size(), nullptr, &err);
    cl_mem b_st  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, d_stitch.size(), nullptr, &err);
    cl_mem b_list = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4, nullptr, &err);
    cl_mem b_head = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 8, nullptr, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "buf %d\n", err); return 1; }

    const uint32_t split_list[1] = { pid };
    const uint32_t heads[2] = { (uint32_t)na0, (uint32_t)ns0 };
    clEnqueueWriteBuffer(q, b_srf, CL_TRUE, 0, ns0*64, body->SurfaceBlob().data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q, b_anc, CL_TRUE, 0, na0*64, body->AnchorBlob().data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q, b_st,  CL_TRUE, 0, ns0*8,  body->StitchBlob().data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q, b_list, CL_TRUE, 0, 4, split_list, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q, b_head, CL_TRUE, 0, 8, heads, 0, nullptr, nullptr);

    clSetKernelArg(k, 0, sizeof(cl_mem), &b_srf);
    clSetKernelArg(k, 1, sizeof(cl_mem), &b_anc);
    clSetKernelArg(k, 2, sizeof(cl_mem), &b_list);
    clSetKernelArg(k, 3, sizeof(cl_mem), &b_head);
    clSetKernelArg(k, 4, sizeof(uint32_t), &split_list[0]); // n_splits = 1
    clSetKernelArg(k, 5, sizeof(cl_float3), &bbmin);
    clSetKernelArg(k, 6, sizeof(cl_float3), &sp);
    clSetKernelArg(k, 7, sizeof(cl_mem), &b_st);
    clSetKernelArg(k, 8, sizeof(float), &s_scale);
    const size_t gsize = 1;
    if (clEnqueueNDRangeKernel(q, k, 1, nullptr, &gsize, nullptr, 0, nullptr, nullptr)
        != CL_SUCCESS) { std::fprintf(stderr, "launch failed\n"); return 1; }
    clFinish(q);

    clEnqueueReadBuffer(q, b_srf, CL_TRUE, 0, d_srf.size(), d_srf.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(q, b_anc, CL_TRUE, 0, d_anc.size(), d_anc.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(q, b_st,  CL_TRUE, 0, d_stitch.size(), d_stitch.data(), 0, nullptr, nullptr);

    int fails = 0;

    // --- B) bookkeeping ---------------------------------------------------
    const uint32_t* heads_out = (const uint32_t*)d_anc.data(); // scratch
    (void)heads_out;
    // read back alloc heads to confirm the cursors advanced by 5 and 3
    std::vector<char> headbuf(8);
    clEnqueueReadBuffer(q, b_head, CL_TRUE, 0, 8, headbuf.data(), 0, nullptr, nullptr);
    const uint32_t ha = ((const uint32_t*)headbuf.data())[0];
    const uint32_t hs = ((const uint32_t*)headbuf.data())[1];
    const bool b_ok = ha == na0 + 5 && hs == ns0 + 3
        && d_stitch.size() == ns1 * 8;
    // the 4 children's stitch slots must be zeroed. The children occupy
    // exactly {pid, ns0, ns0+1, ns0+2} (child 0 reuses the parent slot, the
    // other 3 are appended) -- NOT the whole [pid, ns1) range, which would
    // also scan the ~500 unrelated surfaces' real seam data.
    bool stitch_zero = true;
    const uint32_t zslots[4] = { pid, (uint32_t)ns0, (uint32_t)ns0 + 1,
                                 (uint32_t)ns0 + 2 };
    for (uint32_t s : zslots)
        for (int t = 0; t < 8; ++t)
            if (d_stitch[s*8 + t] != 0) stitch_zero = false;
    // every child has 4 distinct in-range anchors
    bool anchors_ok = true;
    for (size_t s = pid; s < ns1; ++s) {
        const uint16_t* ai = (const uint16_t*)(d_srf.data() + s*64);
        for (int a = 0; a < 4; ++a) {
            if (ai[a] >= na1) anchors_ok = false;
            for (int b = a + 1; b < 4; ++b)
                if (ai[a] == ai[b]) anchors_ok = false;
        }
    }
    std::printf("B) bookkeeping: heads %zu->%u anc / %zu->%u surf, stitch "
                "zeroed %s, child anchors distinct+in-range %s\n",
                na0, ha, ns0, hs, stitch_zero ? "yes" : "NO",
                anchors_ok ? "yes" : "NO");
    fails += !(b_ok && stitch_zero && anchors_ok);

    // --- A) fidelity: GPU children vs the parent snapshot -----------------
    {
        // build a CPU body from the GPU output so we can Evaluate the
        // children with the real Patch code
        std::vector<AnchorRecord> ga((const AnchorRecord*)d_anc.data(),
                                     (const AnchorRecord*)d_anc.data() + na1);
        std::vector<SurfaceRecord> gs((const SurfaceRecord*)d_srf.data(),
                                      (const SurfaceRecord*)d_srf.data() + ns1);
        std::vector<int8_t> gst(d_stitch.begin(), d_stitch.end());
        auto gbody = PSFBody::FromRecords(std::move(ga), std::move(gs),
                                          std::move(gst), bbmin, sp, s_scale);
        // the GPU appended children at ns0, ns0+1, ns0+2; child 0 reused
        // the parent slot `pid`. Map quadrant -> child id the same way the
        // CPU SplitPatch does: [pid, ns0, ns0+1, ns0+2] =
        // [u-v-, u-v+, u+v-, u+v+]
        const uint32_t kids[4] = { pid, (uint32_t)ns0,
                                   (uint32_t)ns0 + 1, (uint32_t)ns0 + 2 };
        float worst = 0;
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float pu = i/float(G-1), pv = j/float(G-1);
                const int qu = pu > 0.5f, qv = pv > 0.5f;
                const uint32_t cid = kids[qu*2 + qv];
                const float cu = pu*2 - qu, cv = pv*2 - qv;
                const Vec3 a = gbody->Patches()[cid].Evaluate(cu, cv);
                const Vec3 b = parent_pts[i*G + j];
                const float d = std::sqrt((a.x-b.x)*(a.x-b.x)
                                 + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
                if (d > worst) worst = d;
            }
        const bool a_ok = worst/diag < 0.0015f;
        std::printf("A) fidelity: worst GPU child-vs-parent deviation "
                    "%.4f%% diag (budget 0.15%%) %s\n",
                    worst/diag*100, a_ok ? "OK" : "FAIL");
        fails += !a_ok;

        // --- D) seams around the GPU children -----------------------------
        std::vector<uint32_t> ids(kids, kids + 4);
        gbody->Seams().RestitchAround(*gbody, ids);
        const float gap = gbody->Seams().MaxGap(*gbody);
        const bool d_ok = gap < 0.0005f * diag;
        std::printf("D) seams: max gap %.4f%% diag (budget 0.05%%) %s\n",
                    gap/diag*100, d_ok ? "OK" : "FAIL");
        fails += !d_ok;

        // --- C) CPU/GPU parity --------------------------------------------
        // run the CPU SplitPatch on an identical fresh copy of the same
        // patch and compare the children's corner anchors + coefficients
        auto cbody = PSFBody::FromFile(psf);
        uint32_t ckids[4];
        if (!cbody->SplitPatch(pid, ckids)) {
            std::fprintf(stderr, "CPU split failed\n");
            return 1;
        }
        // CPU child order: [pid, appended, appended, appended] =
        // [u-v-, u-v+, u+v-, u+v+] (children_out[0] = pid)
        float worst_anchor = 0, worst_coef = 0;
        for (int q = 0; q < 4; ++q) {
            const SurfaceRecord& gc = gbody->SurfaceBlob()[kids[q]];
            const SurfaceRecord& cc = cbody->SurfaceBlob()[ckids[q]];
            for (int a = 0; a < 4; ++a) {
                const Vec3 gp = gbody->Anchors().Position(gc.anchor_idx[a]);
                const Vec3 cp = cbody->Anchors().Position(cc.anchor_idx[a]);
                const float d = std::sqrt((gp.x-cp.x)*(gp.x-cp.x)
                                 + (gp.y-cp.y)*(gp.y-cp.y)
                                 + (gp.z-cp.z)*(gp.z-cp.z));
                if (d > worst_anchor) worst_anchor = d;
            }
            for (int i = 0; i < 8; ++i) {
                const float d = std::fabs(gc.coef[i] - cc.coef[i]);
                if (d > worst_coef) worst_coef = d;
            }
        }
        const bool c_ok = worst_anchor/diag < 0.0005f && worst_coef < 1e-3f;
        std::printf("C) parity: worst anchor dev %.4f%% diag (budget "
                    "0.05%%), worst coef dev %.3e (budget 1e-3) %s\n",
                    worst_anchor/diag*100, worst_coef, c_ok ? "OK" : "FAIL");
        fails += !c_ok;
    }

    clReleaseMemObject(b_srf);
    clReleaseMemObject(b_anc);
    clReleaseMemObject(b_st);
    clReleaseMemObject(b_list);
    clReleaseMemObject(b_head);
    clReleaseKernel(k);
    clReleaseProgram(pr);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    std::printf(fails ? "GPU SPLIT VALIDATION FAILED (%d)\n"
                      : "GPU SPLIT VALIDATION PASSED\n", fails);
    return fails;
}