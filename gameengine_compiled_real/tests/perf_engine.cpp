// =============================================================================
// perf_engine.cpp - Performance report for the current engine.
// Not a pass/fail gate (always exits 0): it times every subsystem that is
// now implemented and prints a per-operation report so we can see how the
// engine performs after the modal/refit/flush/split work.
//
//  A) LOAD        PSFBody::FromFile + ModalDynamics::LoadModes
//  B) EVAL        Patch::Evaluate over a 9x9 grid on every patch
//  C) CLOSEST     ClosestPointEx on 2000 random near-surface points
//  D) RAYCAST     Raycast on 2000 rays aimed at known surface points
//  E) MODAL       ModalDynamics::Step + WriteBack (600 steps, ~12 frames)
//  F) AMR         RefineWhere + CoarsenWhere (live split/merge)
//  G) REFIT       RefitLocal over a region
//  H) SEAMS       SeamGraph::Build + MaxGap
//  I) GPU FLUSH   FlushToGPU: full re-upload, partial (3 records), anchor
//
// Each section does a warm-up pass then N timed iterations; we report the
// total, the per-op mean, and a throughput where it makes sense. Timings
// are wall-clock (std::chrono::steady_clock).
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/physics/ModalDynamics.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

using namespace DeliRium;
using Clock = std::chrono::steady_clock;

namespace {
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
void report(const char* tag, double total_ms, long ops) {
    const double per = ops ? total_ms / ops : 0.0;
    const double mhz = ops ? (ops / (total_ms / 1000.0)) / 1e6 : 0.0;
    std::printf("  %-9s %10.3f ms total | %8ld ops | %9.4f ms/op | %8.2f Mop/s\n",
                tag, total_ms, ops, per, mhz);
}
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
} // namespace

int main(int argc, char** argv) {
    const char* psf = argc > 1 ? argv[1] : "assets/torus.psf";
    const char* psm = argc > 2 ? argv[2] : "assets/torus.modes";

    // --- A) LOAD ----------------------------------------------------------
    auto t0 = Clock::now();
    auto body = PSFBody::FromFile(psf);
    if (!body) { std::fprintf(stderr, "psf load failed\n"); return 1; }
    ModalDynamics dyn;
    const bool have_modes = dyn.LoadModes(psm, *body);
    const double load_ms = ms_since(t0);
    const size_t na = body->AnchorBlob().size();
    const size_t ns = body->SurfaceBlob().size();
    const Vec3 sp = body->BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    std::printf("== DeliRium engine performance report ==\n");
    std::printf("asset: %s  (%zu anchors, %zu surfaces, diag %.3f)\n",
                psf, na, ns, diag);
    std::printf("modes: %s (%d modes)\n",
                have_modes ? "loaded" : "absent", dyn.ModeCount());
    std::printf("A) load: %.3f ms (psf + modes)\n", load_ms);

    std::srand(12345);
    auto frand = [] { return std::rand() / (float)RAND_MAX; };
    auto rand_patch = [&](uint32_t n) {
        return (uint32_t)(frand() * n);
    };

    // --- B) EVAL (9x9 grid on every patch) --------------------------------
    {
        const int G = 9;
        // warm-up
        for (const Patch& p : body->Patches())
            for (int i = 0; i < G; ++i)
                for (int j = 0; j < G; ++j)
                    p.Evaluate(i/float(G-1), j/float(G-1));
        const long ops = (long)body->Patches().size() * G * G;
        t0 = Clock::now();
        double sink = 0;
        for (const Patch& p : body->Patches())
            for (int i = 0; i < G; ++i)
                for (int j = 0; j < G; ++j) {
                    const Vec3 v = p.Evaluate(i/float(G-1), j/float(G-1));
                    sink += v.x + v.y + v.z;
                }
        const double m = ms_since(t0);
        report("eval", m, ops);
        (void)sink;
    }

    // --- C) CLOSEST POINT -------------------------------------------------
    {
        const int N = 2000;
        std::vector<SurfaceHit> hits(N);
        // warm-up
        for (int i = 0; i < 50; ++i) {
            const uint32_t pid = rand_patch((uint32_t)body->Patches().size());
            const float u = 0.1f + 0.8f*frand(), v = 0.1f + 0.8f*frand();
            const Vec3 s = body->Patches()[pid].Evaluate(u, v);
            const Vec3 n = body->Patches()[pid].EvaluateNormal(u, v);
            body->ClosestPointEx({s.x+0.02f*diag*n.x, s.y+0.02f*diag*n.y,
                                  s.z+0.02f*diag*n.z}, hits[0]);
        }
        t0 = Clock::now();
        int ok = 0;
        for (int i = 0; i < N; ++i) {
            const uint32_t pid = rand_patch((uint32_t)body->Patches().size());
            const float u = 0.1f + 0.8f*frand(), v = 0.1f + 0.8f*frand();
            const Vec3 s = body->Patches()[pid].Evaluate(u, v);
            const Vec3 n = body->Patches()[pid].EvaluateNormal(u, v);
            if (body->ClosestPointEx({s.x+0.02f*diag*n.x, s.y+0.02f*diag*n.y,
                                      s.z+0.02f*diag*n.z}, hits[i])) ++ok;
        }
        const double m = ms_since(t0);
        report("closest", m, N);
        std::printf("        (hit rate %d/%d)\n", ok, N);
    }

    // --- D) RAYCAST -------------------------------------------------------
    {
        const int N = 2000;
        std::vector<SurfaceHit> hits(N);
        // warm-up
        for (int i = 0; i < 50; ++i) {
            const uint32_t pid = rand_patch((uint32_t)body->Patches().size());
            const float u = 0.15f + 0.7f*frand(), v = 0.15f + 0.7f*frand();
            const Vec3 s = body->Patches()[pid].Evaluate(u, v);
            const Vec3 n = body->Patches()[pid].EvaluateNormal(u, v);
            const Vec3 o = {s.x+0.8f*diag*n.x, s.y+0.8f*diag*n.y, s.z+0.8f*diag*n.z};
            const Vec3 d = {s.x-o.x, s.y-o.y, s.z-o.z};
            body->Raycast(o, d, hits[0]);
        }
        t0 = Clock::now();
        int ok = 0;
        for (int i = 0; i < N; ++i) {
            const uint32_t pid = rand_patch((uint32_t)body->Patches().size());
            const float u = 0.15f + 0.7f*frand(), v = 0.15f + 0.7f*frand();
            const Vec3 s = body->Patches()[pid].Evaluate(u, v);
            const Vec3 n = body->Patches()[pid].EvaluateNormal(u, v);
            const Vec3 o = {s.x+0.8f*diag*n.x, s.y+0.8f*diag*n.y, s.z+0.8f*diag*n.z};
            const Vec3 d = {s.x-o.x, s.y-o.y, s.z-o.z};
            if (body->Raycast(o, d, hits[i])) ++ok;
        }
        const double m = ms_since(t0);
        report("raycast", m, N);
        std::printf("        (hit rate %d/%d)\n", ok, N);
    }

    // --- E) MODAL (Step + WriteBack) --------------------------------------
    if (have_modes) {
        const int STEPS = 600;
        dyn.Reset();
        const float amp = 0.01f * diag;
        for (int m = 0; m < dyn.ModeCount(); ++m)
            dyn.SetAmplitude(m, amp * (0.5f + 0.5f*frand()));
        // warm-up
        for (int i = 0; i < 20; ++i) dyn.Step(1.0f/2400.0f);
        dyn.WriteBack(*body);
        t0 = Clock::now();
        for (int i = 0; i < STEPS; ++i) {
            dyn.Step(1.0f/2400.0f);
            dyn.WriteBack(*body);
        }
        const double m = ms_since(t0);
        report("modal", m, STEPS);
        dyn.Reset();
        dyn.WriteBack(*body);
        body->RebuildViews();
    } else {
        std::printf("  E) modal  SKIPPED (no .modes file)\n");
    }

    // --- F) AMR (RefineWhere + CoarsenWhere) ------------------------------
    {
        const Vec3 c = body->Patches()[10].Evaluate(0.5f, 0.5f);
        const float r = 0.15f * diag;
        // warm-up: one full cycle
        body->RefineWhere(c, r, r*0.5f);
        body->CoarsenWhere(c, r*1.5f, 0.4f/100.0f);
        // alternate refine/coarsen so EVERY iteration does real split or
        // merge work (a plain refine-then-coarsen loop would no-op after
        // the first pass, since the region is already at the target size)
        const int ITERS = 40;
        t0 = Clock::now();
        int merges = 0;
        for (int i = 0; i < ITERS; ++i) {
            if (i % 2 == 0) body->RefineWhere(c, r, r*0.5f);
            else            merges += body->CoarsenWhere(c, r*1.5f, 0.4f/100.0f);
        }
        const double m = ms_since(t0);
        report("amr", m, ITERS);
        std::printf("        (final: %zu surfaces, %d total merges)\n",
                    body->SurfaceBlob().size(), merges);
    }

    // --- G) REFIT ---------------------------------------------------------
    {
        const Vec3 c = body->Patches()[40].Evaluate(0.5f, 0.5f);
        const float r = 0.2f * diag;
        body->RefitLocal(c, r); // warm-up
        const int ITERS = 50;
        t0 = Clock::now();
        for (int i = 0; i < ITERS; ++i) body->RefitLocal(c, r);
        const double m = ms_since(t0);
        report("refit", m, ITERS);
    }

    // --- H) SEAMS ---------------------------------------------------------
    {
        body->Seams().Build(*body); // warm-up
        body->Seams().MaxGap(*body);
        const int ITERS = 200;
        t0 = Clock::now();
        double gmax = 0;
        for (int i = 0; i < ITERS; ++i) {
            body->Seams().Build(*body);
            gmax = body->Seams().MaxGap(*body);
        }
        const double m = ms_since(t0);
        report("seams", m, ITERS);
        std::printf("        (max gap %.4f%% diag, %zu seams)\n",
                    gmax/diag*100, body->Seams().SeamCount());
    }

    // --- I) GPU FLUSH -----------------------------------------------------
    // Reload a FRESH body: section F (AMR) changed the surface count, and
    // the device buffers below are sized for the original blob. A fresh
    // body keeps the sizes stable so the partial/anchor paths stay in-bounds.
    {
        auto fbody = PSFBody::FromFile(psf);
        if (!fbody) { std::fprintf(stderr, "reload for flush failed\n"); return 1; }
        const size_t na2 = fbody->AnchorBlob().size();
        const size_t ns2 = fbody->SurfaceBlob().size();
        cl_device_id dev = pick_device();
        if (!dev) {
            std::printf("  I) flush  SKIPPED (no OpenCL device)\n");
        } else {
            char dname[128] = {};
            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
            std::printf("  device: %s\n", dname);
            cl_int err = CL_SUCCESS;
            cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
            cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
            const size_t ab = na2*64, sb = ns2*64, tb = fbody->StitchBlob().size();
            PSFFlushState fs;
            fs.queue = q;
            fs.anchors_buf  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ab, nullptr, &err);
            fs.surfaces_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sb, nullptr, &err);
            fs.stitch_buf   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tb, nullptr, &err);
            fs.anchors_bytes = fs.surfaces_bytes = fs.stitch_bytes = 0;

            // I.1 full re-upload (zero capacity)
            fbody->FlushToGPU(&fs); clFinish(q);
            t0 = Clock::now();
            const int ITERS = 20;
            for (int i = 0; i < ITERS; ++i) {
                fs.anchors_bytes = fs.surfaces_bytes = fs.stitch_bytes = 0;
                fbody->FlushToGPU(&fs);
            }
            clFinish(q);
            const double m = ms_since(t0);
            const double bytes = (double)(ab + sb + tb) * ITERS;
            report("flush_full", m, ITERS);
            std::printf("        (%.2f MB total, %.1f MB/s)\n",
                        bytes/1e6, bytes/1e6 / (m/1000.0));

            // I.2 partial: 3 dirty surface records
            for (int i = 0; i < 3; ++i)
                fbody->MutablePatches()[i].SetCoefficient(0, 0.1f);
            fbody->FlushToGPU(&fs); clFinish(q); // establish
            t0 = Clock::now();
            for (int i = 0; i < ITERS; ++i) {
                fbody->MutablePatches()[i % 3].SetCoefficient(1, 0.2f);
                fbody->FlushToGPU(&fs);
            }
            clFinish(q);
            const double mp = ms_since(t0);
            report("flush_part", mp, ITERS);

            // I.3 anchor dirty: whole anchor blob (SetPosition writes the
            // record; the caller sets the body flag, as WriteBack does)
            const Vec3 p0 = fbody->Anchors().Position(0);
            const Vec3 n0 = fbody->Anchors().Normal(0);
            fbody->Anchors().SetPosition(0, {p0.x+0.01f*n0.x, p0.y+0.01f*n0.y, p0.z+0.01f*n0.z});
            fbody->MarkAnchorsDirty();
            fbody->FlushToGPU(&fs); clFinish(q);
            t0 = Clock::now();
            for (int i = 0; i < ITERS; ++i) {
                fbody->Anchors().SetPosition(0, {p0.x+0.01f*n0.x, p0.y+0.01f*n0.y, p0.z+0.01f*n0.z});
                fbody->MarkAnchorsDirty();
                fbody->FlushToGPU(&fs);
            }
            clFinish(q);
            const double ma = ms_since(t0);
            report("flush_anch", ma, ITERS);

            clReleaseMemObject(fs.anchors_buf);
            clReleaseMemObject(fs.surfaces_buf);
            clReleaseMemObject(fs.stitch_buf);
            clReleaseCommandQueue(q);
            clReleaseContext(ctx);
        }
    }

    std::printf("== report complete ==\n");
    return 0;
}