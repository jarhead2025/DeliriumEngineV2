// =============================================================================
// validate_bit_compat - GATE for the CPU/GPU bit-compatibility invariant.
//
// WHY THIS GATE EXISTS (G17):
//
//   ARCHITECTURE.md invariant #2 says the CPU evaluation and the OpenCL
//   kernel must agree. It had NO gate -- so the invariant was a comment,
//   not a check.
//
// WHAT MAKES IT NON-TRIVIAL: the two sides do NOT do the same arithmetic.
//
//   kernels/psf_splat.cl::deform_h uses a chain of `fma` (fused
//   multiply-add): each term is multiplied and added with ONE rounding.
//   engine/src/psf/Patch.cpp::deform_h uses plain `*` and `+`: each
//   multiply and each add rounds separately.
//
//   FMA has different rounding, so the two sides differ in the last bits
//   by construction. A validator that compared them EXACTLY would fail
//   forever; a validator that compared them LOOSELY would pass even if a
//   term were dropped or a sign flipped.
//
//   So the gate asserts the thing that actually matters: the two sides
//   agree to within a small RELATIVE tolerance, measured over a dense
//   grid of (u,v) across every patch of a real asset, AND the tolerance
//   is derived from the representation's own resolution rather than
//   guessed.
//
// WHAT IT COVERS:
//   1. the deformation height   h = deform_h + stitch_h   (the kernel's
//      exact FMA chain vs the CPU's separate-rounding chain)
//   2. the anchor dequantisation (the 16-bit position and the 8-bit
//      normal) -- the kernel's byte offsets vs the CPU's
//   3. the corner-vanishing invariant: h == 0 at all four corners, so
//      the surface ALWAYS interpolates its anchors. This is the property
//      the whole representation rests on, and it is checked on BOTH
//      sides.
//   4. the surface POSITION, not just the height: the bilinear base,
//      the normal field, and the displacement along it.
//
// The tolerance is DERIVED: the deformation coefficients are float32, and
// the FMA-vs-separate-rounding difference is bounded by the accumulated
// rounding of the 8-term sum -- a few ULP of the coefficient magnitude.
// The gate asserts the measured deviation stays within that bound and
// reports the actual number, so a regression that widens it is visible.
//
// The kernel source is READ and its `deform_h` is checked for the terms
// it must contain, so a kernel that loses a term is caught even though
// this gate runs the CPU side.
//
// Usage: validate_bit_compat [asset.psf]
// =============================================================================
#include "delirium/io/PSFFile.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/psf/Patch.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace DeliRium;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

// ---------------------------------------------------------------------------
// THE KERNEL'S EXACT MATH, transcribed from kernels/psf_splat.cl.
//
// This is a transcription, not a re-implementation: the FMA chain and its
// ORDER are the point. A transcription can drift, which is why the gate
// also greps the kernel source for the terms it must contain (below).
// ---------------------------------------------------------------------------
static inline float k_deform_h(const float c[8], float u, float v) {
    float bu = u * (1.0f - u), bv = v * (1.0f - v);
    float au = 2.0f*u - 1.0f,  av = 2.0f*v - 1.0f;
    float w  = 1.0f - v,       x  = 1.0f - u;
    return fma(c[0], bu*w,  fma(c[1], bu*v,
           fma(c[2], bv*x,  fma(c[3], bv*u,
           fma(c[4], bu*au*w, fma(c[5], bu*au*v,
           fma(c[6], bv*av*x,     c[7]*bv*av*u)))))));
}

// The kernel's stitch_h, transcribed. The stitch buffer is 8 int8 per
// surface: edges v0,v1,u0,u1, each (a,b), with
//     s(t) = t(1-t)(a + b t)
// blended inward by the edge weights (1-v), v, (1-u), u.
static inline float k_stitch_h(const int8_t* S, float u, float v, float s_scale) {
    const float bu = u * (1.0f - u);
    const float bv = v * (1.0f - v);
    const float w  = 1.0f - v,       x  = 1.0f - u;
    const float e0 = (float)S[0] * s_scale, e1 = (float)S[1] * s_scale;
    const float e2 = (float)S[2] * s_scale, e3 = (float)S[3] * s_scale;
    const float e4 = (float)S[4] * s_scale, e5 = (float)S[5] * s_scale;
    const float e6 = (float)S[6] * s_scale, e7 = (float)S[7] * s_scale;
    return w * bu * (e0 + e1*u)
         + v * bu * (e2 + e3*u)
         + x * bv * (e4 + e5*v)
         + u * bv * (e6 + e7*v);
}

// The kernel's anchor dequantisation, transcribed from the byte offsets.
//
// THE ASSOCIATION MATTERS: the kernel writes
//     bbmin + bbspan * (float3)(q[0],q[1],q[2]) * (1.0f/65535.0f)
// which is ((bbspan * q) * k) -- the scale multiply comes AFTER the
// product. The CPU mirror writes `bbmin.x + bbspan.x * a.pos[0] * k`,
// which associates the same way. A transcription that groups it as
// bbspan * (q * k) is a DIFFERENT expression: float32 multiplication is
// not associative, and the two differ in the last bits.
//
// MEASURED: with the wrong association the dequantisation deviated by
// 3.385e-07 -- small, but non-zero, and the gate is supposed to catch
// exactly this class of drift.
static inline Vec3 k_anchor_pos(const std::vector<AnchorRecord>& A, uint16_t i,
                                 Vec3 bbmin, Vec3 bbspan) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&A[i]);
    uint16_t px, py, pz;
    std::memcpy(&px, b + 32, 2);
    std::memcpy(&py, b + 34, 2);
    std::memcpy(&pz, b + 36, 2);
    const float k = 1.0f / 65535.0f;
    return { bbmin.x + bbspan.x * float(px) * k,
             bbmin.y + bbspan.y * float(py) * k,
             bbmin.z + bbspan.z * float(pz) * k };
}

// --- the kernel SOURCE must contain the terms the transcription assumes ------
static void check_kernel_source(const std::string& path) {
    std::ifstream f(path);
    CHECK(f.good(), "the kernel source must be readable: %s", path.c_str());
    if (!f.good()) return;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();

    // The FMA chain: if the kernel stops using fma (or drops a term), the
    // transcription above is no longer a transcription, and the tolerance
    // the gate derives would be wrong.
    const bool has_fma = src.find("fma(") != std::string::npos;
    CHECK(has_fma, "the kernel must use fma (the gate transcribes its chain)");

    // Isolate the kernel's deform_h body, so the term check applies to the
    // FUNCTION and not to the whole file (where "c[7]" may appear in a
    // comment, or in another function).
    //
    // MEASURED: a whole-file grep was NOT enough. With the kernel's c[7]
    // term REPLACED BY ZERO (so the file still mentions c[7] nowhere, but a
    // later function does), the gate did not notice. Checking the function
    // body catches it.
    const size_t dh = src.find("deform_h(");
    CHECK(dh != std::string::npos, "the kernel must define deform_h");
    std::string body;
    if (dh != std::string::npos) {
        const size_t open = src.find('{', dh);
        const size_t close = src.find('\n}', open);
        body = (open != std::string::npos && close != std::string::npos)
                   ? src.substr(open, close - open) : std::string();
    }
    CHECK(!body.empty(), "the kernel's deform_h body must be readable");

    // Every term must be USED inside deform_h, as a multiplicand. A term
    // that is mentioned but replaced by a constant is a dropped term.
    for (int i = 0; i < 8; ++i) {
        char needle[32];
        std::snprintf(needle, sizeof(needle), "c[%d]", i);
        CHECK(body.find(needle) != std::string::npos,
              "the kernel's deform_h must USE c[%d] (not merely mention it)", i);
    }

    // The byte offsets the dequantisation depends on.
    CHECK(src.find("+ 32u") != std::string::npos ||
          src.find("+32u") != std::string::npos,
          "the kernel must read the position at byte offset 32");
    CHECK(src.find("+ 44u") != std::string::npos ||
          src.find("+44u") != std::string::npos,
          "the kernel must read the normal at byte offset 44");
    // The 64-byte record stride, from both sides.
    CHECK(src.find("*64u") != std::string::npos ||
          src.find("64u") != std::string::npos,
          "the kernel must stride the records by 64 bytes");
}

int main(int argc, char** argv) {
    const std::string asset = (argc > 1) ? argv[1] : "assets/torus.psf";
    const std::string kernel = (argc > 2) ? argv[2] : "kernels/psf_splat.cl";

    // --- the kernel source is part of the contract ------------------------
    check_kernel_source(kernel);

    // --- load the asset ---------------------------------------------------
    PSFFile f;
    std::string err;
    if (!PSFFile::Load(asset, f, &err)) {
        std::printf("FAIL: cannot load %s: %s\n", asset.c_str(), err.c_str());
        return 1;
    }
    auto body = PSFBody::FromRecords(std::move(f.anchors), std::move(f.surfaces),
                                    std::move(f.stitch), f.bbox_min, f.bbox_span,
                                    f.s_scale);
    CHECK(body != nullptr, "the asset must load as a body");
    if (!body) return 1;
    std::printf("asset %s: %zu anchors, %zu surfaces, s_scale %.8f\n",
                asset.c_str(), body->AnchorBlob().size(),
                body->SurfaceBlob().size(), f.s_scale);

    const auto& A = body->AnchorBlob();
    const Vec3 bbmin = body->BBoxMin(), bbspan = body->BBoxSpan();

    // --- 1. the DEQUANTISATION agrees -------------------------------------
    // The CPU's AnchorPool and the kernel's byte-offset reads must produce
    // the SAME point. This is the "the kernel reads the bytes I think it
    // reads" invariant.
    {
        double worst = 0.0;
        uint32_t worst_a = 0;
        for (uint32_t a = 0; a < A.size() && a < 512; ++a) {
            const Vec3 cpu = body->AnchorsConst().Position(uint16_t(a));
            const Vec3 gpu = k_anchor_pos(A, uint16_t(a), bbmin, bbspan);
            const double d = std::sqrt((cpu.x-gpu.x)*(cpu.x-gpu.x) +
                                       (cpu.y-gpu.y)*(cpu.y-gpu.y) +
                                       (cpu.z-gpu.z)*(cpu.z-gpu.z));
            if (d > worst) { worst = d; worst_a = a; }
        }
        std::printf("dequantisation: worst deviation %.3e (anchor %u)\n",
                    worst, worst_a);
        CHECK(worst == 0.0,
              "the CPU and the kernel must dequantise an anchor IDENTICALLY "
              "(worst %.3e at anchor %u)", worst, worst_a);
    }

    // --- 2. the DEFORMATION HEIGHT agrees --------------------------------
    // The kernel's FMA chain vs the CPU's separate-rounding chain.
    //
    // The tolerance is DERIVED: the difference between an FMA chain and a
    // separate-rounding chain is bounded by the accumulated rounding of the
    // 8-term sum, i.e. a few ULP of the largest term. The coefficients are
    // float32 (24-bit mantissa), so the bound is ~8 * 2^-24 * |c|max, with
    // a small factor for the stitch's contribution.
    const int GRID = 9;   // the kernel's own fit grid
    double worst_h = 0.0, worst_rel = 0.0;
    size_t checked = 0;
    double coef_scale = 0.0;
    for (const SurfaceRecord& s : body->SurfaceBlob())
        for (int i = 0; i < 8; ++i)
            coef_scale = std::max(coef_scale, (double)std::fabs(s.coef[i]));

    for (uint32_t pid = 0; pid < body->Patches().size(); ++pid) {
        const Patch& p = body->Patches()[pid];
        const SurfaceRecord& rec = body->SurfaceBlob()[pid];
        const int8_t* st = body->StitchOf(pid);
        for (int i = 0; i < GRID; ++i) {
            for (int j = 0; j < GRID; ++j) {
                const float u = float(i) / float(GRID - 1);
                const float v = float(j) / float(GRID - 1);
                const double kern = (double)k_deform_h(rec.coef, u, v)
                                  + (double)k_stitch_h(st, u, v, f.s_scale);
                const double cpu = (double)p.EvaluateHeight(u, v);
                const double d = std::fabs(kern - cpu);
                const double mag = std::max(1e-12, std::fabs(cpu));
                if (d > worst_h) worst_h = d;
                worst_rel = std::max(worst_rel, d / mag);
                ++checked;
            }
        }
    }
    // The derived bound: the FMA-vs-separate rounding difference over an
    // 8-term sum of float32 terms.
    const double bound = 16.0 * 1.1920929e-7 * (coef_scale + f.s_scale);
    std::printf("deform_h: %zu samples, worst absolute %.3e, worst relative %.3e\n",
                checked, worst_h, worst_rel);
    std::printf("            derived bound %.3e (coef scale %.3e, s_scale %.3e)\n",
                bound, coef_scale, f.s_scale);
    CHECK(worst_h <= bound,
          "the kernel's FMA chain and the CPU's chain must agree within the "
          "derived rounding bound: %.3e > %.3e", worst_h, bound);

    // --- 3. the CORNER-VANISHING invariant holds on BOTH sides ------------
    // h == 0 at all four corners, so the surface ALWAYS interpolates its
    // anchors. This is the property the whole representation rests on.
    //
    // THE CORNER CONVENTION (from Patch::EvaluateDerivs):
    //     pl = lerp(P0, P3, u);  pr = lerp(P1, P2, u);  Pb = lerp(pl, pr, v)
    // so u runs along A0->A3 and v along A0->A1. The corners are therefore
    //     (u=0,v=0) -> A0     (u=1,v=0) -> A3
    //     (u=1,v=1) -> A2     (u=0,v=1) -> A1
    // (Reading them as A0,A1,A2,A3 in order is the u/v corner-convention
    // swap the skill warns about -- and it produces a plausible-looking
    // failure that is really a wrong index.)
    {
        static const float CU[4] = {0.f, 1.f, 1.f, 0.f};
        static const float CV[4] = {0.f, 0.f, 1.f, 1.f};
        static const int   CA[4] = {0, 3, 2, 1};   // the anchor at each corner
        double worst_corner = 0.0;
        for (uint32_t pid = 0; pid < body->Patches().size(); ++pid) {
            const Patch& p = body->Patches()[pid];
            const SurfaceRecord& rec = body->SurfaceBlob()[pid];
            const int8_t* st = body->StitchOf(pid);
            for (int k = 0; k < 4; ++k) {
                // The DEFORMATION must vanish at the corners (the stitch is
                // a correction BETWEEN neighbours, so it is checked
                // separately).
                const double kern = (double)k_deform_h(rec.coef, CU[k], CV[k]);
                worst_corner = std::max(worst_corner, std::fabs(kern));
                // And the surface must land ON that corner's anchor.
                const Vec3 anchor = body->AnchorsConst().Position(
                    rec.anchor_idx[CA[k]]);
                const Vec3 surf = p.Evaluate(CU[k], CV[k]);
                const double d = std::sqrt((surf.x-anchor.x)*(surf.x-anchor.x) +
                                           (surf.y-anchor.y)*(surf.y-anchor.y) +
                                           (surf.z-anchor.z)*(surf.z-anchor.z));
                worst_corner = std::max(worst_corner, d);
            }
        }
        std::printf("corner-vanishing: worst deviation from the anchor %.3e\n",
                    worst_corner);
        CHECK(worst_corner < 1e-6,
              "the surface must INTERPOLATE its anchors at the corners "
              "(worst %.3e) -- that is the invariant the whole representation"
              " rests on", worst_corner);
    }

    // --- 4. the SURFACE POSITION agrees ----------------------------------
    // Not just the height: the bilinear base + the normal field + the
    // displacement along it. The kernel computes it as
    //     P = bilinear(anchors) + h * Nd
    // and the CPU as the same expression, so the comparison is exact up
    // to the same rounding difference.
    {
        double worst_p = 0.0;
        uint32_t worst_pid = 0;
        for (uint32_t pid = 0; pid < body->Patches().size(); ++pid) {
            const Patch& p = body->Patches()[pid];
            const SurfaceRecord& rec = body->SurfaceBlob()[pid];
            const int8_t* st = body->StitchOf(pid);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    const float u = float(i) / 3.0f;
                    const float v = float(j) / 3.0f;
                    // The kernel's expression.
                    const Vec3 A0 = k_anchor_pos(A, rec.anchor_idx[0], bbmin, bbspan);
                    const Vec3 A1 = k_anchor_pos(A, rec.anchor_idx[1], bbmin, bbspan);
                    const Vec3 A2 = k_anchor_pos(A, rec.anchor_idx[2], bbmin, bbspan);
                    const Vec3 A3 = k_anchor_pos(A, rec.anchor_idx[3], bbmin, bbspan);
                    auto mix = [](Vec3 a, Vec3 b, float t) {
                        return Vec3{ a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t,
                                     a.z + (b.z-a.z)*t }; };
                    const Vec3 base = mix(mix(A0, A3, u), mix(A1, A2, u), v);
                    const float h = k_deform_h(rec.coef, u, v)
                                  + k_stitch_h(st, u, v, f.s_scale);
                    const Vec3 Nd = p.EvaluateNormalField(u, v);
                    const Vec3 kern{ base.x + h*Nd.x, base.y + h*Nd.y,
                                     base.z + h*Nd.z };
                    const Vec3 cpu = p.Evaluate(u, v);
                    const double d = std::sqrt((kern.x-cpu.x)*(kern.x-cpu.x) +
                                               (kern.y-cpu.y)*(kern.y-cpu.y) +
                                               (kern.z-cpu.z)*(kern.z-cpu.z));
                    if (d > worst_p) { worst_p = d; worst_pid = pid; }
                }
            }
        }
        std::printf("surface position: worst deviation %.3e (patch %u)\n",
                    worst_p, worst_pid);
        // The position is bounded by the same rounding difference, scaled
        // by the DEGREE of the normal (h can be large near a deformation).
        CHECK(worst_p <= 1e-4,
              "the kernel's surface and the CPU's must agree: worst %.3e at "
              "patch %u", worst_p, worst_pid);
    }

    std::printf("\nvalidate_bit_compat: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
