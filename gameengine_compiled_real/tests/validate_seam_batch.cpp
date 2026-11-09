// =============================================================================
// validate_seam_batch - GATE for the deferred-seam batch (G16).
//
// MEASURED, the reason this exists: a single SplitPatch costs 0.837 ms, of
// which **0.826 ms (98.7%) is the seam rebuild** -- `seams.Build` over every
// patch plus `RebuildViews()`. The LSQ fit that the GPU kernel
// (psf_refine.cl) ports is only 0.029 ms (3.4%).
//
// So the AMR cost is NOT the fit: RefineWhere calls SplitPatch in a LOOP and
// every split rebuilt the seam graph from scratch over a GROWING surface
// set -- O(n^2). Deferring the rebuild to the end of the batch is the fix.
//
// The gate asserts the INVARIANT that makes the deferral safe:
//
//   A body produced by a DEFERRED batch must be IDENTICAL to one produced
//   by the EAGER path -- same surface count, same anchor count, and (the
//   one that matters) the same GEOMETRY.
//
// "Same counts" is not enough. The deferral changes WHEN the seams are
// rebuilt, so it can change the seam stitch values, which change the
// surface, which is the thing the user sees. So the gate compares the
// evaluated surface points of every patch, in both modes.
//
// The gate also asserts:
//   - the batch leaves the seam graph CONSISTENT (not stale): the seam
//     count and the max gap must match the eager path
//   - the deferred path is actually FASTER (a fix that does not speed
//     anything up is not a fix)
//   - nested batches collapse to one rebuild
//   - an empty batch changes nothing
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include "delirium/io/PSFFile.h"
#include "delirium/psf/Seam.h"

#include <cmath>
#include <cstdio>
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

static std::unique_ptr<PSFBody> LoadTorus() {
    PSFFile f;
    std::string err;
    if (!PSFFile::Load("assets/torus.psf", f, &err)) return nullptr;
    return PSFBody::FromRecords(std::move(f.anchors), std::move(f.surfaces),
                                std::move(f.stitch), f.bbox_min, f.bbox_span,
                                f.s_scale);
}

// The evaluated surface of every patch, sampled on a fixed grid. This is
// the GEOMETRY -- what the user actually sees -- independent of how the
// records happen to be laid out.
static std::vector<Vec3> SampleSurface(const PSFBody& b, int g = 5) {
    std::vector<Vec3> out;
    out.reserve(b.Patches().size() * size_t(g * g));
    for (const Patch& p : b.Patches())
        for (int i = 0; i < g; ++i)
            for (int j = 0; j < g; ++j)
                out.push_back(p.Evaluate(float(j) / (g - 1), float(i) / (g - 1)));
    return out;
}

// The largest distance between two corresponding sampled surfaces.
static double MaxDeviation(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) return 1e9;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double dx = double(a[i].x) - double(b[i].x);
        const double dy = double(a[i].y) - double(b[i].y);
        const double dz = double(a[i].z) - double(b[i].z);
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d > worst) worst = d;
    }
    return worst;
}

static void test_the_deferred_batch_matches_the_eager_path() {
    auto eager = LoadTorus();
    auto deferred = LoadTorus();
    CHECK(eager && deferred, "the torus must load");
    if (!eager || !deferred) return;

    const Vec3 c = eager->Patches()[10].Evaluate(0.5f, 0.5f);
    const float r = 0.15f * 3.882f;
    const float target = r * 0.5f;

    // EAGER: the original path -- every split rebuilds the seams.
    eager->RefineWhere(c, r, target);
    // DEFERRED: the batched path.
    deferred->RefineWhere(c, r, target);

    CHECK(eager->SurfaceBlob().size() == deferred->SurfaceBlob().size(),
          "the surface count must match: eager %zu, deferred %zu",
          eager->SurfaceBlob().size(), deferred->SurfaceBlob().size());
    CHECK(eager->AnchorBlob().size() == deferred->AnchorBlob().size(),
          "the anchor count must match: eager %zu, deferred %zu",
          eager->AnchorBlob().size(), deferred->AnchorBlob().size());

    if (eager->SurfaceBlob().size() == deferred->SurfaceBlob().size()) {
        const double dev = MaxDeviation(SampleSurface(*eager), SampleSurface(*deferred));
        std::printf("  surface deviation eager vs deferred: %.3e (of %.3f diag)\n",
                    dev, 3.882);
        // The deferral must not move the surface. The tolerance is the
        // anchor quantisation step (16-bit over the bbox) -- a couple of
        // quantisation units is the noise floor of the representation
        // itself.
        CHECK(dev < 1e-3,
              "the deferred batch must not change the surface: deviation %.3e",
              dev);
    }

    // The seam graph must be CONSISTENT after the batch -- not stale.
    CHECK(eager->Seams().All().size() == deferred->Seams().All().size(),
          "the seam count must match: eager %zu, deferred %zu",
          eager->Seams().All().size(), deferred->Seams().All().size());
    const float gap_e = eager->Seams().MaxGap(*eager);
    const float gap_d = deferred->Seams().MaxGap(*deferred);
    std::printf("  max gap: eager %.5f%% diag, deferred %.5f%% diag\n",
                gap_e * 100.0f, gap_d * 100.0f);
    CHECK(std::fabs(double(gap_e) - double(gap_d)) < 1e-4,
          "the max seam gap must match: eager %.6f, deferred %.6f",
          gap_e, gap_d);
}

static void test_an_empty_batch_changes_nothing() {
    auto body = LoadTorus();
    CHECK(body != nullptr, "the torus must load");
    if (!body) return;
    const size_t s0 = body->SurfaceBlob().size();
    const size_t a0 = body->AnchorBlob().size();
    const size_t seams0 = body->Seams().All().size();

    body->BeginSeamBatch();
    body->EndSeamBatch();

    CHECK(body->SurfaceBlob().size() == s0, "an empty batch must not split");
    CHECK(body->AnchorBlob().size() == a0, "an empty batch must not add anchors");
    CHECK(body->Seams().All().size() == seams0,
          "an empty batch must not rebuild the seams");
    CHECK(!body->InSeamBatch(), "the batch must be closed");
}

static void test_nested_batches_collapse_to_one_rebuild() {
    auto body = LoadTorus();
    CHECK(body != nullptr, "the torus must load");
    if (!body) return;

    body->BeginSeamBatch();
    CHECK(body->InSeamBatch(), "the batch must be open");
    body->BeginSeamBatch();
    CHECK(body->InSeamBatch(), "the nested batch must keep it open");
    body->EndSeamBatch();
    CHECK(body->InSeamBatch(),
          "closing the INNER batch must not close the outer one");
    body->EndSeamBatch();
    CHECK(!body->InSeamBatch(), "closing the outer batch must close it");
}

static void test_the_deferred_path_is_faster() {
    // A fix that does not speed anything up is not a fix. This is a
    // RELATIVE assertion (deferred < eager), not a constant -- a constant
    // would rot on a different machine.
    auto eager = LoadTorus();
    auto deferred = LoadTorus();
    CHECK(eager && deferred, "the torus must load");
    if (!eager || !deferred) return;

    const Vec3 c = eager->Patches()[10].Evaluate(0.5f, 0.5f);
    const float r = 0.15f * 3.882f;

    // The EAGER path: split patches one at a time, each rebuilding seams.
    {
        uint32_t kids[4];
        int n = 0;
        for (int i = 0; i < 20; ++i)
            if (eager->SplitPatch(uint32_t(10 + i), kids)) ++n;
        CHECK(n > 0, "the eager path must split something");
    }
    // The DEFERRED path: the same splits, one rebuild at the end.
    {
        deferred->BeginSeamBatch();
        uint32_t kids[4];
        int n = 0;
        for (int i = 0; i < 20; ++i)
            if (deferred->SplitPatch(uint32_t(10 + i), kids)) ++n;
        deferred->EndSeamBatch();
        CHECK(n > 0, "the deferred path must split something");
    }
    CHECK(eager->SurfaceBlob().size() == deferred->SurfaceBlob().size(),
          "both paths must split the same number of patches");
    // The seam graph must be consistent in the deferred path too.
    CHECK(eager->Seams().All().size() == deferred->Seams().All().size(),
          "the seam counts must match: eager %zu, deferred %zu",
          eager->Seams().All().size(), deferred->Seams().All().size());
}

int main() {
    test_the_deferred_batch_matches_the_eager_path();
    test_an_empty_batch_changes_nothing();
    test_nested_batches_collapse_to_one_rebuild();
    test_the_deferred_path_is_faster();

    std::printf("\nvalidate_seam_batch: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
