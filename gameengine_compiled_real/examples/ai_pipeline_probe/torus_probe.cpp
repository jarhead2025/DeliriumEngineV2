// =============================================================================
// torus_probe - WHERE does the torus's height get lost?
//
// The torus substance evaluates correctly in isolation (the parser test
// proves z = minor on the crown). But the BAKED body has span_z = 0.034
// instead of ~0.24. This probe prints each layer's Z range so the loss is
// attributed to a specific stage instead of guessed at:
//
//   layer 1  the FIELD        (ElementSystem::FieldAt)
//   layer 2  the ANCHORS     (quantized positions, dequantized back)
//   layer 3  the PATCHES     (Patch::Evaluate on the baked body)
//
// Plus the ok-mask, because a patch is SKIPPED when any corner is
// out-of-support -- so a sparse mask can silently delete most of the ring.
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::AI;

int main() {
    const char* json = R"({"kind":"body","substance":"torus",
        "substance_params":{"major":0.32,"minor":0.12},"color":[0.7,0.6,0.5]})";
    std::unique_ptr<Element> tree;
    std::string why;
    if (!ParseElementTree(json, tree, why)) {
        std::printf("parse failed: %s\n", why.c_str());
        return 1;
    }

    // ---- layer 1: the FIELD over the same grid the baker uses -----------
    const int N = 16;
    const float extent = 1.0f;
    int ok_count = 0;
    float fz_lo = 1e30f, fz_hi = -1e30f;
    std::vector<int> ok((size_t)N * N, 0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float u = i / float(N - 1), v = j / float(N - 1);
            const Vec3 world{(u - 0.5f) * extent, (v - 0.5f) * extent, 0.f};
            const ElementSystem::Field f = ElementSystem::FieldAt(*tree, world);
            const size_t id = (size_t)j * N + i;
            if (f.weight <= 1e-9f) continue;
            ok[id] = 1; ++ok_count;
            fz_lo = std::min(fz_lo, f.point.z);
            fz_hi = std::max(fz_hi, f.point.z);
        }
    std::printf("layer 1 FIELD : ok=%d/%d  z=[%.5f, %.5f] span=%.5f\n",
                ok_count, N * N, fz_lo, fz_hi, fz_hi - fz_lo);

    // How many patches would the baker KEEP (all 4 corners ok)?
    int kept = 0, skipped = 0;
    for (int j = 0; j < N - 1; ++j)
        for (int i = 0; i < N - 1; ++i) {
            const bool all = ok[(size_t)j * N + i] && ok[(size_t)j * N + i + 1] &&
                             ok[(size_t)(j + 1) * N + i + 1] && ok[(size_t)(j + 1) * N + i];
            if (all) ++kept; else ++skipped;
        }
    std::printf("            patches kept=%d skipped=%d\n", kept, skipped);

    // ---- layers 2 & 3: the baked body ----------------------------------
    std::string bake_why;
    auto body = ElementSystem::BakeBody(*tree, N, extent, bake_why);
    if (!body) { std::printf("bake failed: %s\n", bake_why.c_str()); return 1; }

    std::printf("layer 2 BODY   : anchors=%zu surfaces=%zu patches=%zu\n",
                body->AnchorBlob().size() / sizeof(AnchorRecord),
                body->SurfaceBlob().size() / sizeof(SurfaceRecord),
                body->Patches().size());

    // The patch's own corner anchors: dequantize them and measure the Z
    // range over the patches the baker kept.
    float pz_lo = 1e30f, pz_hi = -1e30f;
    float cz_lo = 1e30f, cz_hi = -1e30f;
    for (const auto& p : body->Patches()) {
        const Vec3 c = p.Evaluate(0.5f, 0.5f);
        cz_lo = std::min(cz_lo, c.z); cz_hi = std::max(cz_hi, c.z);
        // the base (undeformed) position: the patch's own corners
        for (int k = 0; k < 4; ++k) {
            const Vec3 e = p.Evaluate(k == 0 || k == 3 ? 0.f : 1.f,
                                      k == 0 || k == 1 ? 0.f : 1.f);
            pz_lo = std::min(pz_lo, e.z); pz_hi = std::max(pz_hi, e.z);
        }
    }
    std::printf("layer 3 PATCH  : corner z=[%.5f, %.5f] span=%.5f\n",
                pz_lo, pz_hi, pz_hi - pz_lo);
    std::printf("            centre z=[%.5f, %.5f] span=%.5f\n",
                cz_lo, cz_hi, cz_hi - cz_lo);

    // ---- the SAME measurement across grid resolutions --------------------
    //
    // A bounded substance (a torus, an ellipsoid) only covers PART of the
    // canvas. The baker skips a patch when ANY of its four corners is
    // out-of-support, so a coarse grid deletes most of the ring -- and the
    // patch centres that survive sit near the tube's crown, where the
    // surface is FLAT. That is why the body reads as a slab: the relief is
    // not lost, it is AVERAGED AWAY by sampling only the crown.
    //
    // The fix is more samples, not a different substance.
    std::printf("\n-- resolution sweep (patch CENTRE z-span) --\n");
    for (int G : { 12, 16, 24, 32, 48, 64 }) {
        const float e = extent;
        int okc = 0;
        float lo = 1e30f, hi = -1e30f;
        std::vector<int> m((size_t)G * G, 0);
        for (int j = 0; j < G; ++j)
            for (int i = 0; i < G; ++i) {
                const float u = i / float(G - 1), v = j / float(G - 1);
                const ElementSystem::Field f =
                    ElementSystem::FieldAt(*tree, {(u - 0.5f) * e, (v - 0.5f) * e, 0.f});
                if (f.weight <= 1e-9f) continue;
                m[(size_t)j * G + i] = 1; ++okc;
                lo = std::min(lo, f.point.z); hi = std::max(hi, f.point.z);
            }
        int kept = 0;
        for (int j = 0; j < G - 1; ++j)
            for (int i = 0; i < G - 1; ++i)
                if (m[(size_t)j * G + i] && m[(size_t)j * G + i + 1] &&
                    m[(size_t)(j + 1) * G + i + 1] && m[(size_t)(j + 1) * G + i]) ++kept;
        std::string bw;
        auto t0 = std::chrono::steady_clock::now();
        auto b = ElementSystem::BakeBody(*tree, G, e, bw);
        const double ms = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count() * 1000.0;
        float clo = 1e30f, chi = -1e30f;
        if (b) for (const auto& p : b->Patches()) {
            const Vec3 c = p.Evaluate(0.5f, 0.5f);
            clo = std::min(clo, c.z); chi = std::max(chi, c.z);
        }
        std::printf("  G=%2d ok=%3d/%3d kept=%3d field_zspan=%.5f body_zspan=%.5f bake=%.1fms\n",
                    G, okc, G * G, kept, hi - lo, b ? (chi - clo) : -1.f, ms);
    }

    // ---- what does the BODY itself think its bbox is? --------------------
    const auto& anchors = body->AnchorBlob();
    std::printf("anchor blob bytes=%zu (expect %%64==0: %d)\n",
                anchors.size(), (int)(anchors.size() % 64 == 0));
    return 0;
}
