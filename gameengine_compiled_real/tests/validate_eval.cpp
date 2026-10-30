// =============================================================================
// validate_eval.cpp - Milestone 1 gate.
//  A) loads assets/torus.psf through the real PSFFile/PSFBody/Patch path
//  B) dumps position+color at a 5x5 grid per patch -> cpp_eval.bin
//     (compared bit-near against tools/reference_eval.py)
//  C) self-checks the ANALYTIC normal against a finite-difference normal
// =============================================================================
#include "delirium/psf/PSFBody.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* in  = argc > 1 ? argv[1] : "assets/torus.psf";
    const char* out = argc > 2 ? argv[2] : "cpp_eval.bin";

    auto body = PSFBody::FromFile(in);
    if (!body) { std::fprintf(stderr, "load failed: %s\n", in); return 1; }
    std::printf("loaded %s: %zu patches\n", in, body->Patches().size());

    const int G = 5;
    std::vector<float> dump;
    dump.reserve(body->Patches().size() * G * G * 6);

    double worst_normal_dev = 0.0;
    for (const Patch& p : body->Patches()) {
        for (int i = 0; i < G; ++i)
            for (int j = 0; j < G; ++j) {
                const float u = i / float(G - 1), v = j / float(G - 1);
                const Vec3 pos = p.Evaluate(u, v);
                const Vec3 col = p.EvaluateColor(u, v);
                dump.insert(dump.end(), {pos.x, pos.y, pos.z, col.x, col.y, col.z});

                // C) analytic vs finite-difference normal (interior only:
                //    central differences need room)
                if (u > 0.05f && u < 0.95f && v > 0.05f && v < 0.95f) {
                    const float e = 1e-3f;
                    const Vec3 pu1 = p.Evaluate(u + e, v), pu0 = p.Evaluate(u - e, v);
                    const Vec3 pv1 = p.Evaluate(u, v + e), pv0 = p.Evaluate(u, v - e);
                    const Vec3 du = {pu1.x-pu0.x, pu1.y-pu0.y, pu1.z-pu0.z};
                    const Vec3 dv = {pv1.x-pv0.x, pv1.y-pv0.y, pv1.z-pv0.z};
                    Vec3 nn = { du.y*dv.z - du.z*dv.y,
                                du.z*dv.x - du.x*dv.z,
                                du.x*dv.y - du.y*dv.x };
                    const float l = std::sqrt(nn.x*nn.x + nn.y*nn.y + nn.z*nn.z);
                    if (l > 1e-12f) {
                        nn = {nn.x/l, nn.y/l, nn.z/l};
                        const Vec3 na = p.EvaluateNormal(u, v);
                        const double dev = 1.0 - (na.x*nn.x + na.y*nn.y + na.z*nn.z);
                        if (dev > worst_normal_dev) worst_normal_dev = dev;
                    }
                }
            }
    }

    FILE* f = std::fopen(out, "wb");
    std::fwrite(dump.data(), sizeof(float), dump.size(), f);
    std::fclose(f);
    std::printf("dumped %zu samples to %s\n", dump.size() / 6, out);
    std::printf("analytic vs finite-diff normal: worst 1-cos = %.3e %s\n",
                worst_normal_dev, worst_normal_dev < 1e-4 ? "(OK)" : "(FAIL)");
    return worst_normal_dev < 1e-4 ? 0 : 2;
}
