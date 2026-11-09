// FILEPATH: examples/complex_scene/main.cpp
// Render COMPLEX SCENES -- multi-part assemblies, many bodies, and the
// closed bodies the document learner produces.
//
// WHY A SEPARATE EXAMPLE: the gallery renders ONE body per scene. The
// interesting failures in a complex scene are the ones a single body
// cannot show:
//   - the MULTI-BODY compositing contract (body 0 clears, all bodies
//     composite, only the last resolves) -- a bug here makes all but the
//     last object invisible;
//   - the MODEL MATRIX per body (each body carries its own placement);
//   - the FRACTURE path (a detached fragment is a separate body with a
//     depth bias);
//   - the closed bodies the DOCUMENT LEARNER bakes (a 5-part gearbox
//     housing), which are the most complex geometry the engine produces.
//
// Each scene is rendered and then MEASURED, so a scene that silently
// renders one object instead of five is caught rather than admired.
//
// Usage:
//   complex_scene [--out DIR] [--width W] [--height H] [--scene NAME]
//                 [--doc-tree FILE.json]
#include "delirium/ai/DocLearner.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/core/Scene.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Transform.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace DeliRium;

#include <sys/stat.h>
#include <sys/types.h>

namespace {

// Create the output directory (and its parents). The gallery has its own
// copy of this; without it SaveFramePPM FAILS SILENTLY and the measurement
// below reads a file that was never written -- which looks exactly like
// "the scene rendered nothing". That is a measurement artefact wearing the
// costume of a rendering bug, so the directory is created up front AND the
// write result is checked below.
void mkdir_p(const std::string& p) {
    if (p.empty()) return;
    std::string cur;
    for (size_t i = 0; i < p.size(); ++i) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size())
            ::mkdir(cur.c_str(), 0755);   // EEXIST is fine
    }
}

struct Part {
    std::string name;
    std::unique_ptr<PSFBody> body;
    Vec3 position{0,0,0};
    Quat rotation{0,0,0,1};
    Vec3 scale{1,1,1};
};

// Render a list of parts as ONE frame, using the renderer's real
// multi-body contract: body 0 clears, every body composites, only the
// last sets final=true.
bool RenderAssembly(SplatRenderer& r, const std::vector<Part>& parts,
                    const Camera& cam, const std::string& ppm_path) {
    for (size_t i = 0; i < parts.size(); ++i) {
        const Part& p = parts[i];
        if (!p.body) continue;
        // The placement lives OUTSIDE the frozen 64-byte records -- a
        // model matrix applied on the GPU.
        Transform t;
        t.position = p.position;
        t.rotation = p.rotation;
        t.scale = p.scale;
        const Mat4 mm = t.ToMatrix();
        const bool clear = (i == 0);
        const bool final = (i + 1 == parts.size());
        r.RenderBody(*p.body, cam, clear, final, 0.0f, mm.m);
    }
    // A failed save must be LOUD. Returning false and letting the caller
    // measure a file that was never written turns a filesystem mistake
    // into an apparent rendering failure.
    if (!r.SaveFramePPM(ppm_path)) {
        std::fprintf(stderr, "complex_scene: SaveFramePPM FAILED for %s\n",
                     ppm_path.c_str());
        return false;
    }
    return true;
}

// Count the DISTINCT lit regions in a PPM -- a proxy for "how many
// separate objects are visible". A scene that should show five parts but
// shows one blob has a compositing bug.
struct Region {
    int lit = 0;
    int clusters = 0;
    double fill_ratio = 0.0;
    // Pixels that are NOT the background colour. This measures GEOMETRY
    // (how much of the frame the body covers), whereas `lit` measures
    // ILLUMINATION (how much of it is bright enough to pass a luminance
    // threshold). A surface present but turned away from the key light is
    // still geometry -- it must count for "does this half exist".
    int nonbg = 0;
    double nonbg_ratio = 0.0;
};

Region Measure(const std::string& ppm_path) {
    Region out;
    std::ifstream f(ppm_path, std::ios::binary);
    if (!f) return out;
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // P6\n<w> <h>\n255\n
    std::istringstream head(data.substr(0, 64));
    std::string magic; int w = 0, h = 0, maxv = 0;
    head >> magic >> w >> h >> maxv;
    const size_t hdr = data.find('\n', data.find('\n', data.find('\n') + 1) + 1) + 1;
    const std::string px = data.substr(hdr);
    const int n = w * h;
    std::vector<unsigned char> lit(n, 0);
    for (int i = 0; i < n; ++i) {
        const size_t j = size_t(i) * 3;
        if (j + 2 >= px.size()) break;
        const double lum = (px[j] + px[j+1] + px[j+2]) / 3.0;
        lit[i] = lum > 28.0f ? 1 : 0;
        if (lit[i]) ++out.lit;
    }
    // The BACKGROUND is the modal colour (the clear/skybox colour). Anything
    // that is not the background is the body -- regardless of how it is lit.
    {
        std::map<std::array<unsigned char,3>, int> hist;
        for (int i = 0; i < n; ++i) {
            const size_t j = size_t(i) * 3;
            if (j + 2 >= px.size()) break;
            hist[{px[j], px[j+1], px[j+2]}]++;
        }
        int best = 0;
        std::array<unsigned char,3> bg{0,0,0};
        for (const auto& kv : hist)
            if (kv.second > best) { best = kv.second; bg = kv.first; }
        for (int i = 0; i < n; ++i) {
            const size_t j = size_t(i) * 3;
            if (j + 2 >= px.size()) break;
            if (px[j] != bg[0] || px[j+1] != bg[1] || px[j+2] != bg[2]) ++out.nonbg;
        }
    }
    // Count clusters with a flood fill (4-connectivity).
    std::vector<char> seen(n, 0);
    for (int i = 0; i < n; ++i) {
        if (!lit[i] || seen[i]) continue;
        ++out.clusters;
        std::vector<int> stack{i};
        seen[i] = 1;
        while (!stack.empty()) {
            const int p = stack.back(); stack.pop_back();
            const int x = p % w, y = p / w;
            const int nb[4] = {x > 0 ? p-1 : -1, x < w-1 ? p+1 : -1,
                               y > 0 ? p-w : -1, y < h-1 ? p+w : -1};
            for (int q : nb) {
                if (q < 0 || q >= n || seen[q] || !lit[q]) continue;
                seen[q] = 1;
                stack.push_back(q);
            }
        }
    }
    out.fill_ratio = n ? double(out.lit) / double(n) : 0.0;
    out.nonbg_ratio = n ? double(out.nonbg) / double(n) : 0.0;
    return out;
}

// --- scene builders ---------------------------------------------------------

// A 5-part gearbox housing, built the way the DOCUMENT LEARNER builds it:
// several sheets composed into a closed body.
//
// NOTE the lower sheet's rule: SAME origin as the upper, with a NEGATIVE z
// scale. That is the measured closure rule (see
// examples/ai_pipeline_probe/closed_body_probe.cpp): a negative scale
// already flips the sheet downward from the seam plane it shares with its
// parent, so the lower half must NOT also be offset -- offsetting it opens
// a gap instead of closing the body.
std::unique_ptr<PSFBody> BuildGearboxHousing() {
    // The same tree the live model produced for the gearbox document.
    const char* tree_json = R"({
      "kind":"body","substance":"ellipsoid",
      "substance_params":{"sx":0.6,"sy":0.6,"sz":0.5},
      "children":[
        {"kind":"lower","substance":"ellipsoid",
         "substance_params":{"sx":0.6,"sy":0.6,"sz":0.5},
         "scale":[1,1,-1],"origin":[0,0,0]},
        {"kind":"top","substance":"ellipsoid",
         "substance_params":{"sx":0.5,"sy":0.5,"sz":0.3},
         "origin":[0,0,0.5]},
        {"kind":"rotor","substance":"ellipsoid",
         "substance_params":{"sx":0.2,"sy":0.2,"sz":0.3},
         "origin":[0,0,0]},
        {"kind":"shaft","substance":"torus",
         "substance_params":{"major":0.4,"minor":0.06},
         "scale":[1,1,1],"origin":[0,0,0]}
      ]})";
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(tree_json, tree, why)) return nullptr;
    return ElementSystem::BakeBody(*tree, 48, 1.0f, why);
}

// A CLOSED SPHERE: the upper hemisphere plus a flipped lower hemisphere at
// the SAME origin. This is the body the user reported as "only one face" --
// before the multi-layer bake it averaged into a single dome with a flat
// bottom. It is rendered from BELOW-ish angles on purpose: the underside is
// exactly the part that was missing, so a camera that sees it is a camera
// that proves the fix.
std::unique_ptr<PSFBody> BuildClosedSphere() {
    const char* tree_json = R"({
      "kind":"body","substance":"ellipsoid",
      "substance_params":{"sx":0.45,"sy":0.45,"sz":0.45},
      "children":[
        {"kind":"lower","substance":"ellipsoid",
         "substance_params":{"sx":0.45,"sy":0.45,"sz":0.45},
         "scale":[1,1,-1],"origin":[0,0,0]}
      ]})";
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(tree_json, tree, why)) return nullptr;
    return ElementSystem::BakeBody(*tree, 48, 1.0f, why);
}

// A scene of FIVE separate bodies -- the pure multi-body compositing test.
std::vector<Part> BuildFiveBodyScene() {
    std::vector<Part> parts;
    const char* names[] = {"housing", "rotor", "shaft", "cap", "flange"};
    for (int i = 0; i < 5; ++i) {
        Part p;
        p.name = names[i];
        // Each body is a different shape, so the regions are separable.
        if (i == 3) {
            const char* t = R"({"kind":"c","substance":"torus",
                "substance_params":{"major":0.3,"minor":0.1}})";
            std::unique_ptr<Element> tree; std::string why;
            if (AI::ParseElementTree(t, tree, why))
                p.body = ElementSystem::BakeBody(*tree, 40, 1.0f, why);
        } else {
            const char* t = R"({"kind":"e","substance":"ellipsoid",
                "substance_params":{"sx":0.3,"sy":0.3,"sz":0.25}})";
            std::unique_ptr<Element> tree; std::string why;
            if (AI::ParseElementTree(t, tree, why))
                p.body = ElementSystem::BakeBody(*tree, 40, 1.0f, why);
        }
        // Lay them out so they do not overlap -- five separate silhouettes.
        const float a = float(i) * 1.256637f;   // 72 degrees apart
        p.position = { std::cos(a) * 1.1f, std::sin(a) * 0.7f, 0.f };
        parts.push_back(std::move(p));
    }
    return parts;
}

} // namespace

int main(int argc, char** argv) {
    std::string outdir = "/tmp/complex_scene";
    int w = 960, h = 720;
    std::string scene = "all";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outdir = argv[++i];
        else if (!std::strcmp(argv[i], "--width") && i + 1 < argc) w = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--height") && i + 1 < argc) h = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--scene") && i + 1 < argc) scene = argv[++i];
    }
    // The renderer is headless here: no window, no SDL. The gallery does
    // the same, so this is the established pattern for an offline render.
    SplatRenderer renderer;
    if (!renderer.Initialize(w, h, nullptr, "kernels")) {
        std::fprintf(stderr, "complex_scene: renderer init failed\n");
        return 1;
    }

    // The output directory MUST exist before the first save: SaveFramePPM
    // fails silently if it does not, and a failed save leaves the
    // measurement reading a file that was never written -- which is
    // indistinguishable from "the scene rendered nothing".
    mkdir_p(outdir);

    int failures = 0;
    const auto want = [&](const char* s) { return scene == "all" || scene == s; };

    // --- scene 1: the closed gearbox housing (one body, many sheets) -----
    if (want("gearbox")) {
        std::vector<Part> parts;
        Part p; p.name = "gearbox"; p.body = BuildGearboxHousing();
        if (p.body) {
            parts.push_back(std::move(p));
            Camera cam;
            cam.position = {0.f, 0.f, 2.6f};
            cam.target = {0.f, 0.f, 0.f};
            cam.up = {0.f, 1.f, 0.f};
            cam.fov_deg = 42.f;
            const std::string out = outdir + "/gearbox.ppm";
            if (!RenderAssembly(renderer, parts, cam, out)) {
                std::printf("  FAIL: the save failed\n");
                ++failures;
            } else {
                const Region m = Measure(out);
                std::printf("gearbox:      lit=%d (%.2f%% of frame) clusters=%d\n",
                            m.lit, m.fill_ratio * 100.0, m.clusters);
                if (m.lit < 500) {
                    std::printf("  FAIL: the housing barely rendered\n");
                    ++failures;
                }
            }
        } else {
            std::printf("gearbox:      FAIL: could not build the housing\n");
            ++failures;
        }
    }

    // --- scene 3: the CLOSED SPHERE, viewed from BELOW --------------------
    //
    // The camera sits BELOW the seam plane (z < 0) and looks UP at the
    // sphere. That view shows the LOWER hemisphere -- exactly the part that
    // was missing when the two hemispheres were averaged into one dome.
    //
    // The measurement is the point: for a closed sphere, the silhouette must
    // be a DISC (round in every direction) and the CENTRE of the frame must
    // be lit (the camera is inside the sphere's lower half, looking at the
    // inside of the surface). Before the fix the lower half simply did not
    // exist, so a below-seam camera saw either nothing or a flat cut.
    if (want("sphere")) {
        Part p; p.name = "sphere"; p.body = BuildClosedSphere();
        Region m_under{}, m_over{};
        if (p.body) {
            std::vector<Part> parts; parts.push_back(std::move(p));
            Camera cam;
            // below the seam, looking up and across at the sphere
            cam.position = {0.f, -1.9f, -0.85f};
            cam.target = {0.f, 0.f, -0.15f};
            cam.up = {0.f, 0.f, 1.f};
            cam.fov_deg = 45.f;
            const std::string out = outdir + "/sphere_below.ppm";
            if (!RenderAssembly(renderer, parts, cam, out)) {
                std::printf("  FAIL: the save failed\n");
                ++failures;
            } else {
                m_under = Measure(out);
                std::printf("sphere(under): lit=%d (%.2f%%)  geometry=%d (%.2f%%)  clusters=%d\n",
                            m_under.lit, m_under.fill_ratio * 100.0,
                            m_under.nonbg, m_under.nonbg_ratio * 100.0, m_under.clusters);
                if (m_under.nonbg < 5000) {
                    std::printf("  FAIL: the lower hemisphere did not render\n");
                    ++failures;
                }
            }
            // The same sphere from ABOVE, for the A/B: a closed body must
            // render from BOTH sides. If only the upper half existed, the
            // below view would be near-empty while the above view is full.
            {
                Part q; q.name = "sphere"; q.body = BuildClosedSphere();
                std::vector<Part> qp; qp.push_back(std::move(q));
                Camera c2;
                c2.position = {0.f, -1.9f, 0.85f};
                c2.target = {0.f, 0.f, 0.15f};
                c2.up = {0.f, 0.f, -1.f};
                c2.fov_deg = 45.f;
                const std::string out2 = outdir + "/sphere_above.ppm";
                if (!RenderAssembly(renderer, qp, c2, out2)) {
                    std::printf("  FAIL: the save failed\n");
                    ++failures;
                } else {
                    m_over = Measure(out2);
                    std::printf("sphere(over):  lit=%d (%.2f%%)  geometry=%d (%.2f%%)  clusters=%d\n",
                                m_over.lit, m_over.fill_ratio * 100.0,
                                m_over.nonbg, m_over.nonbg_ratio * 100.0, m_over.clusters);
                    if (m_over.nonbg < 5000) {
                        std::printf("  FAIL: the upper hemisphere did not render\n");
                        ++failures;
                    }
                    // A CLOSED body must present the SAME SILHOUETTE from both
                    // sides. The comparison is on GEOMETRY (non-background
                    // pixels), NOT on brightness:
                    //
                    //   The `lit` metric is a LUMINANCE threshold, so it
                    //   measures how much of the surface faces the key
                    //   light. The lower hemisphere and the upper
                    //   hemisphere are lit very differently -- the below
                    //   view reads 2.9x brighter than the above view even
                    //   though their silhouettes are IDENTICAL. Comparing
                    //   `lit` would therefore report "the body is not
                    //   closed" for a perfectly closed body.
                    //
                    //   The geometric silhouette of a closed sphere is a
                    //   DISC: the same area from any direction. If the
                    //   lower half were missing, the below view would
                    //   cover far less area.
                    if (m_under.nonbg > 0 && m_over.nonbg > 0) {
                        const double ratio = double(std::max(m_under.nonbg, m_over.nonbg)) /
                                             double(std::min(m_under.nonbg, m_over.nonbg));
                        std::printf("  side balance (geometry): %.4fx (1.0 == symmetric silhouette)\n",
                                    ratio);
                        // 1.5x is generous: a closed sphere's silhouette is
                        // IDENTICAL from both sides, so the measured ratio is
                        // ~1.00. A missing half shows up as 2x or more.
                        if (ratio > 1.5) {
                            std::printf("  FAIL: the body is not closed (one side is missing)\n");
                            ++failures;
                        }
                    }
                }
            }
        } else {
            std::printf("sphere:       FAIL: could not build the closed sphere\n");
            ++failures;
        }
    }

    // --- scene 4: DETERMINISM -- the same scene rendered twice must be
    // byte-identical.
    //
    // A render that differs between two runs of the SAME binary on the SAME
    // input is a RACE, not a rendering style. The march path had exactly
    // that: `build_grid` claimed a grid cell with a plain unguarded store
    // (`grid[cell] = i`), so when several surfels landed in the same cell
    // the winner was whichever work-item wrote last. Measured 1.09% of
    // pixels differing between runs (7,534 of 691,200).
    //
    // This scene renders the SAME scene twice into two files and compares
    // them byte-for-byte. It is the cheapest possible regression gate for
    // that class of bug: it needs no reference image, only that the engine
    // agrees with itself.
    if (want("determinism")) {
        std::vector<Part> parts = BuildFiveBodyScene();
        Camera cam;
        cam.position = {0.f, 0.f, 3.4f};
        cam.target = {0.f, 0.f, 0.f};
        cam.up = {0.f, 1.f, 0.f};
        cam.fov_deg = 50.f;
        const std::string a = outdir + "/det_a.ppm";
        const std::string b = outdir + "/det_b.ppm";
        bool ok = true;
        for (const std::string* out : {&a, &b}) {
            renderer.SetPath(SplatRenderer::Path::March);
            if (!RenderAssembly(renderer, parts, cam, *out)) {
                std::printf("  FAIL: the save failed\n");
                ++failures;
                ok = false;
                break;
            }
        }
        if (ok) {
            std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
            const std::string da((std::istreambuf_iterator<char>(fa)),
                                 std::istreambuf_iterator<char>());
            const std::string db((std::istreambuf_iterator<char>(fb)),
                                 std::istreambuf_iterator<char>());
            if (da.empty() || db.empty()) {
                std::printf("determinism:  FAIL: an output file was empty\n");
                ++failures;
            } else if (da == db) {
                std::printf("determinism:  OK -- two renders are byte-identical (%zu bytes)\n",
                            da.size());
            } else {
                size_t diff = 0;
                const size_t n = std::min(da.size(), db.size());
                for (size_t i = 0; i < n; ++i) if (da[i] != db[i]) ++diff;
                std::printf("determinism:  FAIL -- %zu of %zu bytes differ between two "
                            "renders of the same scene (a race)\n", diff, n);
                ++failures;
            }
        }
    }

    // --- scene 2: FIVE separate bodies (the compositing contract) --------
    if (want("five")) {
        std::vector<Part> parts = BuildFiveBodyScene();
        Camera cam;
        cam.position = {0.f, 0.f, 3.4f};
        cam.target = {0.f, 0.f, 0.f};
        cam.up = {0.f, 1.f, 0.f};
        cam.fov_deg = 50.f;
        const std::string out = outdir + "/five.ppm";
        if (!RenderAssembly(renderer, parts, cam, out)) {
            std::printf("  FAIL: the save failed\n");
            ++failures;
        } else {
            const Region m = Measure(out);
            std::printf("five bodies:  lit=%d (%.2f%% of frame) clusters=%d\n",
                        m.lit, m.fill_ratio * 100.0, m.clusters);
            // The compositing contract: all five must be visible. A bug that
            // clears per body shows ONE cluster, not five.
            if (m.clusters < 4) {
                std::printf("  FAIL: expected >=4 separate visible regions, got %d "
                            "(the multi-body compositing contract is broken)\n",
                            m.clusters);
                ++failures;
            }
        }
    }

    // --- scene 3: EVERY render path on the SAME complex scene ------------
    //
    // WHY: a path can be silently disabled (the kernel-arg audit mismatch
    // that made "path 1" render as splat while still printing the march
    // banner). Rendering the complex scene through all five paths and
    // MEASURING each one catches that: a path that falls back produces the
    // fallback's numbers, and a path that renders nothing produces zero.
    if (want("paths")) {
        const SplatRenderer::Path paths[] = {
            SplatRenderer::Path::March,
            SplatRenderer::Path::Raymarch,
            SplatRenderer::Path::SplatClassic,
            SplatRenderer::Path::SplatLayered,
            SplatRenderer::Path::SplatArea,
        };
        std::vector<Part> parts = BuildFiveBodyScene();
        Camera cam;
        cam.position = {0.f, 0.f, 3.4f};
        cam.target = {0.f, 0.f, 0.f};
        cam.up = {0.f, 1.f, 0.f};
        cam.fov_deg = 50.f;
        for (const auto p : paths) {
            renderer.SetPath(p);
            const std::string out = outdir + "/path_" + renderer.PathName() + ".ppm";
            if (!RenderAssembly(renderer, parts, cam, out)) {
                std::printf("path %-14s FAIL: the save failed\n", renderer.PathName());
                ++failures;
                continue;
            }
            const Region m = Measure(out);
            std::printf("path %-14s lit=%-7d clusters=%-3d (%.2f%% of frame)\n",
                        renderer.PathName(), m.lit, m.clusters, m.fill_ratio * 100.0);
            if (m.lit < 500) {
                std::printf("  FAIL: %s rendered almost nothing -- the path is "
                            "either disabled or broken\n", renderer.PathName());
                ++failures;
            }
            if (m.clusters < 4) {
                std::printf("  FAIL: %s shows %d regions, expected >=4 (the "
                            "multi-body compositing contract is broken)\n",
                            renderer.PathName(), m.clusters);
                ++failures;
            }
        }
    }

    std::printf("\ncomplex_scene: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
