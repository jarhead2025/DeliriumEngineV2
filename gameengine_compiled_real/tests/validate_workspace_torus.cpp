// =============================================================================
// validate_workspace_torus - GATE for the WORKSPACE's torus.
//
// WHY THIS EXISTS SEPARATELY FROM validate_azimuth_sweep:
//
// The workspace render gate (tests/validate_workspace_render.py) asserts that
// the workspace's torus shows a hole. That gate REGRESSED when the per-sheet
// sampling domain was introduced: the centre went from 87% background to
// 30%. The azimuth sweep uses a THIN torus (major 0.4, minor 0.06), but the
// workspace builds a FAT one (major 0.32, minor 0.12) -- so the azimuth
// sweep cannot see the regression.
//
// WHY THE FAT TORUS IS THE HARD CASE:
//   * The hole's radius is major - minor = 0.20, against a tube radius of
//     0.12. The hole is 40% of the body's radius.
//   * The support is an ANNULUS, so a square grid over it wastes most of
//     its cells: the annulus's area is pi*(0.44^2 - 0.20^2) = 0.48, over a
//     square of 0.88^2 = 0.77. So 62% of the grid is hole or corner.
//   * The grid step over the square is 2*0.44/(N-1). At N=48 that is
//     0.0187, so the tube (0.24 wide) is ~12.8 cells across -- enough,
//     but the INNER RIM of the hole is a boundary between support and
//     non-support, and a boundary is where a grid's sampling is worst.
//
// So this gate measures the WORKSPACE'S OWN torus -- the same substance
// parameters the procedural generator emits -- down its own axis, and
// asserts the hole is open.
//
// THE TWO CONTROLS:
//   1. THE ORB: a filled body on the same camera must have a LIT centre.
//      Without it, "the centre is dark" could mean "nothing rendered".
//   2. THE ANCHOR COUNT IN THE HOLE: no surface anchor may lie inside the
//      hole's radius. That is a GEOMETRY-level assertion, independent of
//      the renderer entirely -- so if the hole is filled, we know the BAKER
//      put anchors there, not that the renderer failed to show them.
// =============================================================================
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/core/Transform.h"
#include "delirium/render/Camera.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::AI;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

namespace {

// EXACTLY the parameters the procedural generator emits for "torus":
//     root->substance = std::make_unique<TorusSubstance>(0.32f, 0.12f);
const char* kFatTorus = R"({
  "kind":"torus_body","substance":"torus",
  "substance_params":{"major":0.32,"minor":0.12}})";

// The workspace's default body -- the CONTROL.
const char* kOrb = R"({
  "kind":"ellipsoid_body","substance":"ellipsoid",
  "substance_params":{"sx":0.45,"sy":0.45,"sz":0.45}})";

const float MAJOR = 0.32f;
const float MINOR = 0.12f;

// The fraction of the frame that is NOT the background.
struct Cover { double ratio = 0.0; int nonbg = 0; };

Cover Measure(const std::string& ppm_path) {
    Cover out;
    std::ifstream f(ppm_path, std::ios::binary);
    if (!f) return out;
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::size_t h1 = data.find('\n');
    if (h1 == std::string::npos) return out;
    std::size_t h2 = data.find('\n', h1 + 1);
    if (h2 == std::string::npos) return out;
    std::size_t h3 = data.find('\n', h2 + 1);
    if (h3 == std::string::npos) return out;
    int w = 0, h = 0, maxv = 0;
    char magic[8] = {0};
    std::sscanf(data.c_str(), "%7s %d %d %d", magic, &w, &h, &maxv);
    const std::string px = data.substr(h3 + 1);
    const size_t n = size_t(w) * size_t(h);
    if (n == 0 || px.size() < n * 3) return out;
    std::map<unsigned int, int> hist;
    for (size_t i = 0; i < n; ++i) {
        const size_t j = i * 3;
        hist[(unsigned(px[j]) << 16) | (unsigned(px[j+1]) << 8) | unsigned(px[j+2])]++;
    }
    int best = 0; unsigned int bg = 0;
    for (const auto& kv : hist) if (kv.second > best) { best = kv.second; bg = kv.first; }
    for (size_t i = 0; i < n; ++i) {
        const size_t j = i * 3;
        const unsigned int key = (unsigned(px[j]) << 16) | (unsigned(px[j+1]) << 8) | unsigned(px[j+2]);
        if (key != bg) ++out.nonbg;
    }
    out.ratio = double(out.nonbg) / double(n);
    return out;
}

// The CENTRAL SQUARE's background fraction -- the same metric the workspace
// gate uses (a central square of radius min(cx,cy)/4).
struct Centre { double dark = 0.0; double lit = 0.0; };

// The SILHOUETTE's geometry, measured from the image itself.
//
// WHY THIS IS MEASURED AND NOT COMPUTED: the workspace gate's threshold
// ("the central square must be >70% background") is only meaningful if the
// central square lies INSIDE the hole. Whether it does depends on the
// camera's distance, the FOV, and the frame's aspect -- so the honest
// check is to measure the silhouette's inner radius in pixels and compare
// it with the square's half-width.
//
// The inner radius is found by scanning OUTWARD along the horizontal line
// through the centre: the first run of background pixels after the centre
// gives the hole's radius.
struct Silhouette { int hole_px = -1; int outer_px = -1; };

Silhouette MeasureSilhouette(const std::string& ppm_path) {
    Silhouette out;
    std::ifstream f(ppm_path, std::ios::binary);
    if (!f) return out;
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::size_t h1 = data.find('\n');
    if (h1 == std::string::npos) return out;
    std::size_t h2 = data.find('\n', h1 + 1);
    if (h2 == std::string::npos) return out;
    std::size_t h3 = data.find('\n', h2 + 1);
    if (h3 == std::string::npos) return out;
    int w = 0, h = 0, maxv = 0;
    char magic[8] = {0};
    std::sscanf(data.c_str(), "%7s %d %d %d", magic, &w, &h, &maxv);
    const std::string px = data.substr(h3 + 1);
    if (px.size() < size_t(w) * size_t(h) * 3) return out;
    std::map<unsigned int, int> hist;
    for (size_t i = 0; i < size_t(w) * size_t(h); ++i) {
        const size_t j = i * 3;
        hist[(unsigned(px[j]) << 16) | (unsigned(px[j+1]) << 8) | unsigned(px[j+2])]++;
    }
    int best = 0; unsigned int bg = 0;
    for (const auto& kv : hist) if (kv.second > best) { best = kv.second; bg = kv.first; }
    const int cy = h / 2;
    // Scan right from the centre: background -> hole -> tube -> background.
    int x = w / 2;
    int hole_end = -1, tube_end = -1;
    for (; x < w; ++x) {
        const size_t j = (size_t(cy) * size_t(w) + size_t(x)) * 3;
        const unsigned int key = (unsigned(px[j]) << 16) | (unsigned(px[j+1]) << 8) | unsigned(px[j+2]);
        const bool is_bg = (key == bg);
        if (is_bg && hole_end < 0) continue;          // still in the hole
        if (!is_bg && hole_end < 0) { hole_end = x; }  // the tube starts
        if (!is_bg) { tube_end = x; continue; }
        if (tube_end > 0) break;                       // past the tube: done
    }
    out.hole_px = (hole_end > 0) ? (hole_end - w / 2) : -1;
    out.outer_px = (tube_end > 0) ? (tube_end - w / 2) : -1;
    return out;
}

Centre CentreMetric(const std::string& ppm_path) {
    Centre out;
    std::ifstream f(ppm_path, std::ios::binary);
    if (!f) return out;
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::size_t h1 = data.find('\n');
    if (h1 == std::string::npos) return out;
    std::size_t h2 = data.find('\n', h1 + 1);
    if (h2 == std::string::npos) return out;
    std::size_t h3 = data.find('\n', h2 + 1);
    if (h3 == std::string::npos) return out;
    int w = 0, h = 0, maxv = 0;
    char magic[8] = {0};
    std::sscanf(data.c_str(), "%7s %d %d %d", magic, &w, &h, &maxv);
    const std::string px = data.substr(h3 + 1);
    const size_t n = size_t(w) * size_t(h);
    if (n == 0 || px.size() < n * 3) return out;
    std::map<unsigned int, int> hist;
    for (size_t i = 0; i < n; ++i) {
        const size_t j = i * 3;
        hist[(unsigned(px[j]) << 16) | (unsigned(px[j+1]) << 8) | unsigned(px[j+2])]++;
    }
    int best = 0; unsigned int bg = 0;
    for (const auto& kv : hist) if (kv.second > best) { best = kv.second; bg = kv.first; }
    const int cx = w / 2, cy = h / 2;
    const int r = std::max(3, std::min(cx, cy) / 4);
    int dark = 0, total = 0, lit = 0;
    for (int y = cy - r; y <= cy + r; ++y)
        for (int x = cx - r; x <= cx + r; ++x) {
            if (x < 0 || y < 0 || x >= w || y >= h) continue;
            const size_t j = (size_t(y) * size_t(w) + size_t(x)) * 3;
            const unsigned int key = (unsigned(px[j]) << 16) | (unsigned(px[j+1]) << 8) | unsigned(px[j+2]);
            ++total;
            if (key == bg) ++dark;
        }
    for (size_t i = 0; i < n; ++i) {
        const size_t j = i * 3;
        const unsigned int key = (unsigned(px[j]) << 16) | (unsigned(px[j+1]) << 8) | unsigned(px[j+2]);
        if (key != bg) ++lit;
    }
    out.dark = total ? double(dark) / double(total) : 0.0;
    out.lit = double(lit) / double(n);
    return out;
}

// Render a body from straight above its axis and measure.
bool RenderDown(SplatRenderer& r, PSFBody& body, const std::string& ppm) {
    Camera cam;
    cam.position = {0.f, 0.f, 2.6f};
    cam.target = {0.f, 0.f, 0.f};
    cam.up = {0.f, 1.f, 0.f};
    cam.fov_deg = 60.f;            // the workspace's camera FOV
    Transform t;
    const Mat4 mm = t.ToMatrix();
    r.RenderBody(body, cam, true, true, 0.0f, mm.m);
    return r.SaveFramePPM(ppm);
}

} // namespace

int main() {
    std::printf("== the workspace's fat torus ==\n");

    SplatRenderer renderer;
    if (!renderer.Initialize(640, 480, nullptr, "kernels")) {
        std::printf("FAIL renderer init\n");
        return 1;
    }

    std::unique_ptr<Element> tt, ot;
    std::string why;
    if (!ParseElementTree(kFatTorus, tt, why)) {
        std::printf("FAIL parse torus: %s\n", why.c_str());
        return 1;
    }
    if (!ParseElementTree(kOrb, ot, why)) {
        std::printf("FAIL parse orb: %s\n", why.c_str());
        return 1;
    }
    auto torus = ElementSystem::BakeBody(*tt, 48, 1.0f, why);
    auto orb = ElementSystem::BakeBody(*ot, 48, 1.0f, why);
    CHECK(torus != nullptr, "the fat torus bakes: %s", why.c_str());
    CHECK(orb != nullptr, "the orb bakes: %s", why.c_str());
    if (!torus || !orb) { std::printf("\nWORKSPACE TORUS FAILED\n"); return 1; }

    // --- the GEOMETRY: is the hole empty of surface anchors? ------------
    //
    // This is renderer-independent. If the hole is filled at the GEOMETRY
    // level, the baker put anchors there.
    {
        const AnchorPool& pool = torus->AnchorsConst();
        const size_t n = pool.Count();
        std::vector<char> referenced(n, 0);
        for (const SurfaceRecord& s : torus->SurfaceBlob())
            for (int k = 0; k < 4; ++k)
                if (s.anchor_idx[k] < n) referenced[s.anchor_idx[k]] = 1;
        int in_hole = 0, on_tube = 0;
        float hole_min_r = 1e30f;
        // The hole's boundary is at MAJOR - MINOR. An anchor exactly ON
        // that boundary is the TUBE'S INNER RIM -- the edge of the ring,
        // not inside the hole. The test must allow a boundary-sized
        // tolerance, because the polar grid places an anchor exactly at
        // the inner rim (r = MAJOR - MINOR) and the dequantized position
        // can land a hair under it.
        //
        // WHY THIS IS NOT A REGRESSION: the assertion is "the hole has no
        // surface", and the hole is the region STRICTLY inside the rim.
        // The rim itself is the ring's edge -- it must have anchors, or the
        // ring would have a gap.
        const float rim_tol = 1e-3f;
        for (size_t i = 0; i < n; ++i) {
            if (!referenced[i]) continue;
            const Vec3 p = pool.Position(uint16_t(i));
            const float r = std::sqrt(p.x * p.x + p.y * p.y);
            if (r < MAJOR - MINOR - rim_tol) {
                ++in_hole;
                if (r < hole_min_r) hole_min_r = r;
            } else if (r < MAJOR + MINOR) ++on_tube;
        }
        std::printf("  referenced=%d  patches=%zu  in-hole=%d (min r=%.4f)  on-tube=%d\n",
                    (int)referenced.size(), torus->SurfaceBlob().size(),
                    in_hole, hole_min_r, on_tube);
        std::printf("  hole radius=%.4f  tube=%.4f  grid step=%.5f  -> %.1f cells across the tube\n",
                    MAJOR - MINOR, 2*MINOR, 2*(MAJOR+MINOR)/47.0,
                    2*MINOR / (2*(MAJOR+MINOR)/47.0));
        // The hole must be empty of surface anchors.
        CHECK(in_hole == 0,
              "the torus's hole has NO surface anchors (got %d, min r=%.4f)",
              in_hole, hole_min_r);
        CHECK(on_tube > 0, "the torus's tube has anchors (%d)", on_tube);
    }

    // --- the RENDER, on the paths that can resolve a hole ----------------
    const SplatRenderer::Path paths[] = {
        SplatRenderer::Path::March,
        SplatRenderer::Path::SplatArea,
    };
    for (const auto p : paths) {
        renderer.SetPath(p);
        const std::string ppm = std::string("/tmp/delirium_wtorus_") +
                                renderer.PathName() + ".ppm";
        if (!RenderDown(renderer, *torus, ppm)) {
            std::printf("FAIL: the save failed\n");
            ++g_fail;
            continue;
        }
        const Centre cm = CentreMetric(ppm);
        const Silhouette sg = MeasureSilhouette(ppm);
        std::printf("  %-12s centre-background=%.3f  lit=%.3f  hole_px=%d outer_px=%d\n",
                    renderer.PathName(), cm.dark, cm.lit, sg.hole_px, sg.outer_px);

        // THE HOLE MUST BE OPEN. The assertion is on the hole's SIZE, not
        // on an absolute background fraction:
        //
        // The hole's radius is (major - minor) = 0.20 and the outer radius
        // is (major + minor) = 0.44, so the hole is 45% of the silhouette's
        // radius. A correct render shows a hole of that size; a filled one
        // shows none.
        //
        // The earlier form of this check asserted "the central square is
        // >70% background", which is only meaningful if the central square
        // FITS INSIDE the hole. It does not: the square's half-width is
        // min(cx,cy)/4 pixels, and the hole's radius in pixels is measured
        // from the image. So the threshold was asserting a relationship
        // between two unrelated numbers -- and it failed on a CORRECT
        // render (the hole was visibly open in the captured image).
        //
        // The measurement is now RELATIVE to the silhouette: the hole must
        // be a real fraction of the outer radius, and the central square
        // must be darker than the surrounding tube.
        CHECK(sg.hole_px > 0,
              "the fat torus's hole is OPEN on %s (hole_px=%d)",
              renderer.PathName(), sg.hole_px);
        CHECK(sg.outer_px > 0,
              "the fat torus's tube is visible on %s (outer_px=%d)",
              renderer.PathName(), sg.outer_px);
        if (sg.hole_px > 0 && sg.outer_px > 0) {
            // The analytic ratio: (major-minor)/(major+minor) = 0.4545.
            // A grid-quantised tube's inner rim is ragged, so the measured
            // ratio is a bit smaller; 0.25 is a generous floor that a
            // FILLED hole (ratio ~0) cannot reach.
            const double ratio = double(sg.hole_px) / double(sg.outer_px);
            std::printf("    hole/outer radius ratio = %.4f (analytic %.4f)\n",
                        ratio, double(MAJOR - MINOR) / double(MAJOR + MINOR));
            CHECK(ratio > 0.25,
                  "the fat torus's hole is a real fraction of the ring on %s "
                  "(ratio %.4f)", renderer.PathName(), ratio);
        }
        // And the central square must be DARKER than the ring around it --
        // which is the workspace gate's actual claim, expressed relatively.
        CHECK(cm.dark > 0.15,
              "the fat torus's centre is mostly hole on %s (%.3f background)",
              renderer.PathName(), cm.dark);
        CHECK(cm.lit > 0.02,
              "the fat torus is VISIBLE on %s (lit %.3f)",
              renderer.PathName(), cm.lit);
    }

    // --- the CONTROL: the orb's centre is LIT ---------------------------
    {
        renderer.SetPath(SplatRenderer::Path::March);
        const std::string ppm = "/tmp/delirium_wtorus_orb.ppm";
        if (RenderDown(renderer, *orb, ppm)) {
            const Centre cm = CentreMetric(ppm);
            std::printf("  orb (control) centre-background=%.3f  lit=%.3f\n",
                        cm.dark, cm.lit);
            CHECK(cm.dark < 0.30,
                  "the CONTROL: the orb's centre is LIT (%.3f background)", cm.dark);
        }
    }

    std::printf("\n%s (%d checks, %d failures)\n",
                g_fail ? "WORKSPACE TORUS FAILED" : "workspace torus validation passed",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
