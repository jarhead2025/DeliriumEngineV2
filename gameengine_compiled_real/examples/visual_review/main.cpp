// =============================================================================
// visual_review -- LOOK at the thing, from several angles, as images.
//
// WHY THIS EXISTS
// ---------------
// The test suite grew to 54 gates, and almost all of them assert on RECORDS:
// residual, connectivity, seam gaps, coverage, winding, quads. Those are
// real invariants and they caught real bugs -- but they are NOT what the
// user sees. A mesh can satisfy every one of them and still look wrong.
//
// The gap this closes: there was no way to launch the engine ALREADY AIMED
// at a target object, sweep it through several viewpoints, and LOOK at the
// result. Everything visual was done by hand, one screenshot at a time,
// with the camera placed by guesswork.
//
// WHAT IT DOES
// ------------
//   1. Builds the target body (a closed sphere by default, or from a tree).
//   2. FRAMES it: the camera is placed from the body's own bounds, so the
//      subject fills the frame the same way every run -- no hand-tuned
//      coordinates that drift when the geometry changes.
//   3. SWEEPS a perspective arc: a set of (yaw, pitch) pairs around the
//      target -- including the views a defect hides in (from BELOW, from
//      the grazing rim, from directly above).
//   4. RENDERS each one with the renderer's real path, and writes a PNG.
//   5. Writes a CONTACT SHEET: every view tiled into one image, so the
//      whole arc can be read at a glance instead of opened file by file.
//
// WHY A CONTACT SHEET: "did the ring render from every side" is a question
// about the SET, not about one picture. Looking at 8 files one at a time
// hides the thing you are looking for -- a shape that is fine from the front
// and missing from the back.
//
// Usage:
//   visual_review [--out DIR] [--width W] [--height H] [--path N]
//                 [--scene NAME] [--tree FILE.json] [--contact FILE.png]
// =============================================================================
#include "delirium/ai/ElementSystem.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Transform.h"
#include "delirium/render/Camera.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/psf/PSFBody.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

using namespace DeliRium;

namespace {

void mkdir_p(const std::string& p) {
    if (p.empty()) return;
    std::string cur;
    for (size_t i = 0; i < p.size(); ++i) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) ::mkdir(cur.c_str(), 0755);
    }
}

bool ReadFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

// --- PPM -> PNG -------------------------------------------------------------
//
// The renderer writes PPM (P6). Rather than add an image dependency to the
// engine, this harness converts with a MINIMAL PNG writer (zlib "stored"
// deflate blocks + CRC32). That keeps the tool dependency-free, which
// matters because the whole point of this harness is that it can be run on
// any machine that can build the engine.
//
// Why not just ship PPM? Because the thing being reviewed is an IMAGE, and
// an image you cannot look at is not a review.
std::uint32_t crc32_of(const unsigned char* data, size_t len) {
    static std::uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[n] = c;
        }
        init = true;
    }
    std::uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void Put32(std::string& s, std::uint32_t v) {
    s.push_back(char((v >> 24) & 0xFF)); s.push_back(char((v >> 16) & 0xFF));
    s.push_back(char((v >> 8) & 0xFF));  s.push_back(char(v & 0xFF));
}

void Chunk(std::string& out, const char* type, const std::string& data) {
    Put32(out, std::uint32_t(data.size()));
    std::string body(type);
    body += data;
    out += body;
    Put32(out, crc32_of((const unsigned char*)body.data(), body.size()));
}

// Write an 8-bit RGB PNG. Uses deflate "stored" blocks -- no compression,
// but a valid PNG that every viewer reads.
bool WritePNG(const std::string& path, int w, int h,
              const std::vector<unsigned char>& rgb) {
    if (w <= 0 || h <= 0 || rgb.size() != size_t(w) * h * 3) return false;
    std::string raw;
    raw.reserve(size_t(h) * (size_t(w) * 3 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);   // filter: none
        raw.append((const char*)&rgb[size_t(y) * w * 3], size_t(w) * 3);
    }
    // zlib stream with stored deflate blocks.
    std::string z;
    z.push_back(0x78); z.push_back(0x01);
    size_t off = 0;
    while (off < raw.size()) {
        const size_t n = std::min<size_t>(65535, raw.size() - off);
        const bool last = (off + n >= raw.size());
        z.push_back(last ? 1 : 0);
        z.push_back(char(n & 0xFF)); z.push_back(char((n >> 8) & 0xFF));
        z.push_back(char(~n & 0xFF)); z.push_back(char((~n >> 8) & 0xFF));
        z.append(raw, off, n);
        off += n;
    }
    std::uint32_t a = 1, b = 0;
    for (unsigned char c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    Put32(z, (b << 16) | a);

    std::string png = "\x89PNG\r\n\x1a\n";
    std::string ihdr;
    Put32(ihdr, std::uint32_t(w)); Put32(ihdr, std::uint32_t(h));
    ihdr.push_back(8);    // bit depth
    ihdr.push_back(2);    // colour type: truecolour
    ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    Chunk(png, "IHDR", ihdr);
    Chunk(png, "IDAT", z);
    Chunk(png, "IEND", std::string());
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(png.data(), std::streamsize(png.size()));
    return f.good();
}

// Read a P6 PPM into RGB.
bool ReadPPM(const std::string& path, int& w, int& h, std::vector<unsigned char>& rgb) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::istringstream head(data.substr(0, 64));
    std::string magic; int maxv = 0;
    head >> magic >> w >> h >> maxv;
    if (magic != "P6" || w <= 0 || h <= 0) return false;
    size_t hdr = data.find('\n', data.find('\n', data.find('\n') + 1) + 1);
    if (hdr == std::string::npos) return false;
    ++hdr;
    rgb.assign(data.begin() + hdr, data.end());
    if (rgb.size() < size_t(w) * h * 3) return false;
    rgb.resize(size_t(w) * h * 3);
    return true;
}

// --- the target bodies ------------------------------------------------------

// The CLOSED SPHERE: upper hemisphere + flipped lower hemisphere at the
// SAME origin (the measured closure rule). This is the body the user
// reported as "only one face" -- so it is the one worth looking at.
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

// A TORUS: the shape whose tube's inner wall is the part a defect hides in.
std::unique_ptr<PSFBody> BuildTorus() {
    const char* tree_json = R"({
      "kind":"body","substance":"torus",
      "substance_params":{"major":0.32,"minor":0.12}})";
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(tree_json, tree, why)) return nullptr;
    return ElementSystem::BakeBody(*tree, 48, 1.0f, why);
}

// The 5-part gearbox housing -- the most complex geometry the document
// learner produces, and the one whose parts must each be visible.
std::unique_ptr<PSFBody> BuildGearbox() {
    const char* tree_json = R"({
      "kind":"body","substance":"ellipsoid",
      "substance_params":{"sx":0.6,"sy":0.6,"sz":0.5},
      "children":[
        {"kind":"lower","substance":"ellipsoid",
         "substance_params":{"sx":0.6,"sy":0.6,"sz":0.5},
         "scale":[1,1,-1],"origin":[0,0,0]},
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

// --- framing ----------------------------------------------------------------

// Frame the camera on the body's own world bounds.
//
// WHY THIS EXISTS: the camera used to be placed by hand-written
// coordinates. That is a measurement that silently rots -- change the
// geometry and the subject leaves the frame, and the harness reports
// "almost nothing rendered" instead of "the camera is wrong".
//
// The distance is derived from the body's RADIUS (so the subject's angular
// size is constant regardless of how big the body is), and the target is
// the bounds' CENTRE (so a body that is not centred on the origin is still
// centred in frame).
struct Frame {
    Vec3 target{0,0,0};
    float radius = 1.f;
};

Frame FrameOf(const PSFBody& body) {
    Frame f;
    const Vec3 lo = body.BBoxMin(), span = body.BBoxSpan();
    const Vec3 hi = lo + span;
    f.target = { (lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f };
    const Vec3 d = span;
    f.radius = std::max(0.15f, std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z) * 0.5f);
    return f;
}

// Place the camera on an orbit around the target.
//
// The distance is the radius divided by tan(fov/2) -- the distance at which
// the subject's bounding SPHERE exactly fills the frame's height. That is
// the framing a human would choose, derived rather than guessed.
Camera Orbit(const Frame& fr, float yaw_deg, float pitch_deg, float fov_deg,
             float fill = 0.62f) {
    const float fov = fov_deg * 3.14159265358979f / 180.f;
    const float dist = (fr.radius / std::max(0.05f, std::tan(fov * 0.5f))) / fill;
    const float yaw = yaw_deg * 3.14159265358979f / 180.f;
    const float pitch = pitch_deg * 3.14159265358979f / 180.f;
    const float cp = std::cos(pitch);
    Camera c;
    c.position = { fr.target.x + dist * cp * std::sin(yaw),
                   fr.target.y + dist * std::sin(pitch),
                   fr.target.z + dist * cp * std::cos(yaw) };
    c.target = fr.target;
    c.up = { 0.f, 1.f, 0.f };
    c.fov_deg = fov_deg;
    c.near_p = std::max(0.01f, dist - fr.radius * 2.0f);
    c.far_p = dist + fr.radius * 3.0f;
    return c;
}

} // namespace

int main(int argc, char** argv) {
    std::string outdir = "/tmp/visual_review";
    std::string scene = "sphere";
    std::string contact = "";
    std::string tree_file = "";
    int w = 480, h = 360;
    int render_path = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if      (a == "--out")    outdir = next();
        else if (a == "--scene")  scene = next();
        else if (a == "--width")  w = std::atoi(next().c_str());
        else if (a == "--height") h = std::atoi(next().c_str());
        else if (a == "--tree")   tree_file = next();
        else if (a == "--contact")contact = next();
        else if (a == "--path")   render_path = std::atoi(next().c_str());
    }
    mkdir_p(outdir);

    // --- the target body ---------------------------------------------------
    std::unique_ptr<PSFBody> body;
    if (!tree_file.empty()) {
        std::string json, why;
        if (!ReadFile(tree_file, json)) {
            std::fprintf(stderr, "visual_review: cannot read %s\n", tree_file.c_str());
            return 1;
        }
        std::unique_ptr<Element> tree;
        if (!AI::ParseElementTree(json, tree, why)) {
            std::fprintf(stderr, "visual_review: bad tree: %s\n", why.c_str());
            return 1;
        }
        body = ElementSystem::BakeBody(*tree, 48, 1.0f, why);
        if (!body) {
            std::fprintf(stderr, "visual_review: bake failed: %s\n", why.c_str());
            return 1;
        }
    } else if (scene == "sphere")  body = BuildClosedSphere();
    else if (scene == "torus")    body = BuildTorus();
    else if (scene == "gearbox")  body = BuildGearbox();
    else {
        std::fprintf(stderr, "visual_review: unknown scene '%s'\n", scene.c_str());
        return 2;
    }
    if (!body) { std::fprintf(stderr, "visual_review: no body\n"); return 1; }

    const Frame fr = FrameOf(*body);
    std::printf("visual_review: scene=%s anchors=%zu surfaces=%zu\n",
                scene.c_str(), (size_t)body->AnchorsConst().Count(),
                (size_t)body->SurfaceBlob().size());
    std::printf("  bounds centre=(%.3f,%.3f,%.3f) radius=%.3f\n",
                fr.target.x, fr.target.y, fr.target.z, fr.radius);

    // --- the renderer ------------------------------------------------------
    SplatRenderer renderer;
    if (!renderer.Initialize(w, h, nullptr, "kernels")) {
        std::fprintf(stderr, "visual_review: renderer init failed\n");
        return 1;
    }
    if (render_path >= 0) renderer.SetPath((SplatRenderer::Path)render_path);

    // --- the sweep ---------------------------------------------------------
    //
    // These are chosen to hit the places a defect HIDES, not to be pretty:
    //   - below the seam: the underside of a closed body
    //   - grazing: the silhouette rim, where normals are nearly tangent
    //   - top-down: the pole, where a parametrisation degenerates
    //   - the ring of side views: any rotational asymmetry
    struct View { const char* name; float yaw; float pitch; };
    const View views[] = {
        { "00_front",     0.f,   0.f  },
        { "01_yaw45",    45.f,   0.f  },
        { "02_yaw90",    90.f,   0.f  },
        { "03_yaw135",  135.f,   0.f  },
        { "04_yaw180",  180.f,   0.f  },
        { "05_yaw225",  225.f,   0.f  },
        { "06_yaw270",  270.f,   0.f  },
        { "07_yaw315",  315.f,   0.f  },
        { "08_above",     0.f,  70.f  },
        { "09_below",     0.f, -70.f  },
        { "10_grazing",  20.f,   8.f  },
        { "11_lowside",  90.f, -20.f  },
    };
    const int nviews = int(sizeof(views) / sizeof(views[0]));

    // --- render each view --------------------------------------------------
    std::vector<std::vector<unsigned char>> shots;
    std::vector<std::string> names;
    Transform tr;   // identity: the body is already in world space
    const Mat4 mm = tr.ToMatrix();

    for (int i = 0; i < nviews; ++i) {
        const Camera cam = Orbit(fr, views[i].yaw, views[i].pitch, 42.f);
        const std::string ppm = outdir + "/" + views[i].name + ".ppm";
        renderer.RenderBody(*body, cam, true, true, 0.0f, mm.m);
        if (!renderer.SaveFramePPM(ppm)) {
            std::fprintf(stderr, "visual_review: save failed for %s\n", views[i].name);
            return 1;
        }
        int pw = 0, ph = 0;
        std::vector<unsigned char> rgb;
        if (!ReadPPM(ppm, pw, ph, rgb)) {
            std::fprintf(stderr, "visual_review: cannot read %s\n", ppm.c_str());
            return 1;
        }
        shots.push_back(rgb);
        names.push_back(views[i].name);
        const std::string png = outdir + "/" + views[i].name + ".png";
        WritePNG(png, pw, ph, rgb);
        std::printf("  %-12s yaw=%6.1f pitch=%6.1f -> %s.png\n",
                    views[i].name, views[i].yaw, views[i].pitch, views[i].name);
    }

    // --- the contact sheet -------------------------------------------------
    //
    // Every view tiled into ONE image. The question "is it right from all
    // sides?" is about the set; reading 12 files one at a time is how you
    // miss the one that is wrong.
    if (!contact.empty()) {
        const int cols = 4;
        const int rows = (nviews + cols - 1) / cols;
        const int cw = shots.empty() ? 0 : int(shots[0].size()) / 3 / w;
        const int ch = shots.empty() ? 0 : int(shots[0].size()) / 3 / h;
        if (cw > 0 && ch > 0) {
            const int sw = cw * cols, sh = ch * rows;
            std::vector<unsigned char> sheet(size_t(sw) * sh * 3, 0);
            for (int i = 0; i < nviews; ++i) {
                const int cx = (i % cols) * cw, cy = (i / cols) * ch;
                for (int y = 0; y < ch; ++y) {
                    const unsigned char* src = &shots[i][size_t(y) * cw * 3];
                    unsigned char* dst = &sheet[(size_t(cy + y) * sw + cx) * 3];
                    std::memcpy(dst, src, size_t(cw) * 3);
                }
            }
            if (WritePNG(contact, sw, sh, sheet)) {
                std::printf("contact sheet: %s (%dx%d, %d views)\n",
                            contact.c_str(), sw, sh, nviews);
            } else {
                std::fprintf(stderr, "visual_review: contact sheet write failed\n");
                return 1;
            }
        }
    }
    std::printf("visual_review: %d views in %s\n", nviews, outdir.c_str());
    return 0;
}
