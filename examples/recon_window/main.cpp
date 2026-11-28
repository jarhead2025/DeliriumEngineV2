// =============================================================================
// recon_window - LIVE window showing GENERIC element-system reconstruction.
//
// Five objects: four are reconstructed IN MEMORY from the element-system baker
// (ProceduralElementGenerator -> ElementSystem::BakeBody), NOT loaded from a
// file. Each is a recursive element tree of overlapping systems -> substance
// leaves:
//
//   orb    : a single ellipsoid "body" substance (closed-ish surface)
//   bump   : host plane + a gaussian node system  (TWO systems overlapping)
//   wave   : host plane + a wave heightfield system (two systems overlapping)
//   mound  : host plane + a gaussian node system  (same tree, second location)
//
// The FIFTH object is the classic torus, loaded from assets/torus.psf (the
// known-good file path) and translated to (0,0,-2.2) so it sits behind the
// 2x2 grid in the center of the frame. It proves the file pipeline and the
// generated pipeline render side by side in one window.
//
// The window shows the GPU rendering all five, live. You can drive the engine
// on whichever object the aim (window center) hits:
//
//   Build:  cmake --build build_delirium --target delirium_recon_window
//   Run:    ./build_delirium/delirium_recon_window       (from project root)
//
// Controls:
//   Mouse look .......... aim (window center is the hit point)
//   Wheel ............... zoom
//   LMB ................. poke (modal impulse; generated objects only --
//                         the file torus has no registered modes)
//   RMB ................. dent (RefineWhere + radial dent, permanent)
//   R ................... refine  (AMR split patches at aim)
//   C ................... coarsen (AMR merge at aim)
//   F ................... fracture (detach a fragment at aim)
//   G ................... restitch (Seams Build + FitAll)
//   T ................... weld (HarmonicWeld global seam relaxation)
//   1 ................... render path: layered march
//   2 ................... render path: GPU equation ray-march
//   3 ................... render path: classic splat (default)
//   V ................... toggle 10s recording (off at launch by default)
//   O ................... toggle auto-orbit
//   P ................... screenshot -> /tmp/recon_window_XXXX.ppm
//   Esc ................. quit
//   (WASD = move, Space/Ctrl = up/down, mouse look, wheel zoom)
//
// Placement note: the engine renders each body at its world position (it does
// NOT apply GameObject::Transform), so the four generated objects are baked
// at their world positions and the file torus is moved with PSFBody::Translate.
// Aiming raycasts every body in world space, so all five are targetable.
// =============================================================================
#include "delirium/core/Engine.h"
#include "delirium/core/GameObject.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/render/Camera.h"
#include "delirium/psf/Reconstructor.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/ai/ElementSystem.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <zlib.h>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

using namespace DeliRium;

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL

namespace {

// PSFBody-level fracture: the same plane split GameObject::Fracture does
// (nearest point -> outward normal -> split at the mid-plane -> detach the
// minority side). Operates on a raw PSFBody so the file-loaded torus gets
// the same treatment as the generated objects. Returns the detached fragment
// (nullptr if the split is degenerate).
// --- cut-plane cap (2026-10-29) --------------------------------------------
// The DIRECT plane-surface intersection the user asked for: every parent-side
// patch edge that straddles the cut plane yields its exact crossing point
// (linear interp between the two anchor positions -- the surface is near-planar
// at patch scale, so the chord crossing is the curve crossing to O(patch^2)).
// Those rim points + a centroid center become a FAN of planar quads (zero
// deformation coefficients, so P(u,v) is the exact bilinear plane -- and a
// bilinear quad whose 4 corners are coplanar stays on the plane, since the
// blend weights sum to 1). The fan renders through the EXISTING splat
// pipeline: the adaptive sample grid sparsely impacts the cap, and the
// infill/lod-stitch post passes AVERAGE the color between those sparse pixel
// impacts -- exactly "average out their colors in between sparsed out pixel
// impacts". Rim color is the parent's channel color evaluated at the crossing
// (u,v), so the cap blends into the surface it closes.
static float cap_pattern_eval(int idx, float u, float v) {
    switch (idx & 15) {
        case 0:  return 1.0f;
        case 1:  return u;
        case 2:  return v;
        case 3:  return u*v;
        case 4:  return u*u;
        case 5:  return v*v;
        case 6:  return u*u*v;
        case 7:  return u*v*v;
        case 8:  return std::cos(3.14159265358979f*u);
        case 9:  return std::cos(3.14159265358979f*v);
        case 10: return std::cos(6.28318530717959f*u);
        case 11: return std::cos(6.28318530717959f*v);
        case 12: return std::cos(3.14159265358979f*u)*std::cos(3.14159265358979f*v);
        case 13: return std::cos(6.28318530717959f*u)*std::cos(6.28318530717959f*v);
        case 14: return std::cos(9.42477796076938f*u);
        default: return std::cos(9.42477796076938f*v);
    }
}
static float cap_channel_color(const SurfaceRecord& rec, int ch, float u, float v) {
    const uint8_t* c = rec.color_pattern + ch*6;
    const int i0 = c[0] >> 4, i1 = c[0] & 15, i2 = c[1] >> 4, i3 = c[1] & 15;
    const int8_t* K = (const int8_t*)(c + 2);
    float r = (K[0]*(1.0f/63.0f)) * cap_pattern_eval(i0, u, v);
    r += (K[1]*(1.0f/63.0f)) * cap_pattern_eval(i1, u, v);
    r += (K[2]*(1.0f/63.0f)) * cap_pattern_eval(i2, u, v);
    r += (K[3]*(1.0f/63.0f)) * cap_pattern_eval(i3, u, v);
    return std::max(0.0f, std::min(1.0f, r));
}
// Fan cap closing the PARENT side's cut. rim = crossing points (world),
// n = cut-plane normal pointing AWAY from the camera (toward the fragment).
// One crossing point per straddling edge. `rgb` is the parent's color
// evaluated at the crossing (u,v) AT COLLECTION TIME -- before
// DetachPatches compacts the surface blob, when the patch index is still
// valid. (BuildCutCap runs after the detach, so it must NOT re-read
// parent.SurfaceBlob()[patch] -- those indices are stale.)
struct CapRimPt { Vec3 p; float u, v; uint32_t patch; uint8_t rgb[3]; };
// Build the cut-plane cap(s) closing the PARENT side's cut.
//
// A plane can intersect an object in SEVERAL closed loops (the torus -> an
// outer + inner loop; any object with a hole). The old code dumped every rim
// point into ONE centroid fan, which self-intersects into a star when there
// are two loops. This version:
//   1. dedups rim points (shared edges are found by both neighbors) via a
//      spatial hash -- O(n), not the old O(n^2) axis-aligned epsilon;
//   2. groups the deduped rim into connected LOOPS via union-find on the
//      "same patch" relation (rim points from one patch are connected, and
//      patches share edges, so a whole cut loop is one component);
//   3. fans EACH loop from its own centroid (sorted by angle in the cut
//      plane), so each loop is a clean non-self-intersecting fan.
//
// The cap is a flat disk on the cut plane. The parent keeps the NEAR side
// (dot(x,n)<d, closer to the camera), so the parent's surface near the cut
// line is in front of the coplanar cap and the cap shows only through the
// hole the detached far side left behind -> it seals the hole. The fragment
// gets a small depth_bias so it sits behind the cap. No bias on the cap.
//
// n = cut-plane normal pointing AWAY from the camera (toward the fragment).
// `rgb` on each rim point is the parent's color at the crossing (u,v),
// collected BEFORE DetachPatches compacts the surface blob.
static std::unique_ptr<PSFBody> BuildCutCap(PSFBody& parent,
                                            std::vector<CapRimPt> rim,   // by value
                                            const Vec3& n) {
    if (rim.size() < 3) return nullptr;

    // ---- 1. spatial-hash dedup (cell = 0.5% of the bbox span) -------------
    const Vec3 bmn = parent.BBoxMin(), bsp = parent.BBoxSpan();
    const float cell = 0.005f * std::max({bsp.x, bsp.y, bsp.z, 1e-6f});
    auto hkey = [&](const Vec3& p) -> long long {
        long long cx = (long long)std::floor(p.x / cell);
        long long cy = (long long)std::floor(p.y / cell);
        long long cz = (long long)std::floor(p.z / cell);
        return (cx * 73856093LL) ^ (cy * 19349663LL) ^ (cz * 83492791LL);
    };
    struct Dedup { Vec3 p; uint8_t rgb[3]; uint32_t patch; };
    std::vector<Dedup> ded;
    std::unordered_map<long long, size_t> h;
    for (const CapRimPt& r : rim) {
        const long long k = hkey(r.p);
        bool merged = false;
        auto it = h.find(k);
        if (it != h.end()) {
            const Dedup& d = ded[it->second];
            const float dx = d.p.x-r.p.x, dy = d.p.y-r.p.y, dz = d.p.z-r.p.z;
            if (dx*dx+dy*dy+dz*dz < cell*cell) { merged = true; }
        }
        if (merged) continue;
        h[k] = ded.size();
        ded.push_back(Dedup{ r.p, { r.rgb[0], r.rgb[1], r.rgb[2] }, r.patch });
    }
    const size_t nded = ded.size();
    if (nded < 3) return nullptr;

    // ---- 2. group the rim into loops by DISTANCE connectivity -------------
    // All rim points lie on the cut plane, so project to 2D (t1,t2) and union
    // any two points closer than a threshold. Consecutive rim points on a loop
    // are ~one patch-size apart (each patch's cut segment has 2 endpoints);
    // separate loops (e.g. a torus's inner + outer) are far apart. Threshold =
    // 3x the MEDIAN nearest-neighbor distance: adaptive to patch size, and it
    // bridges a loop without jumping to a distant one.
    // (The old "same patch" relation failed: a junction point shared by two
    // adjacent patches carries only ONE patch id after dedup, so it never
    // linked the two patches' groups -> each patch was its own 2-point
    // component -> no loop -> empty cap.)
    Vec3 t1g;
    if (std::fabs(n.y) < 0.9f) t1g = { n.z, 0.f, -n.x };
    else                        t1g = { 1.f, 0.f, 0.f };
    { const float l = std::sqrt(t1g.x*t1g.x+t1g.y*t1g.y+t1g.z*t1g.z);
      t1g.x/=l; t1g.y/=l; t1g.z/=l; }
    const Vec3 t2g{ n.y*t1g.z - n.z*t1g.y, n.z*t1g.x - n.x*t1g.z,
                    n.x*t1g.y - n.y*t1g.x };
    auto proj2 = [&](const Vec3& p) {
        return std::pair<float,float>{
            (p.x*t1g.x + p.y*t1g.y + p.z*t1g.z),
            (p.x*t2g.x + p.y*t2g.y + p.z*t2g.z) };
    };
    std::vector<std::pair<float,float>> prj(nded);
    for (int i = 0; i < (int)nded; ++i) prj[i] = proj2(ded[i].p);
    // median nearest-neighbor distance (adaptive loop spacing)
    std::vector<float> nnd(nded, 1e30f);
    for (int i = 0; i < (int)nded; ++i)
        for (int j = 0; j < (int)nded; ++j) if (j != i) {
            const float dx = prj[i].first-prj[j].first, dy = prj[i].second-prj[j].second;
            const float d = std::sqrt(dx*dx+dy*dy);
            if (d < nnd[i]) nnd[i] = d;
        }
    std::vector<float> nnsrt = nnd;
    std::sort(nnsrt.begin(), nnsrt.end());
    float thr = 3.0f * nnsrt[nnsrt.size()/2];
    if (thr < 1e-9f) thr = cell;   // degenerate: fall back to the dedup cell
    std::vector<int> uf(nded);
    for (int i = 0; i < (int)nded; ++i) uf[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) uf[b] = a; };
    for (int i = 0; i < (int)nded; ++i)
        for (int j = i+1; j < (int)nded; ++j) {
            const float dx = prj[i].first-prj[j].first, dy = prj[i].second-prj[j].second;
            if (std::sqrt(dx*dx+dy*dy) < thr) unite(i, j);
        }
    std::unordered_map<int, std::vector<int>> comps;
    for (int i = 0; i < (int)nded; ++i) comps[find(i)].push_back(i);
    std::vector<std::vector<int>> loops;
    for (auto& kv : comps) if (kv.second.size() >= 3) loops.push_back(kv.second);
    if (loops.empty()) return nullptr;

    // ---- 3. fan each loop from its own centroid ---------------------------
    std::vector<AnchorRecord> A;
    std::vector<SurfaceRecord> S;
    auto mk_anchor = [&](const Vec3& p, const uint8_t rgb[3]) -> uint16_t {
        AnchorRecord ar{};
        auto q = [&](float lo, float sp, float v) -> uint16_t {
            float t = sp > 1e-12f ? (v - lo) / sp : 0.5f;
            t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
            return (uint16_t)(t * 65535.0f + 0.5f);
        };
        ar.pos[0] = q(bmn.x, bsp.x, p.x);
        ar.pos[1] = q(bmn.y, bsp.y, p.y);
        ar.pos[2] = q(bmn.z, bsp.z, p.z);
        // displacement direction = cut normal (h=0 -> never displaces; it only
        // feeds the smooth-normal field, and the kernel's flip-to-face-camera
        // makes the cap two-sided anyway)
        const int8_t nx = (int8_t)(std::max(-1.0f, std::min(1.0f, n.x)) * 127.0f);
        const int8_t ny = (int8_t)(std::max(-1.0f, std::min(1.0f, n.y)) * 127.0f);
        const int8_t nz = (int8_t)(std::max(-1.0f, std::min(1.0f, n.z)) * 127.0f);
        ar.normal[0] = (uint8_t)nx; ar.normal[1] = (uint8_t)ny; ar.normal[2] = (uint8_t)nz;
        ar.color[0] = rgb[0]; ar.color[1] = rgb[1]; ar.color[2] = rgb[2];
        A.push_back(ar);
        return (uint16_t)(A.size() - 1);
    };
    // rim anchors, deduped by position (shared vertices reuse one anchor).
    // Use a FINE cell (0.25x the rim-dedup cell): two rim points that
    // survived dedup are >= `cell` apart, so they can never share a fine cell
    // -> no accidental merge of distinct points.
    const float fcell = cell * 0.25f;
    auto fkey = [&](const Vec3& p) -> long long {
        long long cx = (long long)std::floor(p.x / fcell);
        long long cy = (long long)std::floor(p.y / fcell);
        long long cz = (long long)std::floor(p.z / fcell);
        return (cx * 73856093LL) ^ (cy * 19349663LL) ^ (cz * 83492791LL);
    };
    std::unordered_map<long long, uint16_t> anchor_hash;
    std::vector<uint16_t> rim_a(nded);
    for (int i = 0; i < (int)nded; ++i) {
        const long long k = fkey(ded[i].p);
        auto it = anchor_hash.find(k);
        if (it != anchor_hash.end()) { rim_a[i] = it->second; continue; }
        const uint8_t rgb[3] = { ded[i].rgb[0], ded[i].rgb[1], ded[i].rgb[2] };
        const uint16_t a = mk_anchor(ded[i].p, rgb);
        anchor_hash[k] = a;
        rim_a[i] = a;
    }
    // K encoding: color c in [0,1] -> int8 K = round(c*63) in [0,63]
    // (decoded as (K/63)*pattern). Storing the raw 0..255 value made K
    // negative -> clamped to black (the old "dark band" bug).
    auto k8 = [](uint8_t c) -> uint8_t {
        const int k = (int)(c * 63.0f / 255.0f + 0.5f);
        return (uint8_t)(int8_t)(k > 127 ? 127 : k);
    };
    for (auto& loop : loops) {
        const size_t m = loop.size();
        // loop centroid (on the cut plane: all rim points satisfy dot(x,n)=d)
        Vec3 cen{};
        for (int idx : loop) { cen.x += ded[idx].p.x; cen.y += ded[idx].p.y; cen.z += ded[idx].p.z; }
        const float inv = 1.0f / (float)m;
        cen.x *= inv; cen.y *= inv; cen.z *= inv;
        // sort by angle around the centroid in the cut plane (tangent basis)
        Vec3 t1;
        if (std::fabs(n.y) < 0.9f) t1 = { n.z, 0.f, -n.x };   // cross(n, up)
        else                        t1 = { 1.f, 0.f, 0.f };
        const float t1l = std::sqrt(t1.x*t1.x + t1.y*t1.y + t1.z*t1.z);
        t1.x /= t1l; t1.y /= t1l; t1.z /= t1l;
        const Vec3 t2{ n.y*t1.z - n.z*t1.y, n.z*t1.x - n.x*t1.z,
                       n.x*t1.y - n.y*t1.x };   // cross(n, t1), orthonormal
        auto ang = [&](int idx) {
            const Vec3& p = ded[idx].p;
            const float ax = (p.x-cen.x)*t1.x + (p.y-cen.y)*t1.y + (p.z-cen.z)*t1.z;
            const float ay = (p.x-cen.x)*t2.x + (p.y-cen.y)*t2.y + (p.z-cen.z)*t2.z;
            return std::atan2(ay, ax);
        };
        std::sort(loop.begin(), loop.end(), [&](int a, int b) { return ang(a) < ang(b); });
        // shared center anchor: average of the loop's rim colors
        float cr = 0, cg = 0, cb = 0;
        for (int idx : loop) { cr += (float)A[rim_a[idx]].color[0];
                               cg += (float)A[rim_a[idx]].color[1];
                               cb += (float)A[rim_a[idx]].color[2]; }
        const uint8_t ccen[3] = { (uint8_t)(cr * inv), (uint8_t)(cg * inv), (uint8_t)(cb * inv) };
        const uint16_t center = mk_anchor(cen, ccen);
        for (size_t i = 0; i < m; ++i) {
            const int ia = loop[i], ib = loop[(i + 1) % m];
            const Vec3& pa = ded[ia].p, &pb = ded[ib].p;
            // winding: cross(pb - pa, cen - pa) should point along n
            const Vec3 e1{ pb.x-pa.x, pb.y-pa.y, pb.z-pa.z };
            const Vec3 e2{ cen.x-pa.x, cen.y-pa.y, cen.z-pa.z };
            const Vec3 crv{ e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z,
                            e1.x*e2.y - e1.y*e2.x };
            const bool flip = (crv.x*n.x + crv.y*n.y + crv.z*n.z) < 0.f;
            // per-quad flat color: average of the two rim colors
            const uint8_t cc[3] = {
                (uint8_t)((A[rim_a[ia]].color[0] + A[rim_a[ib]].color[0]) * 0.5f),
                (uint8_t)((A[rim_a[ia]].color[1] + A[rim_a[ib]].color[1]) * 0.5f),
                (uint8_t)((A[rim_a[ia]].color[2] + A[rim_a[ib]].color[2]) * 0.5f)
            };
            SurfaceRecord sr{};
            sr.anchor_idx[0] = center;
            sr.anchor_idx[1] = flip ? rim_a[ia] : rim_a[ib];
            sr.anchor_idx[2] = flip ? rim_a[ib] : rim_a[ia];
            sr.anchor_idx[3] = center;
            sr.color_pattern[0] = 0x00;  sr.color_pattern[1] = 0x00;
            sr.color_pattern[2] = k8(cc[0]);
            sr.color_pattern[6] = 0x00;  sr.color_pattern[7] = 0x00;
            sr.color_pattern[8] = k8(cc[1]);
            sr.color_pattern[12] = 0x00; sr.color_pattern[13] = 0x00;
            sr.color_pattern[14] = k8(cc[2]);
            S.push_back(sr);
        }
    }
    std::printf("  [cap] %zu rim -> %zu dedup, %zu loop(s), %zu quads\n",
                rim.size(), nded, loops.size(), S.size());
    std::vector<int8_t> stitch(S.size() * 8, 0);
    return PSFBody::FromRecords(std::move(A), std::move(S), std::move(stitch),
                                parent.BBoxMin(), parent.BBoxSpan(),
                                parent.StitchScale());
}

// Fracture: camera-parallel cut plane + detached fragment + cap on the
// parent side. Returns the fragment and (out) the cap.
struct FractureResult { std::unique_ptr<PSFBody> fragment, cap; };
FractureResult FractureBody(PSFBody& body, const Vec3& p, float energy,
                            const Camera& cam) {
    FractureResult out;
    if (energy <= 0) return out;
    SurfaceHit hit;
    if (!body.ClosestPointEx(p, hit)) return out;
    auto& pool = body.AnchorsConst();
    Vec3 cen{}; uint32_t N = 0;
    for (uint32_t a = 0; a < pool.Count(); ++a) {
        Vec3 q = pool.Position(a);
        cen.x += q.x; cen.y += q.y; cen.z += q.z; ++N;
    }
    if (N) { cen.x /= N; cen.y /= N; cen.z /= N; }
    // CAMERA-PARALLEL CUT PLANE (user request 2026-10-29): the plane normal is
    // the camera's forward direction (position -> target), so the cut plane is
    // PARALLEL to the image plane regardless of where the object sits in view.
    // The OLD normal was normalize(hit.p - cen) -- radial from the centroid,
    // which is only camera-parallel when the object's centroid lies on the
    // center-screen ray. For the torus (centroid = the hole) that gave a
    // radial cross-section, not a camera-parallel slice. The plane still
    // passes through the midpoint (cen + (hit-cen)/2), so the cut is roughly
    // centered on the object. The far side (away from the camera,
    // dot(x,n) > d) is the detached fragment.
    const Vec3 fwd{ cam.target.x - cam.position.x,
                    cam.target.y - cam.position.y,
                    cam.target.z - cam.position.z };
    const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
    Vec3 n;
    if (fl > 1e-9f) {
        n = { fwd.x/fl, fwd.y/fl, fwd.z/fl };
    } else {
        // degenerate camera: fall back to the radial normal
        n = { hit.p.x - cen.x, hit.p.y - cen.y, hit.p.z - cen.z };
        const float nl = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        if (nl < 1e-9f) {
            n = body.Patches()[hit.patch].EvaluateNormal(hit.u, hit.v);
        } else { n.x/=nl; n.y/=nl; n.z/=nl; }
    }
    const Vec3 mid{ cen.x + (hit.p.x - cen.x) * 0.5f,
                    cen.y + (hit.p.y - cen.y) * 0.5f,
                    cen.z + (hit.p.z - cen.z) * 0.5f };
    const float d = mid.x*n.x + mid.y*n.y + mid.z*n.z;

    auto anchor_side = [&](uint16_t a) {
        const Vec3 q = pool.Position(a);
        return (q.x*n.x + q.y*n.y + q.z*n.z) - d;   // >0 = fragment side
    };
    auto patch_side = [&](uint32_t pid) {
        const SurfaceRecord& s = body.SurfaceBlob()[pid];
        Vec3 ctr{};
        for (int k = 0; k < 4; ++k) {
            Vec3 a = pool.Position(s.anchor_idx[k]);
            ctr.x += a.x * 0.25f; ctr.y += a.y * 0.25f; ctr.z += a.z * 0.25f;
        }
        return (ctr.x*n.x + ctr.y*n.y + ctr.z*n.z) - d;
    };
    std::vector<uint32_t> minority;
    // RIM: the DIRECT plane-surface intersections. For every PARENT-side
    // patch, each edge whose endpoints sit on opposite sides of the cut
    // plane yields its exact crossing (linear interp of the anchor
    // positions + params). These become the cap's rim. Deduped by position
    // (shared edges are found by both neighbors). Colors are evaluated HERE,
    // before DetachPatches compacts the surface blob.
    std::vector<CapRimPt> rim;
    const float rim_eps = 1e-3f * body.BBoxSpan().x;
    auto rim_add = [&](const CapRimPt& rp) {
        for (const CapRimPt& e : rim)
            if (std::fabs(e.p.x-rp.p.x) < rim_eps &&
                std::fabs(e.p.y-rp.p.y) < rim_eps &&
                std::fabs(e.p.z-rp.p.z) < rim_eps) return;
        rim.push_back(rp);
    };
    static const int EP[4][2] = {{0,1},{1,2},{2,3},{3,0}};
    for (uint32_t pid = 0; pid < body.SurfaceBlob().size(); ++pid) {
        if (patch_side(pid) > 0) { minority.push_back(pid); continue; }
        // parent-side patch: find edges crossing the plane
        const SurfaceRecord& s = body.SurfaceBlob()[pid];
        for (int e = 0; e < 4; ++e) {
            const uint16_t a0 = s.anchor_idx[EP[e][0]], a1 = s.anchor_idx[EP[e][1]];
            const float s0 = anchor_side(a0), s1 = anchor_side(a1);
            if ((s0 <= 0.f && s1 > 0.f) || (s0 > 0.f && s1 <= 0.f)) {
                const float t = s0 / (s0 - s1);   // 0 at a0, 1 at a1
                const Vec3 q0 = pool.Position(a0), q1 = pool.Position(a1);
                CapRimPt rp{};
                rp.p = { q0.x + (q1.x-q0.x)*t, q0.y + (q1.y-q0.y)*t,
                         q0.z + (q1.z-q0.z)*t };
                // slot (u,v): 0=(0,0) 1=(0,1) 2=(1,1) 3=(1,0). The crossing's
                // (u,v) is the lerp of the two endpoint slots' (u,v) at the
                // same t (the edge is a straight line in (u,v) space, exact).
                static const float SU[4] = {0.f, 0.f, 1.f, 1.f};
                static const float SV[4] = {0.f, 1.f, 1.f, 0.f};
                rp.u = SU[a0] + (SU[a1]-SU[a0]) * t;
                rp.v = SV[a0] + (SV[a1]-SV[a0]) * t;
                rp.patch = pid;
                rp.rgb[0] = (uint8_t)(cap_channel_color(s, 0, rp.u, rp.v) * 255.0f);
                rp.rgb[1] = (uint8_t)(cap_channel_color(s, 1, rp.u, rp.v) * 255.0f);
                rp.rgb[2] = (uint8_t)(cap_channel_color(s, 2, rp.u, rp.v) * 255.0f);
                rim_add(rp);
            }
        }
    }
    if (minority.empty() || minority.size() == body.SurfaceBlob().size())
        return out;

    out.fragment = body.DetachPatches(minority);
    body.Seams().Build(body);
    // cap closes the PARENT side's cut; its normal faces the CAMERA
    // (=-n, since n points away from the camera toward the fragment)
    if (rim.size() >= 3)
        out.cap = BuildCutCap(body, rim, { -n.x, -n.y, -n.z });
    return out;
}

// =============================================================================
// 10-SECOND RECORDER -- captures the final framebuffer every frame for 10s,
// saves each as a PNG, then encodes to MP4 with ffmpeg (spawned in a
// background thread so the window keeps running). If ffmpeg is not on PATH
// the PNG frames are kept and a command line is printed.
//
// Why this exists: the fracture demo has "objects flying in and out of the
// toroids at really fast speeds" that are too fast to describe in words. A
// 10s recording lets us scrub frame-by-frame and see exactly what is moving.
// =============================================================================
namespace {

// ---- minimal PNG writer (8-bit RGB, zlib deflate; no SDL_image needed) ----
void png_u32(std::vector<unsigned char>& v, uint32_t x) {
    v.push_back((unsigned char)(x >> 24)); v.push_back((unsigned char)(x >> 16));
    v.push_back((unsigned char)(x >> 8));  v.push_back((unsigned char)x);
}
void png_chunk(std::vector<unsigned char>& v, const char* type,
               const unsigned char* data, size_t n) {
    png_u32(v, (uint32_t)n);
    size_t t0 = v.size();
    for (int i = 0; i < 4; ++i) v.push_back((unsigned char)type[i]);
    v.insert(v.end(), data, data + n);
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, v.data() + t0, (uInt)(v.size() - t0));
    png_u32(v, (uint32_t)crc);
}
bool WritePNG(const std::string& path, int w, int h,
              const unsigned char* rgba) {
    std::vector<unsigned char> out;
    out.reserve(size_t(w)*h*3 + 256);
    out.insert(out.end(), {0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A});
    { std::vector<unsigned char> t; png_u32(t, (uint32_t)w); png_u32(t, (uint32_t)h);
      t.push_back(8); t.push_back(2); t.push_back(0); t.push_back(0); t.push_back(0);
      png_chunk(out, "IHDR", t.data(), t.size()); }
    std::vector<unsigned char> raw;
    raw.reserve(size_t(w)*h*4);
    for (int y = 0; y < h; ++y) {
        raw.push_back(0); // filter: none
        const unsigned char* row = rgba + size_t(y)*w*4;
        for (int x = 0; x < w; ++x) {
            raw.push_back(row[x*4]); raw.push_back(row[x*4+1]); raw.push_back(row[x*4+2]);
        }
    }
    uLongf clen = compressBound((uLong)raw.size());
    std::vector<unsigned char> comp(clen);
    if (compress2(comp.data(), &clen, raw.data(), (uLong)raw.size(), 6) != Z_OK)
        return false;
    comp.resize(clen);
    png_chunk(out, "IDAT", comp.data(), comp.size());
    png_chunk(out, "IEND", nullptr, 0);
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    size_t wr = std::fwrite(out.data(), 1, out.size(), fp);
    std::fclose(fp);
    return wr == out.size();
}

struct Recorder {
    std::string dir;
    int w = 0, h = 0;
    std::chrono::steady_clock::time_point t0;
    int n = 0;
    bool ffmpeg_ok = false;

    void start(int W, int H, const std::string& base) {
        w = W; h = H; n = 0;
        dir = base + "_rec";
        std::system(("mkdir -p " + dir).c_str());
        t0 = std::chrono::steady_clock::now();
        std::printf("  [rec] recording 10s -> %s/frame_%%04d.png\n", dir.c_str());
    }
    void frame(const std::vector<unsigned char>& rgba) {
        char p[512];
        std::snprintf(p, sizeof(p), "%s/frame_%04d.png", dir.c_str(), n);
        if (WritePNG(p, w, h, rgba.data())) ++n;
    }
    void finalize() {
        if (n == 0) { std::printf("  [rec] no frames captured\n"); return; }
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        int fps = (n > 0) ? (int)(n / secs + 0.5f) : 60;
        if (fps < 1) fps = 1; if (fps > 240) fps = 240;
        std::printf("  [rec] %d frames in %.2fs -> %d fps\n", n, secs, fps);
        std::string out = dir + ".mp4";
        // ffmpeg on PATH? (spawned in a background thread so the window keeps
        // running while it encodes)
        ffmpeg_ok = (std::system("command -v ffmpeg > /dev/null 2>&1") == 0);
        if (ffmpeg_ok) {
            std::string cmd = "ffmpeg -y -framerate " + std::to_string(fps) +
                " -i " + dir + "/frame_%04d.png -c:v libx264 -pix_fmt yuv420p " +
                out + " > /dev/null 2>&1";
            std::thread([cmd]{ std::system(cmd.c_str()); }).detach();
            std::printf("  [rec] encoding MP4 in background -> %s\n", out.c_str());
        } else {
            std::printf("  [rec] ffmpeg not found; keeping PNG frames in %s\n", dir.c_str());
            std::printf("  [rec] to make the video: ffmpeg -framerate %d -i %s/frame_%%04d.png -c:v libx264 -pix_fmt yuv420p %s\n",
                        fps, dir.c_str(), out.c_str());
        }
    }
};

} // namespace


// =============================================================================
// HUD OVERLAY — SDL pixel-glyph overlay drawn AFTER the PSF frame, before
// Present. Self-contained bitmap font + helpers; no external font dependency.
// Shows: FPS/frame-time, camera position/target, render mode, live aim-target
// stats (patches/anchors/seam gap), controls legend, and action toasts.
// =============================================================================

namespace {
constexpr int FONT_ROWS = 7, FONT_COLS = 5, FONT_H = 7, FONT_W = 5;
constexpr int FONT_FIRST = 32, FONT_LAST = 126, FONT_COUNT = FONT_LAST - FONT_FIRST + 1;
constexpr int FONT_ADV = FONT_COLS + 1;   // glyph width + 1px spacing

// Classic public-domain 5x7 ASCII font. One byte per COLUMN (5 columns),
// bit 0 = TOP row. Covers printable ASCII 32..126 (95 glyphs). Used to build
// a single SDL_Texture atlas drawn with SDL_RenderCopy (no per-pixel fills).
constexpr uint8_t FONT[95][5] = {
    // 32 space
    {0x00,0x00,0x00,0x00,0x00},
    // 33 !
    {0x00,0x00,0x5F,0x00,0x00},
    // 34 "
    {0x00,0x07,0x00,0x07,0x00},
    // 35 #
    {0x14,0x7F,0x14,0x7F,0x14},
    // 36 $
    {0x24,0x2A,0x7F,0x2A,0x12},
    // 37 %
    {0x23,0x13,0x08,0x64,0x62},
    // 38 &
    {0x36,0x49,0x55,0x22,0x50},
    // 39 '
    {0x00,0x05,0x03,0x00,0x00},
    // 40 (
    {0x00,0x1C,0x22,0x41,0x00},
    // 41 )
    {0x00,0x41,0x22,0x1C,0x00},
    // 42 *
    {0x08,0x2A,0x1C,0x2A,0x08},
    // 43 +
    {0x08,0x08,0x3E,0x08,0x08},
    // 44 ,
    {0x00,0x50,0x30,0x00,0x00},
    // 45 -
    {0x08,0x08,0x08,0x08,0x08},
    // 46 .
    {0x00,0x60,0x60,0x00,0x00},
    // 47 /
    {0x20,0x10,0x08,0x04,0x02},
    // 48 0
    {0x3E,0x51,0x49,0x45,0x3E},
    // 49 1
    {0x00,0x42,0x7F,0x40,0x00},
    // 50 2
    {0x42,0x61,0x51,0x49,0x46},
    // 51 3
    {0x21,0x41,0x45,0x4B,0x31},
    // 52 4
    {0x18,0x14,0x12,0x7F,0x10},
    // 53 5
    {0x27,0x45,0x45,0x45,0x39},
    // 54 6
    {0x3C,0x4A,0x49,0x49,0x30},
    // 55 7
    {0x01,0x71,0x09,0x05,0x03},
    // 56 8
    {0x36,0x49,0x49,0x49,0x36},
    // 57 9
    {0x06,0x49,0x49,0x29,0x1E},
    // 58 :
    {0x00,0x36,0x36,0x00,0x00},
    // 59 ;
    {0x00,0x56,0x36,0x00,0x00},
    // 60 <
    {0x00,0x08,0x14,0x22,0x41},
    // 61 =
    {0x14,0x14,0x14,0x14,0x14},
    // 62 >
    {0x41,0x22,0x14,0x08,0x00},
    // 63 ?
    {0x02,0x01,0x51,0x09,0x06},
    // 64 @
    {0x32,0x49,0x79,0x41,0x3E},
    // 65 A
    {0x7F,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x7F},
    // 66 B
    {0x7E,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x7E},
    // 67 C
    {0x3E,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x3E},
    // 68 D
    {0x7C,0x42,0x41,0x41,0x41,0x41,0x41,0x42,0x7C},
    // 69 E
    {0x7F,0x49,0x49,0x49,0x49,0x49,0x49,0x49,0x41},
    // 70 F
    {0x7F,0x09,0x09,0x09,0x09,0x09,0x09,0x09,0x01},
    // 71 G
    {0x3E,0x41,0x41,0x41,0x51,0x33,0x17,0x17,0x3E},
    // 72 H
    {0x7F,0x08,0x08,0x08,0x08,0x08,0x08,0x08,0x7F},
    // 73 I
    {0x00,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x00},
    // 74 J
    {0x07,0x02,0x02,0x02,0x02,0x02,0x02,0x44,0x38},
    // 75 K
    {0x7F,0x10,0x18,0x14,0x12,0x11,0x10,0x10,0x10},
    // 76 L
    {0x7F,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40},
    // 77 M
    {0x7F,0x02,0x0C,0x02,0x02,0x02,0x02,0x02,0x7F},
    // 78 N
    {0x7F,0x04,0x08,0x10,0x20,0x40,0x80,0x40,0x7F},
    // 79 O
    {0x3E,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x3E},
    // 80 P
    {0x7F,0x09,0x09,0x09,0x09,0x09,0x09,0x09,0x00},
    // 81 Q
    {0x3E,0x41,0x41,0x41,0x41,0x41,0x51,0x29,0x16},
    // 82 R
    {0x7F,0x09,0x09,0x09,0x09,0x11,0x21,0x41,0x00},
    // 83 S
    {0x46,0x49,0x49,0x49,0x49,0x49,0x49,0x49,0x31},
    // 84 T
    {0x01,0x01,0x7F,0x01,0x01,0x01,0x01,0x01,0x01},
    // 85 U
    {0x3F,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x3F},
    // 86 V
    {0x1F,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x1F},
    // 87 W
    {0x3F,0x20,0x20,0x20,0x28,0x28,0x28,0x20,0x3F},
    // 88 X
    {0x63,0x14,0x08,0x14,0x63,0x00,0x00,0x00,0x00},
    // 89 Y
    {0x07,0x08,0x08,0x08,0x70,0x08,0x08,0x08,0x07},
    // 90 Z
    {0x61,0x51,0x49,0x45,0x43,0x41,0x41,0x41,0x7F},
    // 91 [
    {0x00,0x7F,0x41,0x41,0x41,0x41,0x41,0x41,0x7F},
    // 92 \
    {0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00,0x00},
    // 93 ]
    {0x00,0x7F,0x40,0x40,0x40,0x40,0x40,0x40,0x7F},
    // 94 ^
    {0x10,0x28,0x44,0x00,0x00,0x00,0x00,0x00,0x00},
    // 95 _
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x7F},
    // 96 `
    {0x08,0x14,0x22,0x00,0x00,0x00,0x00,0x00,0x00},
    // 97 a
    {0x00,0x00,0x46,0x49,0x49,0x49,0x49,0x49,0x32},
    // 98 b
    {0x3F,0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x3F},
    // 99 c
    {0x00,0x00,0x34,0x48,0x48,0x48,0x48,0x48,0x30},
    // 100 d
    {0x3F,0x08,0x08,0x08,0x08,0x08,0x08,0x08,0x3F},
    // 101 e
    {0x00,0x00,0x38,0x44,0x44,0x44,0x44,0x44,0x38},
    // 102 f
    {0x18,0x24,0x24,0x7F,0x24,0x24,0x24,0x24,0x00},
    // 103 g
    {0x00,0x00,0x32,0x49,0x49,0x49,0x49,0x32,0x2C},
    // 104 h
    {0x3F,0x08,0x08,0x08,0x08,0x08,0x08,0x08,0x7F},
    // 105 i
    {0x00,0x00,0x06,0x09,0x09,0x09,0x09,0x09,0x06},
    // 106 j
    {0x00,0x00,0x86,0x89,0x89,0x89,0x89,0x89,0x76},
    // 107 k
    {0x3F,0x10,0x18,0x14,0x12,0x11,0x10,0x10,0x10},
    // 108 l
    {0x06,0x09,0x09,0x09,0x09,0x09,0x09,0x09,0x06},
    // 109 m
    {0x00,0x00,0x66,0x89,0x89,0x89,0x89,0x89,0x66},
    // 110 n
    {0x00,0x00,0x3E,0x41,0x41,0x41,0x41,0x41,0x41},
    // 111 o
    {0x00,0x00,0x36,0x49,0x49,0x49,0x49,0x49,0x36},
    // 112 p
    {0x00,0x00,0x7C,0x12,0x11,0x11,0x11,0x11,0x11,0x7E},
    // 113 q
    {0x00,0x00,0x3E,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x7C},
    // 114 r
    {0x00,0x00,0x44,0x71,0x41,0x41,0x41,0x41,0x41},
    // 115 s
    {0x00,0x00,0x38,0x44,0x44,0x44,0x44,0x44,0x38},
    // 116 t
    {0x10,0x10,0x7F,0x10,0x10,0x10,0x10,0x10,0x3E},
    // 117 u
    {0x00,0x00,0x3F,0x20,0x20,0x20,0x20,0x20,0x3F},
    // 118 v
    {0x00,0x00,0x1F,0x20,0x20,0x20,0x20,0x20,0x1F},
    // 119 w
    {0x00,0x00,0x3F,0x20,0x20,0x28,0x28,0x20,0x3F},
    // 120 x
    {0x00,0x00,0x63,0x14,0x08,0x14,0x63,0x00,0x00},
    // 121 y
    {0x00,0x00,0x37,0x08,0x08,0x08,0x08,0x08,0x77,0x07},
    // 122 z
    {0x00,0x00,0x73,0x41,0x41,0x41,0x41,0x41,0x3B},
    // 123 {
    {0x00,0x00,0x1C,0x22,0x22,0x22,0x22,0x22,0x1C},
    // 124 |
    {0x00,0x00,0x09,0x09,0x09,0x09,0x09,0x09,0x09},
    // 125 }
    {0x00,0x00,0x46,0x28,0x28,0x28,0x28,0x28,0x46},
    // 126 ~
    {0x00,0x00,0x40,0x42,0x14,0x08,0x40,0x00,0x00},
};
static_assert(sizeof(FONT) == sizeof(FONT[0]) * FONT_COUNT, "FONT row width must be 5");

// Draw a filled rectangle (SDL_RenderFillRect).
inline void hud_box(SDL_Renderer* r, int x, int y, int w, int h,
                    Uint8 rr, Uint8 gg, Uint8 bb) {
    SDL_SetRenderDrawColor(r, rr, gg, bb, 255);
    SDL_Rect rect{x, y, w, h};
    SDL_RenderFillRect(r, &rect);
}

// Draw one glyph at (x,y) scaled by s, in color (rr,gg,bb).
inline void hud_glyph(SDL_Renderer* r, int x, int y, char c, int s,
                      Uint8 rr, Uint8 gg, Uint8 bb) {
    if (c < 32 || c > 126) return;
    const uint8_t* row = FONT[c - 32];
    SDL_SetRenderDrawColor(r, rr, gg, bb, 255);
    for (int ry = 0; ry < FONT_ROWS; ++ry) {
        uint8_t bits = row[ry];
        for (int cx = 0; cx < FONT_COLS; ++cx) {
            if (bits & (0x10 >> cx)) {
                SDL_Rect p{x + cx * s, y + ry * s, s, s};
                SDL_RenderFillRect(r, &p);
            }
        }
    }
}

// Draw a null-terminated string at (x,y), scale s, color (rr,gg,bb).
// Returns the x advance (width) of the drawn text.
inline int hud_text(SDL_Renderer* r, int x, int y, const char* str, int s,
                    Uint8 rr, Uint8 gg, Uint8 bb) {
    int ox = x;
    for (const char* p = str; *p; ++p) {
        hud_glyph(r, ox, y, *p, s, rr, gg, bb);
        ox += (FONT_COLS + 1) * s;
    }
    return ox - x;
}

// Draw text with a semi-transparent dark backing box (for readability over
// the render). box_w is the full text width; box_h the line height.
inline void hud_text_panel(SDL_Renderer* r, int x, int y, const char* str,
                           int s, Uint8 rr, Uint8 gg, Uint8 bb, int box_w) {
    hud_box(r, x - 2 * s, y - 2 * s, box_w + 4 * s, FONT_H * s + 4 * s,
            0, 0, 0);
    hud_text(r, x, y, str, s, rr, gg, bb);
}
} // namespace

struct Obj {
    GameObject* go = nullptr;      // null for file-loaded bodies (torus)
    PSFBody*    body = nullptr;    // raw body (torus); go->Body() otherwise
    PSFBody*    fragment = nullptr;// fracture output (kept alive by Win::frags)
    PSFBody*    cap = nullptr;     // cut-plane cap on the parent side (kept alive by Win::frags)
    Vec3  center{0,0,0};           // world position (baked in / translated)
    float diag = 1.f;
    const char* label = "";
    PSFBody& body_ref() const { return body ? *body : go->Body(); }
};

struct Win {
    std::vector<Obj> objs;
    std::vector<std::unique_ptr<PSFBody>> frags;   // keeps fragments alive
    SplatRenderer* renderer = nullptr;
    int w = 0, h = 0;
    int shot = 0;
    bool orbit = false;
    bool show_cap = true;  // X toggles the cut-plane cap (A/B the cap)
    int render_path = 4;   // TriInfill GPU (only path)
    Recorder rec;          // 10s frame recorder (V to start, auto-starts once)
    bool rec_active = false;

    // HUD state: aim target index (-1 = nothing aimed at), action toast text
    // + expiry frame counter (toast fades after toast_ttl frames).
    int  aim_idx = -1;
    char toast[128] = "";
    int  toast_ttl = 0;
    bool hud_visible = true;
    bool physics_detail = true;   // toggle physics/material panel depth (M key)
    void set_toast(const char* msg) {
        std::snprintf(toast, sizeof(toast), "%s", msg ? msg : "");
        toast_ttl = 120;   // ~2s at 60 FPS
    }

    // Which object does the window-center ray hit (nearest)? Bodies are baked
    // at their world positions, so a single ray tests all of them in world
    // space. Hit point is returned in world space.
    int aim(const Camera& cam, Vec3& out_p) const {
        Vec3 o, d;
        cam.PixelRay(w * 0.5f, h * 0.5f, w, h, o, d);
        int best = -1; float best_t = 1e30f;
        for (int i = 0; i < (int)objs.size(); ++i) {
            SurfaceHit hit;
            if (objs[i].body_ref().Raycast(o, d, hit) && hit.t < best_t) {
                best_t = hit.t; best = i; out_p = hit.p;
            }
        }
        return best;
    }

    void status(int i, const char* what) {
        PSFBody& b = objs[i].body_ref();
        b.Seams().Build(b);
        const float g = b.Seams().MaxGap(b);
        std::printf("  [%s %s] patches=%zu anchors=%zu seam=%.4f%% diag\n",
                    what, objs[i].label, b.Patches().size(),
                    b.AnchorBlob().size(), g / objs[i].diag * 100.f);
    }

    // TriInfill GPU is the ONLY render path — the future AI reconstructor
    // targets this path exclusively. Other paths (march/raymarch/splat) are
    // legacy and produce artifacts on generated objects; they are removed.
    void set_render_path(int /*p*/) {
        render_path = 4;
        ::setenv("DELIRIUM_TRI", "1", 1);
        ::unsetenv("DELIRIUM_MARCH");
        ::unsetenv("DELIRIUM_RAYMARCH");
        std::printf("  [path] TriInfill GPU (only render path)\n");
    }
};

// =============================================================================
// HUD RENDER — draws the in-window overlay after the PSF frame. Layout:
//   top-left   : title + FPS/frame-time + render mode
//   top-right  : camera position/target + distance
//   bottom-left: controls legend (key bindings)
//   bottom-right: live aim-target stats (label, patches, anchors, seam gap)
//   center     : action toast (fades after toast_ttl frames)
// =============================================================================
void draw_hud(SDL_Renderer* r, Win* S, const Camera& cam) {
    if (!r) return;
    const int sc = 2;   // glyph scale
    const int pad = 8;  // margin from edges
    if (!S->hud_visible) return;   // H toggles full HUD off/on

    // FPS/frame-time computed from frame timestamps so the HUD stays in sync
    // without relying on the console printf cadence.
    static auto fps_t0 = std::chrono::steady_clock::now();
    static int fps_frames = 0;
    static float fps_val = 60.0f, frame_ms = 16.6f;
    {
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - fps_t0).count();
        ++fps_frames;
        if (dt >= 0.25) {
            fps_val = (float)fps_frames / (float)dt;
            frame_ms = (float)(dt / fps_frames * 1000.0);
            fps_t0 = now;
            fps_frames = 0;
        }
    }

    // =====================================================================
    // TOP-LEFT: title + render pipeline + performance
    // =====================================================================
    hud_box(r, pad, pad, 280 * sc, 14 * sc, 0, 0, 0);
    hud_text(r, pad + 4 * sc, pad + 2 * sc, "DeliRium recon window", sc, 120, 200, 255);
    hud_text(r, pad + 4 * sc, pad + 4 * sc, "Render: TriInfill GPU", sc, 80, 160, 220);
    hud_text(r, pad + 4 * sc, pad + 6 * sc, "  sample PSF -> triangles", sc, 140, 150, 170);
    hud_text(r, pad + 4 * sc, pad + 8 * sc, "  GPU raster (Gouraud)", sc, 140, 150, 170);
    hud_text(r, pad + 4 * sc, pad + 10 * sc, "  anchors = physics state", sc, 140, 150, 170);
    char perf_buf[72];
    std::snprintf(perf_buf, sizeof(perf_buf), "FPS: %.1f   frame: %.1f ms", fps_val, frame_ms);
    hud_text(r, pad + 4 * sc, pad + 12 * sc, perf_buf, sc, 100, 255, 120);

    // =====================================================================
    // TOP-RIGHT: camera + scene summary
    // =====================================================================
    float dist = std::sqrt((cam.position.x - cam.target.x) * (cam.position.x - cam.target.x) +
                           (cam.position.y - cam.target.y) * (cam.position.y - cam.target.y) +
                           (cam.position.z - cam.target.z) * (cam.position.z - cam.target.z));
    char cam_buf[160];
    std::snprintf(cam_buf, sizeof(cam_buf), "cam (%.2f, %.2f, %.2f)",
                  cam.position.x, cam.position.y, cam.position.z);
    int cw = (int)std::strlen(cam_buf) * (FONT_COLS + 1) * sc;
    hud_box(r, S->w - cw - pad - 4 * sc, pad, cw + 4 * sc, 10 * sc, 0, 0, 0);
    hud_text(r, S->w - cw - pad, pad + 2 * sc, cam_buf, sc, 200, 200, 220);
    std::snprintf(cam_buf, sizeof(cam_buf), "target (%.2f, %.2f, %.2f)",
                  cam.target.x, cam.target.y, cam.target.z);
    cw = (int)std::strlen(cam_buf) * (FONT_COLS + 1) * sc;
    hud_text(r, S->w - cw - pad, pad + 4 * sc, cam_buf, sc, 180, 180, 200);
    std::snprintf(cam_buf, sizeof(cam_buf), "dist %.2f   fov %.0f", dist, cam.fov_deg);
    cw = (int)std::strlen(cam_buf) * (FONT_COLS + 1) * sc;
    hud_text(r, S->w - cw - pad, pad + 6 * sc, cam_buf, sc, 160, 160, 180);
    std::snprintf(cam_buf, sizeof(cam_buf), "objects: %zu", S->objs.size());
    cw = (int)std::strlen(cam_buf) * (FONT_COLS + 1) * sc;
    hud_text(r, S->w - cw - pad, pad + 8 * sc, cam_buf, sc, 160, 160, 180);

    // =====================================================================
    // BOTTOM-LEFT: controls legend
    // =====================================================================
    const char* controls[] = {
        "Mouse look / wheel zoom",
        "LMB poke   RMB dent",
        "R refine   C coarsen",
        "F fracture G restitch",
        "T weld     X cap on/off",
        "O orbit    P shot",
        "V record   H HUD toggle",
        "M physics detail",
        "Esc quit",
    };
    const int nctrl = 9;
    int legend_h = (nctrl + 1) * (FONT_H + 1) * sc + 4 * sc;
    hud_box(r, pad, S->h - pad - legend_h, 270 * sc, legend_h, 0, 0, 0);
    hud_text(r, pad + 4 * sc, S->h - pad - legend_h + 4 * sc, "Controls", sc, 120, 200, 255);
    for (int i = 0; i < nctrl; ++i)
        hud_text(r, pad + 4 * sc, S->h - pad - legend_h + (6 + i) * (FONT_H + 1) * sc,
                 controls[i], sc, 180, 190, 200);

    // =====================================================================
    // BOTTOM-RIGHT: aim-target stats + physics/material diagnostics
    // =====================================================================
    if (S->aim_idx >= 0 && S->aim_idx < (int)S->objs.size()) {
        PSFBody& b = S->objs[S->aim_idx].body_ref();
        const Obj& o = S->objs[S->aim_idx];
        b.Seams().Build(b);
        const float gap_pct = b.Seams().MaxGap(b) / o.diag * 100.f;
        const float vol = b.Volume();

        char stats[160];
        std::snprintf(stats, sizeof(stats), "AIM: %s", o.label);
        int sw = (int)std::strlen(stats) * (FONT_COLS + 1) * sc;
        hud_box(r, S->w - sw - pad - 4 * sc, S->h - pad - legend_h, sw + 4 * sc, legend_h, 0, 0, 0);
        hud_text(r, S->w - sw - pad, S->h - pad - legend_h + 2 * sc, stats, sc, 255, 220, 120);

        // Geometry stats
        std::snprintf(stats, sizeof(stats), "patches %zu anchors %zu",
                      b.Patches().size(), b.AnchorBlob().size());
        sw = (int)std::strlen(stats) * (FONT_COLS + 1) * sc;
        hud_text(r, S->w - sw - pad, S->h - pad - legend_h + 4 * sc, stats, sc, 200, 200, 200);

        const char* seam_col = gap_pct < 0.5f ? "100,255,120" :
                               (gap_pct < 2.0f ? "255,200,100" : "255,100,100");
        int rc, gc, bc;
        std::sscanf(seam_col, "%d,%d,%d", &rc, &gc, &bc);
        std::snprintf(stats, sizeof(stats), "seam %.3f%% diag", gap_pct);
        sw = (int)std::strlen(stats) * (FONT_COLS + 1) * sc;
        hud_text(r, S->w - sw - pad, S->h - pad - legend_h + 6 * sc, stats, sc, rc, gc, bc);

        std::snprintf(stats, sizeof(stats), "volume %.3f   diag %.2f", vol, o.diag);
        sw = (int)std::strlen(stats) * (FONT_COLS + 1) * sc;
        hud_text(r, S->w - sw - pad, S->h - pad - legend_h + 8 * sc, stats, sc, 180, 180, 200);

        // Physics/material detail panel (toggle with M)
        if (S->physics_detail && o.go) {
            const Material& m = o.go->Mat();
            const RigidState& rs = o.go->Rigid();
            std::snprintf(stats, sizeof(stats), "rho %.0f E %.0f nu %.2f",
                          m.density, m.youngs_modulus, m.poisson);
            sw = (int)std::strlen(stats) * (FONT_COLS + 1) * sc;
            hud_text(r, S->w - sw - pad, S->h - pad - legend_h + 10 * sc, stats, sc, 160, 190, 220);
            std::snprintf(stats, sizeof(stats), "mass %.2f bounce %.2f damp %.2f",
                          rs.mass, rs.restitution, m.damping);
            sw = (int)std::strlen(stats) * (FONT_COLS + 1) * sc;
            hud_text(r, S->w - sw - pad, S->h - pad - legend_h + 12 * sc, stats, sc, 160, 190, 220);
        }
    } else {
        hud_box(r, S->w - 200 * sc, S->h - pad - 4 * sc, 200 * sc, 4 * sc, 0, 0, 0);
        hud_text(r, S->w - 200 * sc + 4 * sc, S->h - pad - 4 * sc + 2 * sc,
                 "aim at an object for stats", sc, 120, 120, 120);
    }

    // =====================================================================
    // CENTER: action toast (fades)
    // =====================================================================
    if (S->toast_ttl > 0 && S->toast[0]) {
        int tw = (int)std::strlen(S->toast) * (FONT_COLS + 1) * sc;
        int tx = (S->w - tw) / 2;
        int ty = S->h / 2 - FONT_H * sc;
        hud_box(r, tx - 4 * sc, ty - 2 * sc, tw + 8 * sc, FONT_H * sc + 4 * sc,
                0, 0, 0);
        hud_text(r, tx, ty, S->toast, sc, 255, 255, 200);
        --S->toast_ttl;
    }
}

bool on_event(const void* evp, Win* S) {
    const SDL_Event* ev = (const SDL_Event*)evp;
    if (ev->type == SDL_MOUSEBUTTONDOWN) {
        Camera& cam = Engine::Instance().CurrentScene()->GetCamera();
        Vec3 p; const int i = S->aim(cam, p);
        S->aim_idx = i;   // track aim target for the HUD stats panel
        if (i < 0) return false;
        if (ev->button.button == SDL_BUTTON_LEFT) {
            if (!S->objs[i].go) return false;   // poke needs modal modes (generated objects only)
            Vec3 to_cam{cam.position.x - p.x, cam.position.y - p.y, cam.position.z - p.z};
            float l = std::sqrt(to_cam.x*to_cam.x + to_cam.y*to_cam.y + to_cam.z*to_cam.z);
            if (l > 1e-6f) { to_cam.x /= l; to_cam.y /= l; to_cam.z /= l; }
            S->objs[i].go->Poke(p, {to_cam.x*0.6f, to_cam.y*0.6f, to_cam.z*0.6f});
            S->status(i, "poke");
            char msg[64];
            std::snprintf(msg, sizeof(msg), "poke %s", S->objs[i].label);
            S->set_toast(msg);
            return true;
        }
        if (ev->button.button == SDL_BUTTON_RIGHT) {
            // PSFBody-level dent (what GameObject::DentLocally wraps):
            // refine the region, then push anchors inward radially.
            PSFBody& b = S->objs[i].body_ref();
            const float r = 0.22f * S->objs[i].diag, dep = 0.08f * S->objs[i].diag;
            b.RefineWhere(p, r, r * 0.8f);
            b.ApplyRadialDent(p, r, dep);
            S->status(i, "dent");
            char msg[64];
            std::snprintf(msg, sizeof(msg), "dent %s", S->objs[i].label);
            S->set_toast(msg);
            return true;
        }
        return false;
    }
    if (ev->type == SDL_KEYDOWN) {
        Camera& cam = Engine::Instance().CurrentScene()->GetCamera();
        Vec3 p; const int i = S->aim(cam, p);
        S->aim_idx = i;   // track aim target for the HUD stats panel
        switch (ev->key.keysym.sym) {
        case SDLK_r:
            if (i >= 0) {
                PSFBody& b = S->objs[i].body_ref();
                const size_t before = b.Patches().size();
                b.RefineWhere(p, 0.30f*S->objs[i].diag, 0.08f*S->objs[i].diag, 64);
                std::printf("  [refine %s] +%zu patches\n", S->objs[i].label,
                            b.Patches().size() - before);
                S->status(i, "refine");
                char msg[64];
                std::snprintf(msg, sizeof(msg), "refine %s (+%zu)", S->objs[i].label,
                              b.Patches().size() - before);
                S->set_toast(msg);
            }
            return true;
        case SDLK_c:
            if (i >= 0) {
                PSFBody& b = S->objs[i].body_ref();
                const size_t before = b.Patches().size();
                const int m = b.CoarsenWhere(p, 0.40f*S->objs[i].diag, 0.004f, 64);
                std::printf("  [coarsen %s] %d merges, -%zu patches\n",
                            S->objs[i].label, m, before - b.Patches().size());
                S->status(i, "coarsen");
                char msg[64];
                std::snprintf(msg, sizeof(msg), "coarsen %s (-%zu)", S->objs[i].label,
                              before - b.Patches().size());
                S->set_toast(msg);
            }
            return true;
        case SDLK_f:
            if (i >= 0) {
                if (S->objs[i].fragment) {
                    std::printf("  [fracture %s] already fractured (one fragment per body)\n",
                                S->objs[i].label);
                    S->set_toast("already fractured");
                    break;
                }
                PSFBody& b = S->objs[i].body_ref();
                FractureResult fr = FractureBody(b, p, 1.0f, cam);
                if (fr.fragment) {
                    S->frags.push_back(std::move(fr.fragment));
                    S->objs[i].fragment = S->frags.back().get();
                    if (fr.cap) {
                        S->frags.push_back(std::move(fr.cap));
                        S->objs[i].cap = S->frags.back().get();
                    }
                    std::printf("  [fracture %s] detached %zu patches%s\n",
                                S->objs[i].label,
                                S->objs[i].fragment->Patches().size(),
                                S->objs[i].cap ? " + cut cap" : "");
                    char msg[80];
                    std::snprintf(msg, sizeof(msg), "fracture %s (%zu patches)",
                                  S->objs[i].label, S->objs[i].fragment->Patches().size());
                    S->set_toast(msg);
                } else {
                    std::printf("  [fracture %s] split degenerate -- no fragment\n",
                                S->objs[i].label);
                    S->set_toast("fracture: degenerate split");
                }
                S->status(i, "fracture");
            }
            return true;
        case SDLK_g:
            if (i >= 0) {
                // PSFBody-level restitch (what GameObject::RecalculateStructure wraps)
                PSFBody& b = S->objs[i].body_ref();
                b.Seams().Build(b);
                b.Seams().FitAll(b);
                b.MarkAnchorsDirty();
                S->status(i, "restitch");
                S->set_toast("restitch");
            }
            return true;
        case SDLK_t:
            if (i >= 0) {
                PSFBody& b = S->objs[i].body_ref();
                b.Seams().Build(b);
                const float gap = b.Seams().HarmonicWeld(b, 24, 0.2f);
                std::printf("  [weld %s] HarmonicWeld gap=%.4f%% diag\n",
                            S->objs[i].label, gap / S->objs[i].diag * 100.f);
                S->status(i, "weld");
                char msg[64];
                std::snprintf(msg, sizeof(msg), "weld %s (%.2f%%)",
                              S->objs[i].label, gap / S->objs[i].diag * 100.f);
                S->set_toast(msg);
            }
            return true;
        // Render path is fixed to TriInfill GPU (only path). No switch keys.
        case SDLK_x:
            S->show_cap = !S->show_cap;
            std::printf("  [cap] %s\n", S->show_cap ? "on" : "off");
            S->set_toast(S->show_cap ? "cut cap ON" : "cut cap OFF");
            return true;
        case SDLK_v:
            if (!S->rec_active) {
                S->rec.start(S->w, S->h, "/tmp/recon_window");
                S->rec_active = true;
                S->set_toast("recording started");
            } else {
                S->rec.finalize();
                S->rec_active = false;
                std::printf("  [rec] stopped\n");
                S->set_toast("recording stopped");
            }
            return true;
        case SDLK_o:
            S->orbit = !S->orbit;
            std::printf("  [orbit] %s\n", S->orbit ? "on" : "off");
            S->set_toast(S->orbit ? "orbit ON" : "orbit OFF");
            return true;
        case SDLK_p: {
            char path[128];
            std::snprintf(path, sizeof(path), "/tmp/recon_window_%04d.ppm", S->shot++);
            if (S->renderer->SaveFramePPM(path)) {
                std::printf("  [shot] %s\n", path);
                S->set_toast("shot saved");
            } else {
                std::printf("  [shot] save failed\n");
                S->set_toast("shot save failed");
            }
            return true;
        }
        case SDLK_h:
            S->hud_visible = !S->hud_visible;
            S->set_toast(S->hud_visible ? "HUD ON" : "HUD OFF");
            return true;
        case SDLK_m:
            S->physics_detail = !S->physics_detail;
            S->set_toast(S->physics_detail ? "physics detail ON" : "physics detail OFF");
            return true;
        default:
            return false;
        }
    }
    return false;
}

// Per-frame: auto-orbit + render every object (composited) with a single
// camera. Bodies sit at their world positions (generated ones baked there,
// the torus translated there), so no per-object camera shift is needed. A
// fracture fragment renders on top of its parent.
void on_frame(Win* S, SplatRenderer& r, Camera& cam) {
    if (S->orbit) {
        const float d = 0.02f;
        const float s = std::sin(d), c = std::cos(d);
        const float rx = cam.position.x - cam.target.x;
        const float rz = cam.position.z - cam.target.z;
        cam.position.x = cam.target.x + (c*rx + s*rz);
        cam.position.z = cam.target.z + (-s*rx + c*rz);
    }
    // final=true only on the LAST body (or its fragment) so the full-screen
    // buf_fb->buf_img resolve + queue finish happen ONCE per frame, not once
    // per body (5x/frame was the old waste).
    for (int i = 0; i < (int)S->objs.size(); ++i) {
        bool frag = S->objs[i].fragment != nullptr;
        bool cap  = S->objs[i].cap != nullptr && S->show_cap;
        // Render order for a fractured object: parent, then cap (closes the
        // parent's cut), then fragment (the far side). final=true on the LAST
        // body actually rendered this frame so the full-screen resolve +
        // finish happens exactly once (buf_fb accumulates all bodies via
        // frontmost-wins atomics; only the last resolve sees the combined
        // image).
        const bool last = (i == (int)S->objs.size() - 1);
        r.RenderBody(S->objs[i].body_ref(), cam, /*clear=*/(i == 0),
                     /*final=*/last && !cap && !frag);
        if (cap)
            // The cap is a flat disk ON the cut plane, filling the hole left
            // in the parent when the far side was detached. The parent's own
            // surface sits on the NEAR side of the plane (dot(x,n)<d, closer
            // to the camera) and curves away from it, so near the cut line the
            // parent is in front of the coplanar cap and the cap is visible
            // only in the hole region where the parent has no surface. The
            // fragment (far side) is pushed just behind the cap by its
            // depth_bias, so the order front->back at the cut line is:
            // parent surface, cap, fragment. No bias on the cap itself.
            r.RenderBody(*S->objs[i].cap, cam, /*clear=*/false,
                         /*final=*/last && !frag);
        if (frag)
            // depth_bias: the fragment is the FAR side of the cut plane. Its
            // cut rim is COPLANAR with the parent/cap cut rim (same plane) ->
            // equal depth -> atom_min flips the winner frame-to-frame = the
            // popping. A tiny +NDC bias pushes the fragment's cut rim just
            // behind the cap, so the cap always wins at the cut line. 1e-4
            // NDC is far above float epsilon (~1e-7) and visually negligible;
            // it only matters where the two rims tie.
            r.RenderBody(*S->objs[i].fragment, cam, /*clear=*/false,
                         /*final=*/last, /*depth_bias=*/1e-4f);
    }
    // 10s RECORDER: after the last body's resolve (final=true -> clFinish),
    // buf_img holds the complete frame. Read it back and save as a PNG.
    // Auto-stops after 10 seconds of wall-clock capture.
    if (S->rec_active) {
        const double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - S->rec.t0).count();
        if (el >= 10.0) {
            S->rec.finalize();
            S->rec_active = false;
            std::printf("  [rec] 10s elapsed -- stopped\n");
        } else {
            std::vector<unsigned char> rgba;
            if (r.GrabFinalRGBA(rgba)) S->rec.frame(rgba);
        }
    }
}

} // namespace

#endif // DELIRIUM_HAS_SDL

int main() {
    Engine& engine = Engine::Instance();
    if (!engine.Initialize(1024, 768, "DeliRium recon window")) return -1;
    SplatRenderer* rnd = engine.GetRenderer();
    if (!rnd) {
        std::fprintf(stderr, "no renderer (OpenCL?) -- nothing to show\n");
        return -1;
    }
    Scene* scene = engine.CreateScene("recon_window");
    // frame the 2x2 grid of objects (spans ~ -1.2..1.2 in X/Y): up and back
    scene->GetCamera().position = {0.f, 3.2f, 4.6f};
    scene->GetCamera().target   = {0.f, 0.f, 0.f};
    scene->GetCamera().up       = {0.f, 1.f, 0.f};

    // Reconstruct four objects IN MEMORY from the element-system baker, each
    // baked at a DISTINCT world position (2x2 grid) so they don't overlap.
    // CLOSED objects (orb) use BuildVolumetricBodies -> ONE TRUE volumetric
    // body with proper UV-sphere topology (patches wrap around, normals point
    // outward). Heightfield objects (bump/wave/mound) are single-sided by
    // nature — use BuildMulti -> one body with a prominent feature.
    NeuralReconstructor maker;
    struct Spec { const char* name; const char* semantic; Vec3 pos; uint32_t patches; };
    const Spec specs[] = {
        {"orb",   "orb",   {-0.7f, -0.7f, 0.f}, 900},
        {"bump",  "bump",   { 0.7f, -0.7f, 0.f}, 900},
        {"wave",  "wave",   {-0.7f,  0.7f, 0.f}, 900},
        {"mound", "mound",  { 0.7f,  0.7f, 0.f}, 900},
    };
    const char* simple_scene = std::getenv("DELIRIUM_SIMPLE_SCENE");
    const bool simple_orb = simple_scene && std::strcmp(simple_scene, "orb") == 0;
    const bool simple_plane = simple_scene && std::strcmp(simple_scene, "plane") == 0;
    const bool simple = simple_orb || simple_plane;

    // Bake a single element tree into a body via ElementSystem::BakeBody
    // (same LSQ the NeuralReconstructor uses). Returns nullptr on failure.
    auto BakeTree = [](const Element& tree, uint32_t max_patches, Vec3 world_offset)
        -> std::unique_ptr<PSFBody> {
        int grid_n = std::max(4, (int)(std::sqrt((double)max_patches) + 1));
        if (grid_n > 64) grid_n = 64;
        std::string why;
        return ElementSystem::BakeBody(tree, grid_n, 1.0f, why, world_offset);
    };

    std::vector<std::unique_ptr<GameObject>> gos;
    std::vector<Vec3> go_pos;   // world position baked into each GameObject (matches gos)

    // Add a reconstructed body to gos/go_pos with a label.
    auto AddBody = [&](const std::string& label, std::unique_ptr<PSFBody> body, Vec3 pos) {
        if (!body) {
            std::fprintf(stderr, "ERROR: reconstruction of '%s' failed\n", label.c_str());
            return false;
        }
        gos.push_back(std::make_unique<GameObject>(label, std::move(body)));
        go_pos.push_back(pos);
        return true;
    };

    if (!simple) for (const Spec& sp : specs) {
        // Build element trees via BuildMulti, then bake each into a body.
        // Closed objects (orb) return upper+lower hemisphere trees; heightfields
        // return a single prominent tree. All baked via ElementSystem::BakeBody.
        auto trees = ProceduralElementGenerator::BuildMulti(sp.semantic);
        if (trees.empty()) {
            std::fprintf(stderr, "ERROR: BuildMulti('%s') returned no trees\n", sp.semantic);
            return -1;
        }
        int idx = 0;
        for (auto& tree : trees) {
            std::string label = std::string(sp.name);
            if (trees.size() > 1) label += (idx == 0 ? " (upper)" : " (lower)");
            auto body = BakeTree(*tree, sp.patches, sp.pos);
            if (!AddBody(label, std::move(body), sp.pos)) return -1;
            ++idx;
        }
    }

    // For repeatable one-object measurements, DELIRIUM_SIMPLE_SCENE=orb
    // keeps only the generated ellipsoid; =plane keeps only a single planar
    // substance. Both use the exact same PSF reconstruction/render pipeline.
    // The default still exercises all five objects. Closed objects (orb) use
    // BuildVolumetricBodies -> ONE TRUE volumetric sphere.
    if (simple) {
        std::string semantic = simple_plane ? "plane" : "orb";
        bool ok = false;
        // Both orb and plane use BuildMulti + BakeTree uniformly now that
        // BuildVolumetricBodies was removed from ProceduralElementGenerator.
        auto trees = ProceduralElementGenerator::BuildMulti(semantic);
        if (trees.empty()) {
            std::fprintf(stderr, "ERROR: simple '%s' BuildMulti returned no trees\n", semantic.c_str());
            return -1;
        }
        auto body = BakeTree(*trees[0], 900, {0.f, 0.f, 0.f});
        if (!body) {
            std::fprintf(stderr, "ERROR: simple '%s' bake failed\n", semantic.c_str());
            return -1;
        }
        ok = AddBody(semantic, std::move(body), {0.f, 0.f, 0.f});
        if (!ok) return -1;
        scene->GetCamera().position = {0.f, 0.6f, 2.8f};
        scene->GetCamera().target = {0.f, 0.f, 0.f};
        std::printf("  [scene] simple %s (1 reconstructed body, 900-patch budget)\n", semantic.c_str());
    }

    // Fifth object: the classic torus, loaded from a file (NOT reconstructed).
    // The file is centered at the origin, which would interpenetrate the 2x2
    // grid, so translate it behind the grid. This exercises the file pipeline
    // (PSFBody::FromFile) side by side with the element-system pipeline.
    std::unique_ptr<PSFBody> torus_body;
    if (!simple) {
        torus_body = PSFBody::FromFile("assets/torus.psf");
        if (torus_body) {
            torus_body->Translate({0.f, 0.f, -2.2f});
        } else {
            std::fprintf(stderr, "note: could not load assets/torus.psf -- showing 4 objects\n");
        }
    }

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    Win S;
    S.renderer = rnd;
    S.w = 1024; S.h = 768;
    // TriInfill GPU is the ONLY render path — set up once at launch.
    // Other paths (march/raymarch/splat) are legacy and produce artifacts
    // on generated objects; the future AI reconstructor targets TriInfill GPU.
    S.set_render_path(4);
    for (size_t i = 0; i < gos.size(); ++i) {
        Obj o;
        o.go = gos[i].get();
        o.body = nullptr;
        o.center = go_pos[i];
        o.label = gos[i]->name.c_str();
        const Vec3 sp = o.go->Body().BBoxSpan();
        o.diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
        S.objs.push_back(o);
    }
    if (torus_body) {
        Obj o;
        o.go = nullptr;
        o.body = torus_body.get();
        o.center = {0.f, 0.f, -2.2f};
        o.label = "torus";
        const Vec3 sp = torus_body->BBoxSpan();
        o.diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
        S.objs.push_back(o);
    }
    engine.SetInputHook([&S](const void* ev) { return on_event(ev, &S); });
    engine.SetRenderHook([&S](SplatRenderer& r, const Camera& cam) {
        on_frame(&S, r, const_cast<Camera&>(cam));
    });
    // Draw the HUD overlay after the PSF frame, before Present.
    engine.SetOverlayHook([&S](void* sdl_rnd, int w, int h) {
        draw_hud((SDL_Renderer*)sdl_rnd, &S, Engine::Instance().CurrentScene()->GetCamera());
    });
    // Capture is deliberately opt-in: each recorded frame synchronously reads
    // 3 MiB back from the GPU and compresses a PNG on the render thread.
    // That work can halve FPS and create spikes; it must not pollute the
    // default interactive-performance measurement. V still starts/stops it.
    if (const char* record = std::getenv("DELIRIUM_AUTO_RECORD")) {
        if (record[0] == '1' && record[1] == '\0') {
            S.rec.start(S.w, S.h, "/tmp/recon_window");
            S.rec_active = true;
        }
    }
    std::printf("Recon window ready. %s\n",
                simple ? (simple_plane ? "One reconstructed plane" : "One reconstructed orb") :
                         "Five objects: four element-system + file-loaded torus");
    std::printf("Recording is OFF by default (V to record, or DELIRIUM_AUTO_RECORD=1 for a 10s launch capture).\n"
                "Render path: TriInfill GPU (only path).\n"
                "Aim = window center (mouse-look). Aim at one, then:\n"
                "  LMB poke (generated only)   RMB dent   R refine   C coarsen   F fracture\n"
                "  G restitch  T weld     O orbit   X cap on/off   P shot   Esc quit\n");
#endif

    engine.Run();
    engine.Shutdown();
    return 0;
}