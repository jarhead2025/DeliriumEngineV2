// FILEPATH: engine/src/ai/ElementSystem.cpp
// Generic element-tree -> PSF records baker. Category-agnostic: it composes
// the overlapping-system field and fits patches with the SAME corner-vanishing
// least-squares the engine uses internally (PSFBody::SplitPatch), so any
// element tree the future categorizer emits becomes a live, valid PSFBody.
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace DeliRium {

namespace {

Vec3 vnorm(const Vec3& a) {
    float l = std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    return l > 1e-12f ? Vec3{a.x/l, a.y/l, a.z/l} : Vec3{0.f, 0.f, 1.f};
}

// Corner-vanishing basis -- IDENTICAL to PSFBody::cv_basis and the render
// deform_h, so a coefficient set fit here reproduces exactly in the renderer.
void cv_basis(float u, float v, double B[8]) {
    const double bu = u*(1.0-u), bv = v*(1.0-v);
    const double au = 2.0*u-1.0, av = 2.0*v-1.0;
    B[0]=bu*(1-v); B[1]=bu*v; B[2]=bv*(1-u); B[3]=bv*u;
    B[4]=bu*au*(1-v); B[5]=bu*au*v; B[6]=bv*av*(1-u); B[7]=bv*av*u;
}

// 8x8 Gaussian elimination with partial pivoting -- identical to
// PSFBody::solve8, so fits here match the engine's own refits.
bool solve8(double A[8][8], double b[8], float out[8]) {
    for (int c = 0; c < 8; ++c) {
        int best = c;
        for (int r = c + 1; r < 8; ++r)
            if (std::fabs(A[r][c]) > std::fabs(A[best][c])) best = r;
        if (std::fabs(A[best][c]) < 1e-14) return false;
        if (best != c) { for (int k = 0; k < 8; ++k) std::swap(A[c][k], A[best][k]);
                         std::swap(b[c], b[best]); }
        for (int r = c + 1; r < 8; ++r) {
            const double f = A[r][c] / A[c][c];
            for (int k = c; k < 8; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    for (int c = 7; c >= 0; --c) {
        double s = b[c];
        for (int k = c + 1; k < 8; ++k) s -= A[c][k] * out[k];
        out[c] = (float)(s / A[c][c]);
    }
    return true;
}

uint16_t q16(float t) {
    if (t < 0.f) t = 0.f; else if (t > 1.f) t = 1.f;
    return (uint16_t)(t * 65535.0f + 0.5f);
}
uint8_t q8(float t) {
    if (t < 0.f) t = 0.f; else if (t > 1.f) t = 1.f;
    return (uint8_t)(t * 255.0f + 0.5f);
}

} // namespace

// =============================================================================
// Substances (the leaf "physical substance")
// =============================================================================
SubstancePoint PlaneSubstance::Eval(float x, float y) const {
    return { Vec3{x, y, 0.f}, Vec3{0.f, 0.f, 1.f} };
}

SubstancePoint EllipsoidSubstance::Eval(float x, float y) const {
    // upper hemisphere: z = sz*sqrt(1 - x^2/sx^2 - y^2/sy^2)
    const float q = (x*x) / (sx_*sx_) + (y*y) / (sy_*sy_);
    if (q >= 1.0f) return { Vec3{x, y, 0.f}, Vec3{0.f, 0.f, 1.f}, false };
    const float z = sz_ * std::sqrt(1.0f - q);
    // implicit F = x^2/sx^2 + y^2/sy^2 + z^2/sz^2 = 1 -> grad = (2x/sx^2, 2y/sy^2, 2z/sz^2)
    Vec3 n{ x / (sx_*sx_), y / (sy_*sy_), z / (sz_*sz_) };
    return { Vec3{x, y, z}, vnorm(n) };
}

SubstancePoint GaussianSystemSubstance::Eval(float x, float y) const {
    const float d2 = x*x + y*y;
    const float h = amp_ * std::exp(-d2 / (2.f * sigma_ * sigma_));
    // dh/dx = -x/sigma^2 * h ; normal ~ normalize(-dh/dx, -dh/dy, 1)
    const float g = h / (sigma_ * sigma_);
    Vec3 n{ -x * g, -y * g, 1.f };
    return { Vec3{x, y, h}, vnorm(n) };
}

SubstancePoint HeightfieldSubstance::Eval(float x, float y) const {
    const float e = 1e-3f;
    const float h  = fn_(x, y);
    const float hx = fn_(x + e, y) - fn_(x - e, y);
    const float hy = fn_(x, y + e) - fn_(x, y - e);
    Vec3 n{ -hx / (2.f * e), -hy / (2.f * e), 1.f };
    return { Vec3{x, y, h}, vnorm(n) };
}

// =============================================================================
// Field composition: the weighted blend of every active system at a location.
// =============================================================================
ElementSystem::Field ElementSystem::FieldAt(const Element& root, const Vec3& world) {
    Field acc{Vec3{0,0,0}, Vec3{0,0,0}, Vec3{0,0,0}, 0.f};
    struct Frame { const Element* el; Vec3 origin; Vec3 scale; };
    std::vector<Frame> stack;
    stack.push_back({&root, Vec3{0,0,0}, Vec3{1,1,1}});
    while (!stack.empty()) {
        Frame f = stack.back(); stack.pop_back();
        const Element* el = f.el;
        // world -> this element's local frame
        const Vec3 local{ (world.x - f.origin.x) / f.scale.x,
                          (world.y - f.origin.y) / f.scale.y,
                          (world.z - f.origin.z) / f.scale.z };
        const float pres = el->presence ? std::max(0.f, std::min(1.f, el->presence(world))) : 1.f;
        const float w = el->weight * pres;
        if (el->substance && w > 1e-6f) {
            SubstancePoint sp = el->substance->Eval(local.x, local.y);
            if (sp.valid) {
                // local -> world (point)
                const Vec3 wp{ f.origin.x + sp.p.x * f.scale.x,
                               f.origin.y + sp.p.y * f.scale.y,
                               f.origin.z + sp.p.z * f.scale.z };
                // normal: inverse-transpose of the (uniform-per-axis) scale
                const Vec3 wn{ sp.n.x / f.scale.x, sp.n.y / f.scale.y, sp.n.z / f.scale.z };
                acc.point.x += w * wp.x; acc.point.y += w * wp.y; acc.point.z += w * wp.z;
                acc.normal.x += w * wn.x; acc.normal.y += w * wn.y; acc.normal.z += w * wn.z;
                acc.color.x += w * el->color.x; acc.color.y += w * el->color.y; acc.color.z += w * el->color.z;
                acc.weight += w;
            }
        }
        for (auto& c : el->children) {
            // child local frame in WORLD = parent_origin + child_origin * parent_scale
            Frame cf;
            cf.el = c.get();
            cf.origin = { f.origin.x + c->origin.x * f.scale.x,
                          f.origin.y + c->origin.y * f.scale.y,
                          f.origin.z + c->origin.z * f.scale.z };
            cf.scale  = { c->scale.x * f.scale.x, c->scale.y * f.scale.y, c->scale.z * f.scale.z };
            stack.push_back(cf);
        }
    }
    if (acc.weight > 1e-9f) {
        const float inv = 1.f / acc.weight;
        acc.point.x *= inv; acc.point.y *= inv; acc.point.z *= inv;
        acc.color.x *= inv; acc.color.y *= inv; acc.color.z *= inv;
        acc.normal = vnorm(acc.normal);
    } else {
        acc.point = {0,0,0}; acc.normal = {0,0,1}; acc.color = {0,0,0};
    }
    return acc;
}

// =============================================================================
// Bake: sample the aggregate field onto a patch grid, fit corner-vanishing
// coefficients per patch, emit frozen 64-byte records.
// =============================================================================
bool ElementSystem::Bake(const Element& root, int grid_n, float extent,
                         ModelOutput& out, std::string& why,
                         Vec3 world_offset) {
    out = ModelOutput{};
    if (grid_n < 2) { why = "grid_n must be >= 2"; return false; }
    if (grid_n > 64) grid_n = 64;
    if (extent <= 1e-6f) { why = "extent must be > 0"; return false; }
    const int N = grid_n;
    const int npatch = N - 1;

    // --- sample the field on the N x N anchor grid -------------------------
    std::vector<Vec3> pts((size_t)N * N);
    std::vector<Vec3> nrm((size_t)N * N);
    std::vector<Vec3> col((size_t)N * N);
    std::vector<char> ok((size_t)N * N, 0);
    Vec3 mn{1e30f,1e30f,1e30f}, mx{-1e30f,-1e30f,-1e30f};
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float u = i / float(N - 1), v = j / float(N - 1);
            const Vec3 world{(u - 0.5f) * extent, (v - 0.5f) * extent, 0.f};
            Field f = FieldAt(root, world);
            const size_t id = (size_t)j * N + i;
            // weight==0: no system covers this location (e.g. outside a
            // bounded substance's support). FieldAt collapses such points to
            // the origin -- they must NOT be baked as surface or the corner
            // patches fit a phantom point (the "flat card + bump" artifact).
            ok[id] = (f.weight > 1e-9f) ? 1 : 0;
            if (!ok[id]) continue;
            pts[id] = f.point; nrm[id] = f.normal; col[id] = f.color;
            // bbox is offset-inclusive: the stored positions are pts+offset,
            // so the quantization box must span them (and BakeBody recomputes
            // the identical box for FromRecords' dequant).
            mn.x = std::min(mn.x, f.point.x + world_offset.x); mn.y = std::min(mn.y, f.point.y + world_offset.y); mn.z = std::min(mn.z, f.point.z + world_offset.z);
            mx.x = std::max(mx.x, f.point.x + world_offset.x); mx.y = std::max(mx.y, f.point.y + world_offset.y); mx.z = std::max(mx.z, f.point.z + world_offset.z);
        }
    if (mn.x > 1e29f) { why = "no in-support samples on the grid"; return false; }
    // margin so later deformation does not push anchors outside the box
    const Vec3 span0{ mx.x-mn.x, mx.y-mn.y, mx.z-mn.z };
    const float diag0 = std::sqrt(span0.x*span0.x + span0.y*span0.y + span0.z*span0.z) + 1e-9f;
    const float m = 0.05f * diag0;
    const Vec3 bbmin{ mn.x - m, mn.y - m, mn.z - m };
    const Vec3 bbspan{ span0.x + 2*m, span0.y + 2*m, span0.z + 2*m };
    auto bbsafe = [](float s) { return s < 1e-6f ? 1e-6f : s; };
    const float bsx = bbsafe(bbspan.x), bsy = bbsafe(bbspan.y), bsz = bbsafe(bbspan.z);

    // --- anchor records (quantized positions + normals + color) ------------
    out.anchors.reserve((size_t)N * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const size_t id = (size_t)j * N + i;
            AnchorRecord a{};
            a.bone_index = 0xFFFFFFFFu;
            a.connectivity[0] = a.connectivity[1] = a.connectivity[2] = a.connectivity[3] = 0xFFFF;
            if (!ok[id]) {
                // out-of-support: safe default at the bbox center, +Z normal.
                // No surface references it (its patches are skipped below),
                // so this is purely a well-formed unused record.
                a.pos[0] = a.pos[1] = a.pos[2] = 32767;
                a.normal[0] = 128; a.normal[1] = 128; a.normal[2] = 255;
                out.anchors.push_back(a);
                continue;
            }
            // world position = local field point + world_offset (placement)
            a.pos[0] = q16((pts[id].x + world_offset.x - bbmin.x) / bsx);
            a.pos[1] = q16((pts[id].y + world_offset.y - bbmin.y) / bsy);
            a.pos[2] = q16((pts[id].z + world_offset.z - bbmin.z) / bsz);
            a.normal[0] = q8(nrm[id].x * 0.5f + 0.5f);
            a.normal[1] = q8(nrm[id].y * 0.5f + 0.5f);
            a.normal[2] = q8(nrm[id].z * 0.5f + 0.5f);
            a.color[0] = q8(col[id].x); a.color[1] = q8(col[id].y); a.color[2] = q8(col[id].z);
            a.roughness = 128; a.metallic = 0; a.anim_weight = 0;
            out.anchors.push_back(a);
        }

    // --- surface records (one patch per grid cell) --------------------------
    out.surfaces.reserve((size_t)npatch * npatch);
    const int G = 8;  // LSQ sample grid per patch (same as SplitPatch)
    for (int j = 0; j < npatch; ++j)
        for (int i = 0; i < npatch; ++i) {
            // corner ordering matches the renderer bilinear: A0(0,0) A1(1,0) A2(1,1) A3(0,1)
            const uint16_t a00 = (uint16_t)((size_t)j * N + i);
            const uint16_t a10 = (uint16_t)((size_t)j * N + i + 1);
            const uint16_t a11 = (uint16_t)((size_t)(j + 1) * N + i + 1);
            const uint16_t a01 = (uint16_t)((size_t)(j + 1) * N + i);
            // skip patches with any out-of-support corner: fitting them would
            // target the phantom origin (FieldAt's zero-weight collapse) and
            // produce degenerate/huge-coef patches (the "flat card + bump"
            // artifact on bounded substances like the ellipsoid).
            if (!ok[a00] || !ok[a10] || !ok[a11] || !ok[a01]) continue;
            // dequant the corner positions (what the renderer will use)
            auto deq = [&](uint16_t a) -> Vec3 {
                const AnchorRecord& r = out.anchors[a];
                return Vec3{ bbmin.x + r.pos[0] / 65535.f * bsx,
                             bbmin.y + r.pos[1] / 65535.f * bsy,
                             bbmin.z + r.pos[2] / 65535.f * bsz };
            };
            const Vec3 CA[4] = { deq(a00), deq(a10), deq(a11), deq(a01) };
            const Vec3 CN[4] = { nrm[a00], nrm[a10], nrm[a11], nrm[a01] };

            double M[8][8] = {{0}}, r[8] = {0};
            for (int s = 0; s < G; ++s)
                for (int t = 0; t < G; ++t) {
                    const float cu = s / float(G - 1), cv = t / float(G - 1);
                    // bilinear of corner positions + normal field (renderer convention)
                    const float um = 1.f - cu, vm = 1.f - cv;
                    Vec3 bil{ um*vm*CA[0].x + cu*vm*CA[1].x + cu*cv*CA[2].x + um*cv*CA[3].x,
                              um*vm*CA[0].y + cu*vm*CA[1].y + cu*cv*CA[2].y + um*cv*CA[3].y,
                              um*vm*CA[0].z + cu*vm*CA[1].z + cu*cv*CA[2].z + um*cv*CA[3].z };
                    Vec3 nf{ um*vm*CN[0].x + cu*vm*CN[1].x + cu*cv*CN[2].x + um*cv*CN[3].x,
                             um*vm*CN[0].y + cu*vm*CN[1].y + cu*cv*CN[2].y + um*cv*CN[3].y,
                             um*vm*CN[0].z + cu*vm*CN[1].z + cu*cv*CN[2].z + um*cv*CN[3].z };
                    nf = vnorm(nf);
                    // target = the aggregate field at this world location
                    const float wu = (i + cu) / float(N - 1), wv = (j + cv) / float(N - 1);
                    const Vec3 world{(wu - 0.5f) * extent, (wv - 0.5f) * extent, 0.f};
                    const Field f = FieldAt(root, world);
                    // target in the SAME frame as bil (dequant = world,
                    // i.e. field point + placement offset)
                    const double h = (f.point.x + world_offset.x - bil.x) * nf.x
                                   + (f.point.y + world_offset.y - bil.y) * nf.y
                                   + (f.point.z + world_offset.z - bil.z) * nf.z;
                    double B[8]; cv_basis(cu, cv, B);
                    for (int x = 0; x < 8; ++x) {
                        r[x] += B[x] * h;
                        for (int y = 0; y < 8; ++y) M[x][y] += B[x] * B[y];
                    }
                }
            SurfaceRecord s{};
            s.anchor_idx[0] = a00; s.anchor_idx[1] = a10;
            s.anchor_idx[2] = a11; s.anchor_idx[3] = a01;
            s.surface_def = 0;
            if (!solve8(M, r, s.coef))
                for (int x = 0; x < 8; ++x) s.coef[x] = 0.f;
            // constant color: pattern 0 (eval==1) in slot 0 ONLY, K[0]=val*63,
            // K[1..3]=0. The kernel sums K[0..3]/63 * pattern_eval, so this
            // decodes to exactly val (the old "all 4 slots = val*126" summed to
            // 4*val*2 = 8x -> clamped white for every mid-range color).
            Vec3 c{ (col[a00].x+col[a10].x+col[a11].x+col[a01].x)*0.25f,
                    (col[a00].y+col[a10].y+col[a11].y+col[a01].y)*0.25f,
                    (col[a00].z+col[a10].z+col[a11].z+col[a01].z)*0.25f };
            for (int ch = 0; ch < 3; ++ch) {
                const float val = (ch == 0 ? c.x : ch == 1 ? c.y : c.z);
                uint8_t* cp = s.color_pattern + ch * 6;
                cp[0] = 0x00; cp[1] = 0x00;   // idx nibbles: pattern 0 x4
                int8_t* K = reinterpret_cast<int8_t*>(cp + 2);
                int kv = (int)(std::max(0.f, std::min(1.f, val)) * 63.f + 0.5f);
                if (kv > 127) kv = 127;
                K[0] = (int8_t)kv;
                K[1] = K[2] = K[3] = 0;
            }
            out.surfaces.push_back(s);
        }

    // sanity: distinct anchors per patch (Validate also checks this)
    for (const SurfaceRecord& s : out.surfaces) {
        if (s.anchor_idx[0] == s.anchor_idx[1] || s.anchor_idx[0] == s.anchor_idx[2] ||
            s.anchor_idx[0] == s.anchor_idx[3] || s.anchor_idx[1] == s.anchor_idx[2] ||
            s.anchor_idx[1] == s.anchor_idx[3] || s.anchor_idx[2] == s.anchor_idx[3]) {
            why = "degenerate patch (duplicate anchors)";
            return false;
        }
    }
    return true;
}

std::unique_ptr<PSFBody> ElementSystem::BakeBody(const Element& root,
                                                 int grid_n, float extent,
                                                 std::string& why,
                                                 Vec3 world_offset) {
    ModelOutput mo;
    if (!Bake(root, grid_n, extent, mo, why, world_offset)) return nullptr;
    std::string vwhy;
    if (!ReconstructionModel::Validate(mo, vwhy)) { why = vwhy; return nullptr; }
    // recompute the same bbox the baker used (margin over the field extent,
    // INCLUDING the placement offset so dequant matches the stored positions)
    Vec3 mn{1e30f,1e30f,1e30f}, mx{-1e30f,-1e30f,-1e30f};
    const int N = std::max(2, std::min(64, grid_n));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float u = i / float(N - 1), v = j / float(N - 1);
            Field f = FieldAt(root, {(u - 0.5f) * extent, (v - 0.5f) * extent, 0.f});
            // MUST match Bake's bbox: out-of-support samples (weight==0) are
            // collapsed to the origin by FieldAt and are excluded there, so
            // they must be excluded here too or dequant uses a different box.
            if (f.weight <= 1e-9f) continue;
            mn.x = std::min(mn.x, f.point.x + world_offset.x); mn.y = std::min(mn.y, f.point.y + world_offset.y); mn.z = std::min(mn.z, f.point.z + world_offset.z);
            mx.x = std::max(mx.x, f.point.x + world_offset.x); mx.y = std::max(mx.y, f.point.y + world_offset.y); mx.z = std::max(mx.z, f.point.z + world_offset.z);
        }
    const Vec3 span0{ mx.x-mn.x, mx.y-mn.y, mx.z-mn.z };
    const float diag0 = std::sqrt(span0.x*span0.x + span0.y*span0.y + span0.z*span0.z) + 1e-9f;
    const float m = 0.05f * diag0;
    return PSFBody::FromRecords(std::move(mo.anchors), std::move(mo.surfaces),
                                std::vector<int8_t>{},
                                Vec3{mn.x - m, mn.y - m, mn.z - m},
                                Vec3{span0.x + 2*m, span0.y + 2*m, span0.z + 2*m},
                                0.0f);
}

// =============================================================================
// ProceduralElementGenerator - generic element trees from a semantic request.
// Stand-in for the future learned categorizer: maps a few semantic families to
// GENERIC element trees (systems + substances). The baker is unchanged.
// =============================================================================
std::unique_ptr<Element> ProceduralElementGenerator::Build(const std::string& semantic) {
    auto root = std::make_unique<Element>();
    root->kind = "object";
    root->color = {0.8f, 0.8f, 0.8f};

    // A small, generic set of semantic families. Unknown -> single plane.
    if (semantic.find("bump") != std::string::npos ||
        semantic.find("mound") != std::string::npos ||
        semantic.find("hill") != std::string::npos) {
        root->kind = "bump_system";
        auto host = std::make_unique<Element>();
        host->kind = "host_plane";
        host->substance = std::make_unique<PlaneSubstance>();
        host->color = {0.6f, 0.65f, 0.7f};
        root->AddChild(std::move(host));
        auto node = std::make_unique<Element>();
        node->kind = "gaussian_node";
        node->substance = std::make_unique<GaussianSystemSubstance>(0.2f, 0.35f);
        node->color = {0.9f, 0.7f, 0.4f};
        root->AddChild(std::move(node));
        return root;
    }
    if (semantic.find("orb") != std::string::npos ||
        semantic.find("sphere") != std::string::npos ||
        semantic.find("ball") != std::string::npos ||
        semantic.find("body") != std::string::npos) {
        root->kind = "ellipsoid_body";
        root->substance = std::make_unique<EllipsoidSubstance>(0.5f, 0.5f, 0.5f);
        root->color = {0.5f, 0.7f, 0.9f};
        return root;
    }
    if (semantic.find("field") != std::string::npos ||
        semantic.find("wave") != std::string::npos ||
        semantic.find("terrain") != std::string::npos) {
        root->kind = "wave_field";
        auto host = std::make_unique<Element>();
        host->kind = "host_plane";
        host->substance = std::make_unique<PlaneSubstance>();
        host->color = {0.6f, 0.65f, 0.7f};
        root->AddChild(std::move(host));
        auto wave = std::make_unique<Element>();
        wave->kind = "wave_system";
        wave->substance = std::make_unique<HeightfieldSubstance>(
            [](float x, float y) { return 0.1f * std::sin(6.f * x) * std::cos(6.f * y); });
        wave->color = {0.4f, 0.8f, 0.7f};
        root->AddChild(std::move(wave));
        return root;
    }
    // default: a single flat plane substance (always a valid body)
    root->kind = "plane";
    root->substance = std::make_unique<PlaneSubstance>();
    return root;
}

} // namespace DeliRium