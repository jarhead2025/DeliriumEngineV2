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

// -----------------------------------------------------------------------------
// SHEET EXTRACTION -- the multi-layer bake.
//
// WHY THIS EXISTS: a substance is a GRAPH z = f(x, y). The field composer
// blends every active sheet at a location into ONE surface point. A closed
// body (a sphere) needs TWO sheets -- an upper hemisphere and a lower one --
// and the blend AVERAGES them into a single dome, which is exactly the
// "the sphere only shows one face" symptom.
//
// The fix is not to change the substance model (a graph cannot represent a
// closed surface; that is a property of graphs). The fix is to bake each
// SHEET as its own LAYER of patches, so the upper hemisphere and the lower
// hemisphere are separate surfaces that MEET at the seam. The engine's
// renderer already composites layers by depth, so the closed body falls out
// of the existing pipeline.
//
// A "sheet" here is one (element, substance) pair -- a leaf in the tree.
// Each sheet is baked over its own grid, and the layers are concatenated
// into one anchor/surface set (the anchor index space is uint16, so the
// total must stay under 65535).
// -----------------------------------------------------------------------------
struct Sheet {
    const Element* el = nullptr;
    Vec3 origin{0,0,0};      // world position of this sheet's local origin
    Vec3 scale{1,1,1};       // world scale of this sheet's local frame
    float weight = 1.f;      // the element's own contribution weight
    int depth = 0;           // nesting depth (for the depth weight)
    // The index of this sheet's bone in the flat bone list, or -1 for a
    // sheet whose element has no bone. This is what an anchor's
    // bone_index will refer to.
    int bone = -1;
};

// Collect every leaf sheet in the tree, with its resolved world frame.
//
// The BONE INDEX is assigned here, in the SAME traversal that collects the
// sheets, so a bone's index is stable and matches the flat bone list the
// baker emits.
//
// WHY THE ORDER MATTERS: the parser resolves Bone::parent into an index
// into the flat bone list, using ITS traversal (a recursive pre-order
// walk). If the baker used a different order, a bone's index would refer
// to a different bone -- and the skeleton would be wrong in a way that
// still validates. So the bone list is built with the SAME recursion the
// parser used, and the sheet's bone index is looked up by POINTER, not by
// position. A pointer lookup cannot drift.
void CollectSheets(const Element& root, std::vector<Sheet>& out,
                   std::vector<const Bone*>* bones = nullptr) {
    struct Frame { const Element* el; Vec3 origin; Vec3 scale; int depth; };
    std::vector<Frame> stack;
    stack.push_back({&root, Vec3{0,0,0}, Vec3{1,1,1}, 0});
    while (!stack.empty()) {
        Frame f = stack.back(); stack.pop_back();
        if (f.el->substance) {
            Sheet s;
            s.el = f.el;
            s.origin = f.origin;
            s.scale = f.scale;
            s.weight = f.el->weight;
            s.depth = f.depth;
            // The bone index: the position of this element's bone in the
            // flat list, found by POINTER so the two traversals cannot
            // disagree.
            if (f.el->has_bone && bones) {
                s.bone = -1;
                for (size_t i = 0; i < bones->size(); ++i)
                    if ((*bones)[i] == &f.el->bone) { s.bone = int(i); break; }
            }
            out.push_back(s);
        }
        for (auto& c : f.el->children) {
            Frame cf;
            cf.el = c.get();
            cf.origin = { f.origin.x + c->origin.x * f.scale.x,
                          f.origin.y + c->origin.y * f.scale.y,
                          f.origin.z + c->origin.z * f.scale.z };
            cf.scale  = { c->scale.x * f.scale.x, c->scale.y * f.scale.y,
                          c->scale.z * f.scale.z };
            cf.depth  = f.depth + 1;
            stack.push_back(cf);
        }
    }
}

// The flat bone list, in the SAME recursive pre-order the parser used to
// assign Bone::parent indices. The baker and the parser MUST agree on this
// order, or a bone's parent index would point at the wrong bone.
void CollectBones(const Element& e, std::vector<const Bone*>& out) {
    if (e.has_bone) out.push_back(&e.bone);
    for (const auto& c : e.children) CollectBones(*c, out);
}

// The world-space frame of a bone: its head and tail, transformed by the
// owning element's frame. A bone is authored in the element's LOCAL frame
// (like every other coordinate), and the element's origin/scale carry it
// into the world -- exactly like a substance's surface points.
struct BoneFrame { Vec3 head, tail, bend; float length; };

BoneFrame BoneWorldFrame(const Bone& b, const Vec3& origin, const Vec3& scale) {
    BoneFrame f;
    f.head = { origin.x + b.head.x * scale.x,
               origin.y + b.head.y * scale.y,
               origin.z + b.head.z * scale.z };
    f.tail = { origin.x + b.tail.x * scale.x,
               origin.y + b.tail.y * scale.y,
               origin.z + b.tail.z * scale.z };
    f.bend = { origin.x + b.bend.x * scale.x,
               origin.y + b.bend.y * scale.y,
               origin.z + b.bend.z * scale.z };
    const Vec3 d = f.tail - f.head;
    f.length = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    return f;
}

// Evaluate ONE sheet (not the blended field) at a local (x, y), returning
// the world point + normal, or weight==0 when the sheet has no surface
// there (outside a bounded domain).
ElementSystem::Field SheetAt(const Sheet& s, float lx, float ly) {
    ElementSystem::Field f{Vec3{0,0,0}, Vec3{0,0,0}, Vec3{0,0,0}, 0.f};
    const SubstancePoint sp = s.el->substance->Eval(lx, ly);
    if (!sp.valid) return f;
    const Vec3 wp{ s.origin.x + sp.p.x * s.scale.x,
                   s.origin.y + sp.p.y * s.scale.y,
                   s.origin.z + sp.p.z * s.scale.z };
    const Vec3 wn{ sp.n.x / s.scale.x, sp.n.y / s.scale.y, sp.n.z / s.scale.z };
    f.point = wp;
    f.normal = vnorm(wn);
    f.color = s.el->color;
    f.weight = s.weight;
    return f;
}

} // namespace

// =============================================================================
// THE SHEET'S SAMPLING DOMAIN.
//
// WHY THIS IS A SHARED HELPER (and not an inline computation at each site):
//
// A sheet's surface is sampled on a GRID, and the grid must cover the
// sheet's own domain -- not the whole canvas. A substance much smaller
// than the canvas would otherwise get almost no samples, and the few it
// got would cluster wherever its support happened to cross the grid.
//
// But there are THREE places that must agree on that domain:
//
//   1. the ANCHOR sampling (where each grid cell's point is evaluated),
//   2. the PATCH-FIT TARGET (where the fitted patch is compared against
//      the sheet's surface), and
//   3. the BBOX derivation in BakeBody (which must produce the identical
//      quantization box the baker used).
//
// When site 1 moved to the per-sheet domain and site 2 did not, the
// mismatch produced a body whose patch coefficients were fitted against
// the WRONG surface -- the patch's displaced position came from the anchor
// positions but its SHAPE was fitted to a target elsewhere on the sheet.
// The visible result was a body with radial "spokes" splaying out of it
// (a dandelion), and a coverage that swung with the camera's azimuth even
// though the anchors were perfectly symmetric.
//
// One helper, three call sites: the domain cannot drift again.
//
// THE RULES:
//   * Start from the canvas.
//   * Ask the substance for its own domain (a substance with no domain
//     leaves the canvas alone).
//   * A degenerate domain (a zero-radius substance) would divide by zero;
//     fall back to the canvas.
//   * CLIP to the canvas: the canvas is the world the tree occupies, so a
//     support reaching past it is not represented anyway -- and clipping
//     keeps the grid step from getting COARSER than it was before the
//     per-substance domains existed (a gaussian with 3 sigma = 0.9 would
//     otherwise stretch the grid over 1.8 units and halve the resolution
//     inside the canvas).
// =============================================================================
static void SheetDomain(const Element& el, float extent, float& hx, float& hy) {
    hx = extent * 0.5f;
    hy = extent * 0.5f;
    if (el.substance) el.substance->Domain(hx, hy);
    if (hx < 1e-6f) hx = extent * 0.5f;
    if (hy < 1e-6f) hy = extent * 0.5f;
    if (hx > extent * 0.5f) hx = extent * 0.5f;
    if (hy > extent * 0.5f) hy = extent * 0.5f;
}

// =============================================================================
// THE EFFECTIVE GRID SIZE.
//
// The grid is GLOBAL: one N x N block of anchors per sheet. A substance whose
// shape does not fill its square domain asks for a FINER grid (see
// ISubstance::SamplingRefinement) -- the torus is the case that matters,
// because its support is an annulus and the cells that land on the TUBE are
// what resolve its inner rim.
//
// The factor is the MAX over the sheets, so a tree with a torus and a plane
// gets the torus's finer grid and the plane simply gains samples it did not
// need.
//
// WHY THIS IS A SHARED HELPER: Bake and BakeBody BOTH need the same N --
// BakeBody re-derives the quantization box by re-sampling the field on the
// grid, and if its N differed from the baker's, every stored position would
// be dequantized against a different box (which is exactly the bug that made
// a rod's mass sit at 25% of its true spread). Two copies of the rule is
// two chances to drift.
// =============================================================================
static int EffectiveGridN(int grid_n, const std::vector<Sheet>& sheets) {
    int n = std::max(2, std::min(64, grid_n));
    float refine = 1.f;
    for (const Sheet& sh : sheets)
        if (sh.el && sh.el->substance)
            refine = std::max(refine, sh.el->substance->SamplingRefinement());
    if (refine > 1.f) {
        const int want = int(std::ceil(float(n) * refine));
        n = std::max(n, std::min(64, want));
    }
    return n;
}

// =============================================================================
// Substances (the leaf "physical substance")
// =============================================================================
SubstancePoint PlaneSubstance::Eval(float x, float y) const {
    return { Vec3{x, y, 0.f}, Vec3{0.f, 0.f, 1.f} };
}

SubstancePoint EllipsoidSubstance::Eval(float x, float y) const {
    // upper hemisphere: z = sz*sqrt(1 - x^2/sx^2 - y^2/sy^2)
    const float q = (x*x) / (sx_*sx_) + (y*y) / (sy_*sy_);
    if (q >= 1.0f) {
        // --- THE RIM ------------------------------------------------------
        // Past the footprint there is no hemisphere -- but there IS an EDGE,
        // and an edge needs a surface that continues past it so two sheets
        // can OVERLAP there rather than meet at a knife edge.
        //
        // The rim's shape: the footprint is the ellipse q = 1. Outside it,
        // the rim is a flat annulus at z = 0 whose normal is the OUTWARD
        // HORIZONTAL direction (the footprint's own gradient). That is the
        // surface's tangent direction at the edge -- so the rim continues
        // the surface's direction instead of stopping dead.
        //
        // WHY THE RIM'S NORMAL IS THE FOOTPRINT'S GRADIENT:
        //
        // The upper hemisphere's normal at the equator is horizontal (+radial
        // outward). The lower, flipped hemisphere's normal there is -radial
        // -- so at the seam the two sheets' normals still OPPOSE, and a
        // knife-edge meeting is what produced the serrated slit.
        //
        // The rim's normal is the footprint's GRADIENT -- the direction the
        // surface is moving as it leaves the footprint -- and the flip is in
        // Z, while the gradient is in the XY plane. So the flip does NOT
        // mirror it: BOTH sheets' rims carry the SAME normal. In the
        // overlap annulus the two surfaces therefore agree about which way
        // is "out", and the overlap is shaded as one continuous surface.
        //
        // The rim's width is a fraction of the footprint, so it stays
        // proportional to the grid step at any resolution.
        const float rim = RimMargin();
        if (rim > 0.f) {
            // The footprint's gradient direction (outward, in the XY plane).
            Vec3 g{ x / (sx_*sx_), y / (sy_*sy_), 0.f };
            const float gl = std::sqrt(g.x*g.x + g.y*g.y);
            if (gl > 1e-9f) { g.x /= gl; g.y /= gl; }
            // How far past the footprint are we, in the gradient direction?
            // The rim extends `rim * half_extent` beyond the boundary.
            const float edge_x = sx_ * g.x, edge_y = sy_ * g.y;
            const float over = (x - edge_x) * g.x + (y - edge_y) * g.y;
            const float reach = rim * std::sqrt((sx_*g.x)*(sx_*g.x) + (sy_*g.y)*(sy_*g.y));
            if (over <= reach) {
                // On the rim: a flat annulus at z = 0, normal = the
                // footprint's gradient. Its RADIUS is clamped to the
                // footprint's edge so the rim is a true annulus.
                const float r_edge = std::sqrt(edge_x*edge_x + edge_y*edge_y);
                const float scale = (r_edge > 1e-9f) ? (r_edge + over) / r_edge : 1.f;
                Vec3 p{ edge_x * scale, edge_y * scale, 0.f };
                return { p, vnorm(g) };
            }
        }
        return { Vec3{x, y, 0.f}, Vec3{0.f, 0.f, 1.f}, false };
    }
    const float z = sz_ * std::sqrt(1.0f - q);
    // implicit F = x^2/sx^2 + y^2/sy^2 + z^2/sz^2 = 1 -> grad = (2x/sx^2, 2y/sy^2, 2z/sz^2)
    Vec3 n{ x / (sx_*sx_), y / (sy_*sy_), z / (sz_*sz_) };
    return { Vec3{x, y, z}, vnorm(n) };
}

SubstancePoint TorusSubstance::Eval(float x, float y) const {
    const float r = std::sqrt(x * x + y * y);
    const float d = r - major_;              // distance from the tube axis
    const float inside = minor_ * minor_ - d * d;
    // Outside the tube's cross-section there is no surface. Inside the hole
    // (d < -minor) the same test fails, which is what leaves the hole open.
    if (inside <= 0.f) return { Vec3{x, y, 0.f}, Vec3{0.f, 0.f, 1.f}, false };
    const float z = std::sqrt(inside);
    // Normal from the implicit surface F = d^2 + z^2 - minor^2:
    //   dF/dx = 2*d*x/r,  dF/dy = 2*d*y/r,  dF/dz = 2*z
    // (the outward normal is the gradient; at r -> 0 the radial part is
    //  undefined, but there the surface does not exist, so it cannot hit).
    //
    // THE NORMAL MUST BE THE TRUE OUTWARD ONE OF THE TUBE.
    //
    // The gradient of F in the (r, z) half-plane is (d, z). That vector is
    // the outward direction of the tube's CROSS-SECTION, and rotating it
    // into 3D gives:
    //
    //     n = (d/r * x, d/r * y, z)
    //
    // NOT (d*x, d*y, z). The difference is the factor 1/r: the raw form
    // over-weights the radial component by r, so the normal leans toward
    // the radial direction and stops agreeing with the true surface normal.
    //
    // MEASURED (the bug this fixes): the old normal disagreed with the true
    // outward normal by up to 0.758 in dot product, and the baked anchor
    // normals came out 91.4% ANTIPARALLEL to the view (mean dot -0.7332).
    // That is what produced the "only the forward-facing faces render"
    // symptom: the march accumulates the 3x3x3 neighbourhood's normals and
    // then flips the SUM, so a wrong sum stays wrong.
    //
    // The z component is POSITIVE here because this substance models the
    // UPPER half of the tube (z = +sqrt(inside)); the lower half is the
    // MIRRORED sheet (scale z = -1), whose normal is this one reflected.
    //
    // The `r` guard: at r -> 0 the radial direction is undefined. For the
    // usual torus (major > minor) r = 0 is not on the surface at all -- the
    // test above already returned invalid. But a degenerate torus
    // (major < minor) WOULD have surface there, so the guard is needed for
    // correctness, not just for the usual case.
    if (r < 1e-9f) return { Vec3{x, y, z}, Vec3{0.f, 0.f, 1.f} };
    Vec3 n{ d * x / r, d * y / r, z };
    const float nl = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (nl > 1e-9f) { n.x /= nl; n.y /= nl; n.z /= nl; }
    return { Vec3{x, y, z}, n };
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
    struct Frame { const Element* el; Vec3 origin; Vec3 scale; int depth; };
    std::vector<Frame> stack;
    stack.push_back({&root, Vec3{0,0,0}, Vec3{1,1,1}, 0});
    while (!stack.empty()) {
        Frame f = stack.back(); stack.pop_back();
        const Element* el = f.el;
        // world -> this element's local frame
        const Vec3 local{ (world.x - f.origin.x) / f.scale.x,
                          (world.y - f.origin.y) / f.scale.y,
                          (world.z - f.origin.z) / f.scale.z };
        const float pres = el->presence ? std::max(0.f, std::min(1.f, el->presence(world))) : 1.f;

        // DEPTH-WEIGHTED CONTRIBUTION.
        //
        // WHY: the user's rule is that overlapping surfaces AVERAGE, so the
        // seams between them do not matter -- but that only holds if the
        // overlapping systems contribute COMPARABLY. With a flat weight, a
        // child drowns its parent: the model asked for "a smooth glass
        // ellipsoid" and emitted an ellipsoid with a plane child, and the
        // plane (an infinite surface at full weight) flattened the dome into
        // a disc. The symptom was "the sphere isn't there, it only shows
        // one face".
        //
        // The fix is a weight that decreases with nesting depth: a direct
        // child contributes fully, a grandchild half as much, and so on. A
        // parent therefore always keeps a majority voice over its own
        // descendants, and a DEEPER, more specific system refines a
        // shallower one instead of replacing it.
        //
        // The parent's own weight is NOT damped -- only the depth factor
        // applies -- so a root with several children still blends them.
        const float depth_scale = 1.0f / float(1 << std::min(f.depth, 6));
        const float w = el->weight * pres * depth_scale;

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
            cf.depth  = f.depth + 1;
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
// Bake: sample the tree onto a patch grid, fit corner-vanishing coefficients
// per patch, emit frozen 64-byte records.
//
// MULTI-LAYER BAKE (the closed-body fix):
//
// The original bake sampled the BLENDED field on one N x N grid at Z=0, so
// every (x,y) produced exactly ONE surface point. Two sheets at the same
// (x,y) were averaged into that one point -- which is why a "sphere" built
// from an upper and a lower hemisphere came out as a single dome with a
// flat bottom ("it only shows one face").
//
// Now each SHEET (a leaf substance in the tree) is baked as its OWN layer:
// its own anchor grid, its own patch set, its own surface. The layers are
// concatenated into one anchor/surface list, so the renderer composites them
// by depth and the upper and lower hemispheres MEET at the seam instead of
// averaging into one dome.
//
// The single-layer path is unchanged for a tree with exactly one sheet, so
// the existing behaviour (and its tests) are preserved.
// =============================================================================
// =============================================================================
// THE DERIVED-MESH BAKE (G22).
//
// The grid bake PLACES its cells on a lattice and hopes the surface lands
// on them. The derived bake DERIVES the cells from the point set.
//
// The layers are baked PER SHEET, exactly as the grid bake does, so a
// closed body (an upper sheet and a mirrored lower one) still MEETS at the
// seam instead of averaging into one dome.
//
// WHY A SEPARATE FUNCTION AND NOT A FLAG IN Bake: the grid path's patch
// loop is built around the lattice -- the patch's corners are at grid
// positions (j*N+i), and the LSQ target is sampled at the patch's (u,v) ON
// THE GRID. A DERIVED patch has no grid position: its corners are four
// arbitrary points and its (u,v) is its own bilinear. The two paths differ
// in three places (the corners, the target sampling, the connectivity), and
// keeping them apart makes the difference auditable.
// =============================================================================
bool ElementSystem::BakeDerived(const Element& root, int sample_n, float extent,
                                ModelOutput& out, std::string& why,
                                MeshDerivation& stats,
                                Vec3 world_offset) {
    out = ModelOutput{};
    stats = MeshDerivation{};
    if (sample_n < 4) { why = "sample_n must be >= 4"; return false; }
    if (sample_n > 64) sample_n = 64;
    if (extent <= 1e-6f) { why = "extent must be > 0"; return false; }

    // The sheets, with their resolved world frames -- the SAME collection
    // the grid bake uses, so the two paths see the same tree.
    std::vector<Sheet> sheets;
    CollectSheets(root, sheets);
    if (sheets.empty()) { why = "the tree has no sheets"; return false; }

    // The BONES are collected in the same pass and in the SAME order the
    // parser used, so the parent index in the emitted BoneRecords refers to
    // the right bone.
    std::vector<const Bone*> bones;
    CollectBones(root, bones);

    // Each sheet's derived mesh is baked into the SAME anchor/surface lists,
    // concatenated -- so the layers compose exactly as the grid path's do.
    std::vector<AnchorRecord> anchors;
    std::vector<SurfaceRecord> surfaces;
    // The anchor count after each sheet's bake -- the LAYER BOUNDARIES, so
    // the bone binding can be applied per layer.
    std::vector<size_t> layer_ends;
    for (const Sheet& sh : sheets) {
        MeshDerivation s;
        if (!BakeDerivedSheet(*sh.el, sh.origin, sh.scale, sample_n, extent,
                              anchors, surfaces, s, why))
            return false;
        layer_ends.push_back(anchors.size());
        // The counts ACCUMULATE across the sheets.
        stats.quads.insert(stats.quads.end(), s.quads.begin(), s.quads.end());
        stats.quads_from_a += s.quads_from_a;
        stats.quads_from_b += s.quads_from_b;
        stats.quads_averaged += s.quads_averaged;
        stats.triangles_unpaired += s.triangles_unpaired;
        stats.triangles_a += s.triangles_a;
        stats.triangles_b += s.triangles_b;
        stats.pairable_edges_a += s.pairable_edges_a;
        stats.pairable_edges_b += s.pairable_edges_b;
        // THE PER-MESH AREAS ACCUMULATE TOO.
        //
        // WHY THEY MUST BE COPIED HERE: this function accumulates the
        // counts field-by-field, so a new field that is not added here is
        // silently ZERO -- and a caller that trusts it measures a coverage
        // of 0%. (Measured: the derived bake reported 0.0% coverage until
        // this was added.)
        stats.area_a += s.area_a;
        stats.area_b += s.area_b;
    }
    stats.points.clear();

    // The BONE BINDING: an anchor belongs to the bone of the sheet it came
    // from. The derived baker does not know the bone index (it is a
    // per-sheet leaf baker), so it is applied HERE -- and it MUST be
    // applied per LAYER, because each sheet contributes its own block of
    // anchors and the sheets have different bones.
    //
    // The layer boundaries are the anchor counts before and after each
    // sheet's bake, which is exactly what the bake loop observed.
    {
        size_t at = 0;
        for (const Sheet& sh : sheets) {
            // this sheet's anchors are [at, layer_end)
            const size_t layer_end = layer_ends[&sh - &sheets[0]];
            const uint32_t bi = (sh.bone >= 0) ? (uint32_t)sh.bone : 0xFFFFFFFFu;
            for (size_t i = at; i < layer_end && i < anchors.size(); ++i)
                anchors[i].bone_index = bi;
            at = layer_end;
        }
    }

    // The bone hierarchy, as frozen BoneRecords -- the same block the grid
    // bake emits (see Bake). The bone records carry the WORLD-space bind
    // pose.
    {
        std::vector<int> parent_of(bones.size(), -1);
        for (size_t i = 0; i < bones.size(); ++i) {
            const std::string& pn = bones[i]->parent_name;
            if (pn.empty()) continue;
            for (size_t j = 0; j < bones.size(); ++j)
                if (bones[j]->name == pn) { parent_of[i] = (int)j; break; }
        }
        for (size_t i = 0; i < bones.size(); ++i) {
            BoneRecord br{};
            const Bone& b = *bones[i];
            const BoneFrame bf = BoneWorldFrame(b, Vec3{0,0,0}, Vec3{1,1,1});
            const uint32_t bi = (uint32_t)i;
            const int par = (parent_of[i] >= 0) ? parent_of[i] : -1;
            br.parent = (par >= 0) ? (uint32_t)par : 0xFFFFFFFFu;
            br.bind_pos[0] = bf.head.x; br.bind_pos[1] = bf.head.y; br.bind_pos[2] = bf.head.z;
            br.bind_rot[0] = 0.f; br.bind_rot[1] = 0.f; br.bind_rot[2] = 0.f; br.bind_rot[3] = 1.f;
            br.length = bf.length;
            // the reserved bytes carry the authored physical properties
            float* res = (float*)br.reserved;
            res[0] = b.mass;
            br.reserved[4] = (uint8_t)(std::max(0.f, std::min(1.f, b.radius)) * 255.f);
            br.reserved[5] = (uint8_t)(std::max(0.f, std::min(1.f, b.stiffness)) * 255.f);
            std::string mat = b.material;
            size_t k = 0;
            for (; k < mat.size() && k < 22; ++k) br.reserved[6 + k] = (uint8_t)mat[k];
            br.reserved[6 + k] = 0;
            (void)bi;
            out.bones.push_back(br);
        }
    }

    out.anchors = std::move(anchors);
    out.surfaces = std::move(surfaces);
    if (out.anchors.empty() || out.surfaces.empty()) {
        why = "the derived mesh produced no records";
        return false;
    }
    return true;
}

bool ElementSystem::Bake(const Element& root, int grid_n, float extent,
                         ModelOutput& out, std::string& why,
                         Vec3 world_offset) {
    out = ModelOutput{};
    if (grid_n < 2) { why = "grid_n must be >= 2"; return false; }
    if (grid_n > 64) grid_n = 64;
    if (extent <= 1e-6f) { why = "extent must be > 0"; return false; }
    const int N_requested = grid_n;
    int N = N_requested;
    int npatch = N - 1;

    // The sheets to bake. A tree with ONE sheet keeps the exact historical
    // behaviour (blended field, one layer). A tree with SEVERAL sheets is
    // baked per sheet so closed bodies close.
    //
    // The BONES are collected in the same pass, in the SAME order the parser
    // used to resolve Bone::parent into an index -- so the parent index in
    // the emitted BoneRecords refers to the right bone.
    std::vector<const Bone*> bones;
    CollectBones(root, bones);

    // --- the bone hierarchy, as frozen BoneRecords -------------------------
    //
    // The bone records carry the WORLD-space bind pose: the pivot (head),
    // the segment's length, and the parent index. bind_rot is the identity
    // quaternion (the bind pose is the authored pose -- there is no separate
    // rest pose yet), and the reserved bytes carry the authored radius,
    // mass and stiffness so the physics/field layers can read them without
    // re-parsing the tree.
    //
    // WHY world space: a bone's bind pose is what the animation system
    // transforms. It must be in the same frame as the anchors it drives,
    // which are world-space.
    //
    // --- COARSENESS: the SKELETON's density comes from the same level ---
    // The level's budget says what fraction of the bones to keep. A joint
    // nobody can see is a joint nobody can afford to solve, so at a coarse
    // level the DEEPEST bones are dropped first -- a leaf joint is the one
    // whose removal changes the least.
    //
    // Dropping a bone must not orphan its children: a child whose parent
    // was dropped is re-parented to the dropped bone's parent, so the
    // hierarchy stays a tree.
    //
    // The tree's bones are NOT mutated. A bake must be side-effect free --
    // baking the same tree twice must give the same skeleton, and mutating
    // Bone::parent would make the SECOND bake different from the first. So
    // the pruned hierarchy lives in a separate parent map.
    std::vector<int> parent_of(bones.size(), -1);
    for (size_t i = 0; i < bones.size(); ++i) parent_of[i] = bones[i]->parent;
    {
        const float keep = Element::BoneKeepForLod(root.lod < 0 ? 1 : root.lod);
        const size_t want = size_t(std::ceil(float(bones.size()) * keep));
        if (want < bones.size() && want >= 1) {
            // Drop the deepest bones first: a bone's depth is how far it is
            // from a root in the parent chain.
            std::vector<int> depth(bones.size(), 0);
            for (size_t i = 0; i < bones.size(); ++i) {
                int d = 0;
                for (int g = parent_of[i]; g >= 0 && d < 64; ) {
                    if (g >= int(bones.size())) break;
                    ++d;
                    g = parent_of[g];
                }
                depth[i] = d;
            }
            // Mark the deepest bones for removal until the budget is met.
            std::vector<char> drop(bones.size(), 0);
            size_t left = bones.size();
            while (left > want) {
                int deepest = -1;
                for (size_t i = 0; i < bones.size(); ++i)
                    if (!drop[i] && (deepest < 0 || depth[i] > depth[deepest]))
                        deepest = int(i);
                if (deepest < 0) break;
                drop[deepest] = 1;
                --left;
            }
            // Rebuild the list, re-parenting the orphans.
            std::vector<const Bone*> kept;
            std::vector<int> remap(bones.size(), -1);
            for (size_t i = 0; i < bones.size(); ++i)
                if (!drop[i]) { remap[i] = int(kept.size()); kept.push_back(bones[i]); }
            std::vector<int> kept_parent;
            kept_parent.reserve(kept.size());
            for (size_t i = 0; i < bones.size(); ++i) {
                if (drop[i]) continue;
                int p = parent_of[i];
                // Walk up past any dropped ancestor.
                while (p >= 0 && p < int(bones.size()) && drop[p]) p = parent_of[p];
                kept_parent.push_back((p >= 0 && p < int(bones.size())) ? remap[p] : -1);
            }
            bones.swap(kept);
            parent_of.swap(kept_parent);
        }
    }
    std::vector<Sheet> sheets;
    CollectSheets(root, sheets, &bones);
    if (sheets.empty()) { why = "the tree has no substance to bake"; return false; }

    // --- the SAMPLING REFINEMENT --------------------------------------------
    //
    // A substance whose shape does not fill its square domain asks for a
    // finer grid. See EffectiveGridN for the rule and the reason the helper
    // is shared with BakeBody.
    N = EffectiveGridN(N_requested, sheets);
    npatch = N - 1;

    out.bones.clear();
    out.bones.reserve(bones.size());
    for (const Bone* b : bones) {
        // The bone's own world frame: its element's origin/scale, which the
        // sheet traversal already resolved. A bone whose element has no
        // substance is not in `sheets`, so its frame is resolved here by
        // walking the tree for the owning element.
        Vec3 origin{0.f,0.f,0.f}, scale{1.f,1.f,1.f};
        {
            // Find the owning element's world frame by re-walking with the
            // same rules CollectSheets used.
            struct F { const Element* el; Vec3 o, s; };
            std::vector<F> st{{&root, Vec3{0,0,0}, Vec3{1,1,1}}};
            while (!st.empty()) {
                F f = st.back(); st.pop_back();
                if (f.el->has_bone && &f.el->bone == b) {
                    origin = f.o; scale = f.s;
                    break;
                }
                for (auto& c : f.el->children)
                    st.push_back({c.get(),
                                  { f.o.x + c->origin.x*f.s.x, f.o.y + c->origin.y*f.s.y,
                                    f.o.z + c->origin.z*f.s.z },
                                  { c->scale.x*f.s.x, c->scale.y*f.s.y, c->scale.z*f.s.z }});
            }
        }
        const BoneFrame bf = BoneWorldFrame(*b, origin, scale);
        BoneRecord br{};
        // The parent index comes from the PRUNED map, not from the tree's
        // Bone::parent -- at a coarse level the two differ.
        //
        // The index of THIS bone in the flat list is found by SEARCH, not by
        // pointer arithmetic. `b` is a pointer to a Bone OBJECT; `bones` is a
        // vector of POINTERS TO Bone. Subtracting them (`&b - bones.data()`)
        // is arithmetic between unrelated types -- undefined behaviour, and
        // in practice it always yields 0, so every bone inherited bone 0's
        // parent and the hierarchy silently flattened.
        size_t bi = SIZE_MAX;
        for (size_t i = 0; i < bones.size(); ++i)
            if (bones[i] == b) { bi = i; break; }
        const int par = (bi != SIZE_MAX && bi < parent_of.size()) ? parent_of[bi] : -1;
        br.parent = (par >= 0 && size_t(par) < bones.size())
                        ? uint32_t(par) : 0xFFFFFFFFu;
        br.bind_pos[0] = bf.head.x; br.bind_pos[1] = bf.head.y; br.bind_pos[2] = bf.head.z;
        // Identity quaternion: the bind pose IS the authored pose.
        br.bind_rot[0] = 0.f; br.bind_rot[1] = 0.f; br.bind_rot[2] = 0.f; br.bind_rot[3] = 1.f;
        br.length = bf.length;
        // The reserved bytes carry the authored physical properties, so the
        // inertia and field layers can read them straight from the frozen
        // record -- no re-parse, no side table.
        //
        // --- WHY MASS IS A FLOAT, NOT A NORMALISED BYTE ---
        // Mass is UNBOUNDED: a 2 kg housing and a 20 kg frame are both
        // legitimate. The earlier encoding packed it as a normalised byte
        // (`min(1, mass)`), which meant an authored mass of 2.0 stored as
        // 1.0 -- and 0.5 stored as 0.5 -- so the mass was silently
        // CLAMPED. Two bodies with masses 2.0 and 8.0 were
        // indistinguishable. That is exactly the kind of loss that makes a
        // mass property wrong rather than missing.
        //
        // The reserved bytes are 28 -- plenty of room for a float. The
        // layout is:
        //   reserved[0..3]   float  mass       (the authored mass, unbounded)
        //   reserved[4]      uint8  radius     (normalised 0..1)
        //   reserved[5]      uint8  stiffness  (already 0..1)
        //   reserved[6..27]  the MATERIAL NAME, as a NUL-terminated string
        //
        // WHY THE MATERIAL IS A NAME AND NOT ITS PROPERTIES: a bone's
        // field properties (permeability, conductivity, ...) are a whole
        // struct, and 22 bytes cannot hold them. Storing the NAME means
        // the registry stays the single source of truth for the numbers --
        // so a "steel" bone and a "steel" bone in two different documents
        // get IDENTICAL constants, and the material can be RE-resolved if
        // the registry is ever extended.
        //
        // The name is truncated at 22 bytes (the space available). A
        // longer name is TRUNCATED rather than rejected, because the name
        // is descriptive -- the physics comes from the registry, and a
        // truncated name still identifies the material's FIRST WORDS.
        // (The registry's lookup strips separators, so "cortical bone"
        // truncates to "cortical bone" at 13 bytes and survives intact.)
        {
            std::memcpy(br.reserved, &b->mass, sizeof(float));
            br.reserved[4] = uint8_t(std::max(0.f, std::min(1.f, b->radius)) * 255.f);
            br.reserved[5] = uint8_t(std::max(0.f, std::min(1.f, b->stiffness)) * 255.f);
            for (int k = 6; k < 28; ++k) br.reserved[k] = 0;
            if (!b->material.empty()) {
                const size_t room = 21;      // 22 bytes, one for the NUL
                const size_t n = std::min(b->material.size(), room);
                std::memcpy(br.reserved + 6, b->material.data(), n);
                br.reserved[6 + n] = 0;
            }
        }
        out.bones.push_back(br);
    }

    // The anchor index space is uint16 (SurfaceRecord::anchor_idx), so the
    // TOTAL across all layers must fit. Report it rather than silently
    // truncating a layer into an invalid record set.
    const size_t max_anchors = 0xFFFFu;
    if (sheets.size() * (size_t)N * N > max_anchors) {
        why = "too many anchors for the uint16 index space (" +
              std::to_string(sheets.size() * (size_t)N * N) + " > 65535); "
              "use a smaller grid_n or fewer sheets";
        return false;
    }

    // --- sample EVERY sheet on the N x N anchor grid ------------------------
    // pts/nrm/col/ok are indexed [sheet][(j*N)+i].
    const size_t nsheets = sheets.size();
    const size_t per = (size_t)N * N;
    std::vector<Vec3> pts(per * nsheets), nrm(per * nsheets), col(per * nsheets);
    std::vector<char> ok(per * nsheets, 0);
    Vec3 mn{1e30f,1e30f,1e30f}, mx{-1e30f,-1e30f,-1e30f};
    for (size_t si = 0; si < nsheets; ++si) {
        const Sheet& sh = sheets[si];
        // The sheet's OWN sampling domain, in its own frame.
        //
        // WHY PER-SHEET AND NOT THE SHARED CANVAS: the canvas is sized for
        // the whole tree, so a substance much smaller than the canvas would
        // get almost no samples -- and the few it got were clustered
        // wherever its support happened to cross the grid. Measured on a
        // rod (sx=0.5, sy=0.02) on a 1.0 canvas: 36 of 1024 samples in
        // support, and 990 of the 1024 anchors were the out-of-support
        // default at the bbox centre -- so the body's MASS sat at one point
        // and its inertia tensor was 25x too small.
        //
        // The grid is now stretched to the substance's domain, so the same
        // N x N samples cover the substance wherever it is. A substance
        // with no stated domain keeps the canvas.
        //
        // The domain is CLIPPED to the canvas. The canvas is the world the
        // tree occupies, so a support reaching past it is not represented
        // anyway -- and clipping keeps the grid step from getting COARSER
        // than it was before the per-substance domains existed (a gaussian
        // with 3 sigma = 0.9 would otherwise stretch the grid over 1.8
        // units and halve the resolution inside the canvas).
        // The sheet's OWN sampling domain, in its own frame. MUST be the
        // SAME domain the patch-fit target uses (see SheetDomain).
        float hx = 0.f, hy = 0.f;
        SheetDomain(*sh.el, extent, hx, hy);
        // THE POLAR SAMPLING (G21).
        //
        // A substance whose support is an ANNULUS (a torus) is sampled in
        // POLAR coordinates: the grid's i runs AROUND the axis and the j
        // runs ACROSS the tube. The grid is still N x N -- the cells are
        // just placed on the annulus instead of on the square, so every
        // cell lands on the support.
        //
        // MEASURED before: the fat torus wasted 38% of its samples and the
        // thin one 65%. The waste was not free -- each wasted sample was
        // written as a well-formed anchor at the bbox centre, which
        // corrupted the mass distribution (G9) and inflated every upload.
        //
        // THE MAPPING: with the grid's (i, j) in [0,1]^2,
        //     angle = 2*pi*i,  radius = r_min + (r_max - r_min)*j
        // so the ANGULAR index wraps around (and its last cell joins its
        // first) and the RADIAL index crosses the tube.
        //
        // WHY i IS THE ANGLE: the grid's i axis is the one that WRAPS, so
        // its connectivity must be periodic -- handled below, where the
        // +i neighbour of the last column is the first column.
        const bool polar = sh.el->substance && sh.el->substance->IsPolar();
        float r_min = 0.f, r_max = 1.f;
        if (polar) sh.el->substance->PolarRadius(r_min, r_max);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const float u = i / float(N - 1), v = j / float(N - 1);
                // The sheet's local (x, y): the grid is the sheet's OWN
                // domain, so a sheet with its own origin and scale samples
                // its own region.
                float lx = (u - 0.5f) * (hx * 2.0f);
                float ly = (v - 0.5f) * (hy * 2.0f);
                if (polar) {
                    // THE ANGULAR AXIS MUST NOT DUPLICATE ITS ENDPOINT.
                    //
                    // The grid's i runs 0..N-1 around the FULL circle, so
                    // i = N-1 would land at angle 2*pi -- the SAME point as
                    // i = 0. That duplicated column put an anchor exactly
                    // on the tube's inner rim (r = major - minor), which
                    // the hole test correctly flags as "an anchor inside
                    // the hole".
                    //
                    // The fix: the angular parameter runs over N-1
                    // intervals, so the cells are CENTRES of equal arcs
                    // and the last one stops just short of 2*pi. The
                    // connectivity WRAPS (handled below), so the tube is
                    // still closed around the axis.
                    const float ang = 6.283185307179586f *
                        ((float(i) + 0.5f) / float(N));
                    const float rad = r_min + (r_max - r_min) * v;
                    lx = rad * std::cos(ang);
                    ly = rad * std::sin(ang);
                }
                const Field f = SheetAt(sh, lx, ly);
                const size_t id = si * per + (size_t)j * N + i;
                // weight==0: no surface here (e.g. outside a bounded
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
    }   // closes the per-sheet SAMPLING loop
    //
    // BRACE-BALANCE NOTE (a real bug that shipped once):
    // This closing brace was MISSING. Without it the per-sheet sampling loop
    // stayed open, so the ANCHOR-writing loop below was NESTED INSIDE it and
    // ran once PER SHEET -- writing nsheets * N * N anchors instead of N * N.
    // The surface count looked right (that loop has its own headers), so the
    // symptom was only a bloated anchor blob whose upper blocks were never
    // referenced by any surface: 9216 anchors where 4608 were expected.
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
    // One grid PER SHEET, concatenated. The anchor index of a layer's
    // anchor is `si * (N*N) + (j*N+i)`.
    out.anchors.reserve(per * nsheets);
    for (size_t si = 0; si < nsheets; ++si) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const size_t id = si * per + (size_t)j * N + i;
            AnchorRecord a{};
            // BONE BINDING: an anchor belongs to the bone of the sheet it
            // came from. A sheet with no bone keeps 0xFFFFFFFF ("no bone"),
            // which is the same sentinel the rest of the engine uses.
            //
            // This is the whole point of the bone vocabulary: the surface
            // records stay byte-identical, but every anchor now says which
            // rigid segment it hangs from. That is what lets a pose change
            // move the skin with the bone, and what makes mass and inertia
            // computable over the skeleton.
            a.bone_index = (sheets[si].bone >= 0)
                             ? uint32_t(sheets[si].bone) : 0xFFFFFFFFu;
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
    }
    }

    // --- anchor CONNECTIVITY: the grid neighbours, within the same bone ---
    //
    // An anchor's connectivity names its neighbours in the surface. The
    // natural neighbours are the grid cells around it -- but ONLY those that
    // share its bone. Two adjacent anchors on DIFFERENT bones are not
    // connected: they can move independently, so a connectivity edge
    // between them would constrain a degree of freedom that is supposed to
    // be free (that is the whole point of an articulation).
    //
    // The four slots are the +i, -i, +j and -j neighbours, matching the
    // corner order the surface records use.
    for (size_t si = 0; si < nsheets; ++si) {
        const uint16_t base = (uint16_t)(si * per);
        const uint32_t bi = (sheets[si].bone >= 0) ? uint32_t(sheets[si].bone) : 0xFFFFFFFFu;
        // THE POLAR SHEET'S ANGULAR AXIS WRAPS (G21): the grid's i runs
        // AROUND the axis, so the last column's +i neighbour is the FIRST
        // column -- otherwise the seam at angle 0 would have no
        // connectivity across it, and the tube would split there.
        const bool polar = sheets[si].el->substance &&
                           sheets[si].el->substance->IsPolar();
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const size_t id = si * per + (size_t)j * N + i;
                if (!ok[id]) continue;                 // out-of-support: no neighbours
                AnchorRecord& a = out.anchors[id];
                int slot = 0;
                int nb[4][2] = {{i+1,j}, {i-1,j}, {i,j+1}, {i,j-1}};
                for (const auto& d : nb) {
                    if (slot >= 4) break;
                    int ni = d[0], nj = d[1];
                    // the angular wrap
                    if (polar && ni >= N) ni = 0;
                    if (polar && ni < 0)  ni = N - 1;
                    if (ni < 0 || ni >= N || nj < 0 || nj >= N) continue;
                    const size_t nid = si * per + (size_t)nj * N + ni;
                    if (!ok[nid]) continue;
                    // Only connect within the same bone.
                    if (out.anchors[nid].bone_index != bi) continue;
                    a.connectivity[slot++] = (uint16_t)nid;
                }
            }
    }

    // --- surface records (one patch per grid cell, PER SHEET) --------------
    out.surfaces.reserve((size_t)npatch * npatch * nsheets);
    const int G = 8;  // LSQ sample grid per patch (same as SplitPatch)
    for (size_t si = 0; si < nsheets; ++si) {
    // The sheet's sampling domain -- MUST match the anchor sampling
    // exactly. The patch-fit target below is evaluated on this domain; if
    // it used the canvas instead, the patch would be fitted against a
    // DIFFERENT part of the sheet than the corners it sits between, and
    // the fitted displacement would be wrong by the difference between the
    // two domains. That is the "dandelion spokes" artifact.
    float phx = 0.f, phy = 0.f;
    SheetDomain(*sheets[si].el, extent, phx, phy);
    for (int j = 0; j < npatch; ++j) {
        for (int i = 0; i < npatch; ++i) {
            // corner ordering matches the renderer bilinear: A0(0,0) A1(1,0) A2(1,1) A3(0,1)
            // The layer's base offset shifts every index into this layer's
            // own anchor block.
            const uint16_t base = (uint16_t)(si * per);
            const uint16_t a00 = (uint16_t)(base + (size_t)j * N + i);
            const uint16_t a10 = (uint16_t)(base + (size_t)j * N + i + 1);
            const uint16_t a11 = (uint16_t)(base + (size_t)(j + 1) * N + i + 1);
            const uint16_t a01 = (uint16_t)(base + (size_t)(j + 1) * N + i);
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
                    // target = THIS SHEET's own surface at this sample point.
                    //
                    // It must be the SHEET's surface, not the blended field:
                    // the blended field averages the upper and lower
                    // hemispheres into one point, so fitting against it
                    // would drag the upper layer back down toward the
                    // average -- which is the flat-bottomed dome again.
                    const float wu = (i + cu) / float(N - 1), wv = (j + cv) / float(N - 1);
                    // The target is evaluated on the SHEET'S OWN DOMAIN --
                    // the same domain the anchor grid was sampled on. Using
                    // the canvas here (as an earlier version did) samples a
                    // different part of the sheet than the patch's corners
                    // occupy, and the fitted displacement is then wrong by
                    // the difference between the two domains.
                    const float lx = (wu - 0.5f) * (phx * 2.0f);
                    const float ly = (wv - 0.5f) * (phy * 2.0f);
                    // THE POLAR SAMPLING MUST BE REPEATED HERE (G21).
                    //
                    // The patch-fit target is evaluated at the patch's own
                    // (u, v), and it MUST land on the same surface the
                    // patch's corners occupy. If the anchors were sampled
                    // in polar coordinates and the target is not, the fit
                    // targets a different part of the sheet -- and that is
                    // EXACTLY the "dandelion spokes" bug (the patch's
                    // displaced position comes from the anchor positions
                    // but its SHAPE is fitted to a target elsewhere).
                    float tx = lx, ty = ly;
                    if (sheets[si].el->substance &&
                        sheets[si].el->substance->IsPolar()) {
                        float pr_min = 0.f, pr_max = 1.f;
                        sheets[si].el->substance->PolarRadius(pr_min, pr_max);
                        const float ang = 6.283185307179586f *
                            ((wu * float(N - 1)) + 0.5f) / float(N);
                        const float rad = pr_min + (pr_max - pr_min) * wv;
                        tx = rad * std::cos(ang);
                        ty = rad * std::sin(ang);
                    }
                    const Field f = SheetAt(sheets[si], tx, ty);
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
            // The color indices are LOCAL to the layer (the global anchor
            // index is layer-offset), so they are offset back by `base`.
            const size_t l00 = (size_t)j * N + i;
            const size_t l10 = (size_t)j * N + i + 1;
            const size_t l11 = (size_t)(j + 1) * N + i + 1;
            const size_t l01 = (size_t)(j + 1) * N + i;
            Vec3 c{ (col[si*per+l00].x+col[si*per+l10].x+col[si*per+l11].x+col[si*per+l01].x)*0.25f,
                    (col[si*per+l00].y+col[si*per+l10].y+col[si*per+l11].y+col[si*per+l01].y)*0.25f,
                    (col[si*per+l00].z+col[si*per+l10].z+col[si*per+l11].z+col[si*per+l01].z)*0.25f };
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
    }
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

std::unique_ptr<PSFBody> ElementSystem::BakeBodyLod(
        const Element& root, int grid_n, float extent, std::string& why,
        Vec3 world_offset) {
    const int g = (grid_n < 0) ? Element::GridForLod(root.lod) : grid_n;
    return BakeBody(root, g, extent, why, world_offset);
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
    //
    // It MUST match Bake's bbox exactly, because the dequantization of the
    // stored positions uses this box. Bake now samples PER SHEET (each sheet
    // has its own origin and scale), so this must too -- using the blended
    // field here would compute a DIFFERENT box and every position would be
    // dequantized wrong.
    //
    // The bbox is RE-DERIVED here with the SAME per-sheet domain the baker
    // used. An earlier version sampled the shared canvas while the baker
    // sampled each sheet's own domain -- so the two boxes disagreed and
    // every position was dequantized against the wrong box, which showed up
    // as a rod whose mass sat at 25% of its true spread.
    Vec3 mn{1e30f,1e30f,1e30f}, mx{-1e30f,-1e30f,-1e30f};
    std::vector<Sheet> sheets;
    CollectSheets(root, sheets);
    // The SAME grid the baker used -- including the sampling refinement. If
    // this N differed from the baker's, the box would be different and every
    // stored position would be dequantized wrong.
    const int N = EffectiveGridN(grid_n, sheets);
    for (const Sheet& sh : sheets) {
        float hx = 0.f, hy = 0.f;
        SheetDomain(*sh.el, extent, hx, hy);
        // THE POLAR SAMPLING MUST BE REPEATED HERE TOO (G21): the bbox must
        // span exactly the points the baker sampled, or the dequantization
        // uses a different box than the quantization did.
        const bool polar = sh.el->substance && sh.el->substance->IsPolar();
        float pr_min = 0.f, pr_max = 1.f;
        if (polar) sh.el->substance->PolarRadius(pr_min, pr_max);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float u = i / float(N - 1), v = j / float(N - 1);
            float lx = (u - 0.5f) * (hx * 2.0f);
            float ly = (v - 0.5f) * (hy * 2.0f);
            if (polar) {
                const float ang = 6.283185307179586f *
                    ((float(i) + 0.5f) / float(N));
                const float rad = pr_min + (pr_max - pr_min) * v;
                lx = rad * std::cos(ang);
                ly = rad * std::sin(ang);
            }
            const Field f = SheetAt(sh, lx, ly);
            // MUST match Bake's bbox: out-of-support samples (weight==0) are
            // collapsed to the origin by the sheet evaluation and are excluded
            // there, so they must be excluded here too or dequant uses a
            // different box.
            if (f.weight <= 1e-9f) continue;
            mn.x = std::min(mn.x, f.point.x + world_offset.x); mn.y = std::min(mn.y, f.point.y + world_offset.y); mn.z = std::min(mn.z, f.point.z + world_offset.z);
            mx.x = std::max(mx.x, f.point.x + world_offset.x); mx.y = std::max(mx.y, f.point.y + world_offset.y); mx.z = std::max(mx.z, f.point.z + world_offset.z);
        }
    }
    if (mn.x > 1e29f) { why = "no in-support samples on the grid"; return nullptr; }
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
    if (semantic.find("torus") != std::string::npos ||
        semantic.find("donut") != std::string::npos ||
        semantic.find("ring") != std::string::npos) {
        root->kind = "torus_body";
        // A FULL TORUS, composed of TWO SHEETS.
        //
        // WHY: a PSF body is a GRAPH z = f(x,y) over a patch grid, so one
        // sheet can only ever describe the UPPER half of the tube. One sheet
        // alone gives a half-torus -- an open shell with no underside, whose
        // baked z-span is 0.1109 against the 0.2400 a full tube needs.
        //
        // The same rule the closed ELLIPSOID uses (and the one
        // validate_closed_body pins): the lower sheet keeps the SAME origin
        // and flips the SCALE. A negative z-scale already reflects the sheet
        // downward from the seam plane it shares with the upper one, so
        // `origin:[0,0,0]` + `scale:[1,1,-1]` puts the lower half exactly
        // under the upper half, and the two MEET at z = 0.
        //
        // MEASURED (before this change): the procedural torus's baked
        // z-span was 0.1109 -- exactly half of 0.2400 -- and its anchor
        // normals were 91.4% antiparallel to the view, which is what
        // produced the "only the forward-facing faces render" symptom.
        auto upper = std::make_unique<Element>();
        upper->kind = "torus_upper";
        upper->substance = std::make_unique<TorusSubstance>(0.32f, 0.12f);
        upper->color = {0.7f, 0.6f, 0.5f};
        root->AddChild(std::move(upper));

        auto lower = std::make_unique<Element>();
        lower->kind = "torus_lower";
        lower->substance = std::make_unique<TorusSubstance>(0.32f, 0.12f);
        lower->color = {0.7f, 0.6f, 0.5f};
        lower->origin = {0.f, 0.f, 0.f};
        lower->scale = {1.f, 1.f, -1.f};   // the flip
        root->AddChild(std::move(lower));
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