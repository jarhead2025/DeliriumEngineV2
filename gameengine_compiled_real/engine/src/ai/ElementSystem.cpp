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
    Vec3 n{ d * x, d * y, z };
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
bool ElementSystem::Bake(const Element& root, int grid_n, float extent,
                         ModelOutput& out, std::string& why,
                         Vec3 world_offset) {
    out = ModelOutput{};
    if (grid_n < 2) { why = "grid_n must be >= 2"; return false; }
    if (grid_n > 64) grid_n = 64;
    if (extent <= 1e-6f) { why = "extent must be > 0"; return false; }
    const int N = grid_n;
    const int npatch = N - 1;

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
        auto put = [&](int i, float v) {
            const float c = std::max(0.f, std::min(1.f, v));
            br.reserved[i] = uint8_t(c * 255.f);
        };
        put(0, b->radius);                       // radius, normalised
        put(1, b->stiffness);                    // stiffness, already 0..1
        put(2, std::min(1.f, b->mass));           // mass, normalised
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
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const float u = i / float(N - 1), v = j / float(N - 1);
                // The sheet's local (x, y): the grid is the canvas in the
                // sheet's OWN frame, so a sheet with its own origin and
                // scale samples its own domain.
                const float lx = (u - 0.5f) * extent;
                const float ly = (v - 0.5f) * extent;
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
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const size_t id = si * per + (size_t)j * N + i;
                if (!ok[id]) continue;                 // out-of-support: no neighbours
                AnchorRecord& a = out.anchors[id];
                int slot = 0;
                const int nb[4][2] = {{i+1,j}, {i-1,j}, {i,j+1}, {i,j-1}};
                for (const auto& d : nb) {
                    if (slot >= 4) break;
                    if (d[0] < 0 || d[0] >= N || d[1] < 0 || d[1] >= N) continue;
                    const size_t nid = si * per + (size_t)d[1] * N + d[0];
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
                    const float lx = (wu - 0.5f) * extent, ly = (wv - 0.5f) * extent;
                    const Field f = SheetAt(sheets[si], lx, ly);
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
    Vec3 mn{1e30f,1e30f,1e30f}, mx{-1e30f,-1e30f,-1e30f};
    const int N = std::max(2, std::min(64, grid_n));
    std::vector<Sheet> sheets;
    CollectSheets(root, sheets);
    for (const Sheet& sh : sheets)
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float u = i / float(N - 1), v = j / float(N - 1);
            const Field f = SheetAt(sh, (u - 0.5f) * extent, (v - 0.5f) * extent);
            // MUST match Bake's bbox: out-of-support samples (weight==0) are
            // collapsed to the origin by the sheet evaluation and are excluded
            // there, so they must be excluded here too or dequant uses a
            // different box.
            if (f.weight <= 1e-9f) continue;
            mn.x = std::min(mn.x, f.point.x + world_offset.x); mn.y = std::min(mn.y, f.point.y + world_offset.y); mn.z = std::min(mn.z, f.point.z + world_offset.z);
            mx.x = std::max(mx.x, f.point.x + world_offset.x); mx.y = std::max(mx.y, f.point.y + world_offset.y); mx.z = std::max(mx.z, f.point.z + world_offset.z);
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
        root->substance = std::make_unique<TorusSubstance>(0.32f, 0.12f);
        root->color = {0.7f, 0.6f, 0.5f};
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