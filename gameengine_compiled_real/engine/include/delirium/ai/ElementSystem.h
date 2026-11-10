// =============================================================================
// ElementSystem.h - GENERIC object representation as a recursive element tree.
//
// Design intent (Josias, 2026-10-26): an object is NOT a mesh and NOT a fixed
// primitive. It is a hierarchy of ELEMENTS (systems) that OVERLAP and INTERACT
// per geometric location -- the way a mammal is composed of systems (skeletal,
// muscular, organ) that coexist and influence each other at every point in
// space. Decomposition is recursive: a system is built from sub-systems, which
// are built from sub-systems, until you reach the physical SUBSTANCE (the leaf
// material behavior).
//
// This header is deliberately CATEGORY-AGNOSTIC. It defines the representation
// (Element tree + Substance leaves) and the BAKER that reduces ANY such tree to
// the engine's native frozen 64-byte PSF records. The future CATEGORIZATION
// method plugs in by EMITTING element trees (assigning kinds, locations,
// presence masks, substances, weights) -- it does not touch the baker.
//
//   Element (system)
//     kind        : category tag (string) -- the categorizer assigns these
//     origin/scale: the element's geometric location (local frame)
//     presence    : world-location -> [0,1] mask (WHERE this system is active)
//     weight      : contribution strength at an overlap
//     substance   : leaf physical material (null for pure grouping)
//     children    : overlapping sub-systems (recursive)
//
// Composition rule (the "interacting per geometric location"): at a world
// location L, EVERY active element in the tree votes. The surface point is the
// WEIGHTED AVERAGE of the contributing systems' surface points at L; the normal
// is the weighted sum of their normals; the color is the weighted average.
// Overlap is first-class: two systems at the same location blend, they do not
// cull each other.
//
// The baker samples this aggregate field onto a parametric patch grid and fits
// each patch's 8 corner-vanishing coefficients with the SAME least-squares the
// engine uses internally (PSFBody::SplitPatch), so the result is a valid,
// live, renderable/physical PSFBody -- no special-casing per object type.
// =============================================================================
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include "../core/Types.h"
#include "../ai/ReconstructionModel.h"   // ModelOutput

namespace DeliRium {

// A point on a substance surface in the element's LOCAL frame, with a unit
// local normal. The element's origin/scale map this to world space. `valid`
// is false when the substance has no surface at this local location (bounded
// domains, e.g. outside an ellipsoid) -- the field composer then skips it.
struct SubstancePoint { Vec3 p; Vec3 n; bool valid = true; };

// -----------------------------------------------------------------------------
// ISubstance - the leaf "physical substance". Maps LOCAL frame coordinates
// (x, y) to a local surface point + unit normal. This is the bottom of the
// decomposition: the irreducible material behavior. The generic model is a
// GRAPH over the local X/Y plane (z = f(x, y)) -- the natural "substance
// occupies a region" model, and the same graph the PSF patch format uses.
// Concrete substances are generic building blocks (plane, ellipsoid
// hemisphere, gaussian system, heightfield) -- NOT named objects. A
// categorizer composes them into systems; a closed body is built from several
// such graph elements arranged in 3D (a future extension).
// -----------------------------------------------------------------------------
class ISubstance {
public:
    virtual ~ISubstance() = default;
    virtual SubstancePoint Eval(float x, float y) const = 0;

    // The region this substance actually occupies, in its own frame.
    //
    // WHY THIS EXISTS: the baker samples each sheet on a grid. If the grid
    // is sized to the shared CANVAS (the whole tree's extent) rather than to
    // the substance's own support, then a substance much smaller than the
    // canvas gets almost NO samples -- and the ones it does get are
    // clustered wherever the support happens to cross the grid.
    //
    // Measured (a rod: sx=0.5, sy=0.02, on a 1.0-wide canvas, grid 32):
    // only 36 of 1024 samples (3.5%) landed in support, and 990 of the
    // 1024 anchors were the out-of-support default at the bbox centre. The
    // body looked fine but its MASS sat at one point -- so its inertia
    // tensor was 25x too small.
    //
    // The default is the full canvas [-0.5,0.5]^2, which is correct for
    // the unbounded substances (plane, heightfield). A bounded substance
    // OVERRIDES it with its own footprint.
    //
    // `half_x`/`half_y` are the HALF-EXTENTS of the sampling region, centred
    // on the substance's own origin. The baker CLIPS this to the canvas: the
    // canvas is the world the tree occupies, so a support that reaches past
    // it is not represented anyway -- and clipping keeps the grid step from
    // getting COARSER than it was before the per-substance domains existed.
    virtual void Domain(float& half_x, float& half_y) const {
        half_x = 0.5f; half_y = 0.5f;
    }

    // -------------------------------------------------------------------------
    // THE SAMPLING REFINEMENT.
    //
    // A substance can ask the baker for a FINER grid than the default, when
    // its own shape needs one. The unit is "how many grid cells must span
    // this substance's SMALLEST FEATURE" -- for a torus, the tube's width.
    //
    // WHY THIS EXISTS: the baker's grid is a square of N x N cells over the
    // substance's domain. A torus's support is an ANNULUS, so the square
    // wastes most of its cells on the hole -- and what matters is the cells
    // that land on the TUBE, which must be enough to resolve its inner rim.
    //
    // Measured (major 0.4, minor 0.06, grid 48): the grid step over the
    // 0.92-wide square is 0.0196, so the 0.12-wide tube was only 6.1 cells
    // across -- below the 8 cells needed for a clean inner rim. The torus's
    // patch count was 612 against the sphere's 3250 on the same grid, and
    // the rim came out serrated.
    //
    // The refinement is a FACTOR applied to the requested grid size, not an
    // absolute count, so it scales with the user's chosen resolution (the
    // coarseness budget). A substance that needs no refinement returns 1.
    virtual float SamplingRefinement() const { return 1.f; }

    // -------------------------------------------------------------------------
    // THE RIM MARGIN.
    //
    // A bounded substance has an EDGE, and at that edge the surface stops
    // abruptly. When two sheets are meant to meet (a closed body: an upper
    // hemisphere and a flipped lower one), their edges land at DIFFERENT
    // places -- the grid's outermost row lands at |z| > 0, never exactly at
    // the seam. So the two sheets leave a GAP of height 2*|z_rim| straight
    // through the body, which renders as a vertical slit with serrated
    // teeth.
    //
    // The rim margin fixes this the way the USER described it: "surfaces
    // that overlap at edges and average out during calculations ... as long
    // as they're pre calculated enough to fit the original geometry". Each
    // sheet's edge is EXTENDED past its footprint into a rim band, and the
    // rim carries the EDGE's own tangent direction -- so the two sheets
    // overlap in an annulus where their normals AGREE, instead of meeting
    // at a knife edge where they oppose.
    //
    // The rim's width is a FRACTION of the footprint, so it scales with the
    // substance and stays proportional to the grid step at any resolution.
    //
    // `RimMargin` returns the rim's width as a fraction of the footprint's
    // half-extent (0 = no rim). A substance with no bounded edge -- the
    // plane, the heightfield -- has no rim, because it has no edge to
    // extend.
    virtual float RimMargin() const { return 0.f; }

    // -------------------------------------------------------------------------
    // THE SUPPORT'S SHAPE.
    //
    // WHY THIS EXISTS (G21): the sampling grid is a SQUARE, so a substance
    // whose support is not square-shaped spends most of its cells on
    // nothing. MEASURED at grid 48 on a 1.0 canvas:
    //
    //   plane / gaussian    in-support 100.0%   wasted  0.0%
    //   ellipsoid 0.5                85.4%          14.6%
    //   rod 0.5 x 0.02               75.3%          24.7%
    //   torus 0.32/0.12              62.0%          38.0%
    //   torus 0.40/0.06              34.7%          65.3%
    //
    // Every wasted sample is not free: the baker writes it as a
    // WELL-FORMED anchor at the bbox centre (the out-of-support
    // placeholder), so it costs memory, upload bandwidth, and -- because
    // the mass calculation counted those placeholders -- a WRONG inertia
    // tensor (that was G9: 970 of 1024 anchors for a rod sat at one
    // point, and its inertia tensor was 25x too small).
    //
    // A substance whose support is an ANNULUS (a torus) can say so here,
    // and the baker then samples in POLAR coordinates: the angular
    // direction around the axis, and the radial direction across the
    // tube. Every cell then lands on the support.
    //
    // THE CONTRACT: `IsPolar()` says "sample me in polar coordinates", and
    // then `PolarRadius(r_min, r_max)` gives the radial extent of the
    // support. The baker maps the grid's (i, j) to (angle, radius) so
    // that the ANGULAR index runs around the axis and the RADIAL index
    // runs across the tube.
    //
    // A substance that does not override this keeps the rectangular
    // sampling, unchanged.
    virtual bool IsPolar() const { return false; }
    // The radial extent of the support, for a polar substance: the inner
    // and outer radius of the annulus.
    virtual void PolarRadius(float& r_min, float& r_max) const {
        r_min = 0.f; r_max = 1.f;
    }
    // The angular step: how many cells the baker should place around the
    // full circle. A polar grid's resolution has TWO aspects (around and
    // across), and the angular one must be fine enough that the tube's
    // cross-section is not faceted. The default returns 0, meaning "use
    // the grid's own N".
    virtual int PolarAngularCells(int n_radial) const { return 0; }
};

// A flat plane z = 0 (infinite domain), facing +Z.
class PlaneSubstance : public ISubstance {
public:
    SubstancePoint Eval(float x, float y) const override;
};

// The upper hemisphere of an ellipsoid of half-extents (sx, sy, sz) centered
// at the local origin: z = sz*sqrt(1 - x^2/sx^2 - y^2/sy^2). The most generic
// closed-surface building block (a soft body, a limb, an organ). Bounded:
// `valid` is false outside the footprint x^2/sx^2 + y^2/sy^2 <= 1.
class EllipsoidSubstance : public ISubstance {
public:
    EllipsoidSubstance(float sx = 0.5f, float sy = 0.5f, float sz = 0.5f)
        : sx_(sx), sy_(sy), sz_(sz) {}
    SubstancePoint Eval(float x, float y) const override;
    // The footprint IS the domain: the samples must cover x in [-sx,sx],
    // y in [-sy,sy], or the sheet is sampled at a handful of points.
    //
    // THE DOMAIN INCLUDES THE RIM. The rim lives OUTSIDE the footprint, so
    // a domain of exactly (sx, sy) never samples it -- and the rim is the
    // whole point of the mechanism. The domain is therefore the footprint
    // PLUS the rim's reach.
    //
    // The rim's reach is `RimMargin() * half_extent`, and the rim extends
    // along the footprint's gradient, so the largest coordinate it reaches
    // is the footprint's half-extent times (1 + rim) in the worst case
    // (along an axis). That is the bound used here.
    void Domain(float& half_x, float& half_y) const override {
        const float rim = RimMargin();
        half_x = sx_ * (1.f + rim);
        half_y = sy_ * (1.f + rim);
    }
    // THE RIM. See ISubstance::RimMargin for what this is for.
    //
    // The rim is 6% of the footprint. That is wide enough to cover the
    // grid's last row (which lands within one grid step of the edge) at
    // any reasonable resolution, and narrow enough that it does not eat
    // into the body.
    float RimMargin() const override { return 0.06f; }
private:
    float sx_, sy_, sz_;
};

// The upper surface of a TORUS with its axis along +Z: a ring of centre
// radius `major` and tube radius `minor`.
//
//   r = sqrt(x^2 + y^2)
//   z = +sqrt(minor^2 - (r - major)^2)     for |r - major| <= minor
//
// WHY THIS IS THE RIGHT SHAPE FOR THE ENGINE: a PSF body is a graph
// z = f(x,y) over a rectangular grid of patches (see ISubstance above), so
// a substance CANNOT describe a closed overhang. What it CAN describe is
// the upper half of the tube, and that half already carries the torus's
// defining feature -- the HOLE. Inside r < major - minor there is no
// surface at all, so the baker marks those samples out-of-support and
// leaves the centre open. The result reads as a donut from any angle
// above the plane.
//
// The normal is the gradient of the implicit surface
//   F = (r - major)^2 + z^2 - minor^2,
// which points radially outward at the tube's outer edge and straight up
// over the centre of the tube -- exactly the orientation a torus needs.
class TorusSubstance : public ISubstance {
public:
    TorusSubstance(float major = 0.32f, float minor = 0.12f)
        : major_(major), minor_(minor) {}
    SubstancePoint Eval(float x, float y) const override;
    // The ring's outer extent: the tube reaches major + minor from the
    // axis, so the samples must cover that disc -- and must also reach the
    // INNER rim (major - minor), or the hole's edge is not resolved.
    void Domain(float& half_x, float& half_y) const override {
        const float r = major_ + minor_;
        half_x = r; half_y = r;
    }
    // The tube must be at least 8 grid cells across for its inner rim to
    // resolve cleanly. The grid step is 2*(major+minor)/(N-1), and the tube
    // is 2*minor wide, so:
    //
    //     cells = 2*minor / (2*(major+minor)/(N-1)) = minor*(N-1)/(major+minor)
    //
    // Requiring cells >= 8 gives N >= 8*(major+minor)/minor. The factor
    // returned here is that requirement relative to a "typical" grid of 48:
    // for the workspace torus (major 0.32, minor 0.12) it is
    // 8*0.44/0.12 / 48 = 0.61 -- i.e. NO refinement is needed, because the
    // fat tube is already well resolved. For a thin torus (major 0.4, minor
    // 0.06) it is 8*0.46/0.06 / 48 = 1.28 -- a 28% finer grid.
    //
    // The requirement is 8.5 cells, not 8.0: the cells-across figure is
    // computed from the grid's ACTUAL step, which is derived from the
    // integer N the baker picked, so an exact 8.0 target lands on 7.99...
    // after rounding. The margin keeps the invariant true rather than
    // borderline.
    float SamplingRefinement() const override {
        const float need = 8.5f * (major_ + minor_) / std::max(1e-6f, minor_);
        return std::max(1.f, need / 48.f);
    }

    // -------------------------------------------------------------------------
    // THE POLAR SUPPORT (G21).
    //
    // A torus's support is an ANNULUS, so a square grid spends most of its
    // cells on the hole: measured at grid 48, the fat torus wastes 38% of
    // its samples and the thin one 65%. The baker can sample the torus in
    // POLAR coordinates instead -- around the axis and across the tube --
    // and every cell then lands on the support.
    //
    // WHY THIS MATTERS BEYOND WASTE: every wasted sample is written as a
    // well-formed anchor at the bbox centre, so the waste also corrupts the
    // mass distribution (G9) and inflates every upload.
    bool IsPolar() const override { return true; }
    void PolarRadius(float& r_min, float& r_max) const override {
        // The tube spans [major - minor, major + minor] in radius. The
        // INNER radius is where the hole's edge is, so the tube's inner rim
        // is resolved exactly.
        r_min = std::max(0.f, major_ - minor_);
        r_max = major_ + minor_;
    }
    // The angular step. The tube's cross-section must not be faceted, so
    // there must be enough cells around the circle for the polygonal
    // approximation of the circle to stay within a fraction of the tube's
    // radius. The chord error of an N-gon of radius R is R*(1 - cos(pi/N)),
    // and the requirement is that it stays under 2% of the tube's radius:
    //
    //     R*(1 - cos(pi/N)) < 0.02 * minor
    //
    // With R = major + minor, solving for N gives a usable bound; the
    // factor of 2 below is the margin that keeps it true after the
    // integer rounding of the cell count.
    int PolarAngularCells(int n_radial) const override {
        const float R = major_ + minor_;
        // 1 - cos(pi/N) < 0.01 * minor / R  =>  cos(pi/N) > 1 - 0.01*minor/R
        // Using the small-angle form 1 - cos(x) ~ x^2/2:
        //     (pi/N)^2 / 2 < 0.01 * minor / R   =>   N > pi / sqrt(2*0.01*minor/R)
        const float need = 3.14159265f /
            std::sqrt(2.f * 0.01f * minor_ / std::max(1e-6f, R));
        const int want = int(std::ceil(need * 2.f));
        // At least as many angular cells as radial ones, so the tube is
        // not under-resolved around the circle.
        return std::max(n_radial * 2, want);
    }
private:
    float major_, minor_;
};

// A localized gaussian "system": a bump of height `amp` with falloff `sigma`
// centered at the local origin, on an otherwise flat plane. Models a subsystem
// that occupies a bounded region (a node, a cluster, a feature) and fades to
// the host plane at its edges -- so it blends cleanly where it overlaps others.
class GaussianSystemSubstance : public ISubstance {
public:
    GaussianSystemSubstance(float amp = 0.2f, float sigma = 0.35f)
        : amp_(amp), sigma_(sigma) {}
    SubstancePoint Eval(float x, float y) const override;
    // The bump fades out at ~3 sigma; sample there, or the tails are cut.
    void Domain(float& half_x, float& half_y) const override {
        const float r = 3.f * sigma_;
        half_x = r; half_y = r;
    }
private:
    float amp_, sigma_;
};

// The most generic leaf: wraps an arbitrary height field h(x, y) (infinite
// domain). The categorizer can supply any analytic/learned field here.
class HeightfieldSubstance : public ISubstance {
public:
    using HeightFn = std::function<float(float x, float y)>;
    explicit HeightfieldSubstance(HeightFn fn) : fn_(std::move(fn)) {}
    SubstancePoint Eval(float x, float y) const override;
private:
    HeightFn fn_;
};

// -----------------------------------------------------------------------------
// Bone - a RIGID SEGMENT of the object: a line segment (head -> tail) with an
// optional bend, a radius, and a mass.
//
// WHY THIS EXISTS (Josias, 2026-11-08): "object reconstruction is encoded
// based on line segment (bones) or points in space (articulations)". A
// SURFACE-only tree cannot express a mechanism: four overlapping ellipsoids
// say nothing about which part ROTATES relative to which. A bone says
// "this is the shaft, it runs from here to there, it pivots at its head,
// and it carries this much mass" -- which is what makes reparametrization,
// inertia, and field physics (magnetic/electric/radiative) computable.
//
// The bone is NOT the surface. It is the DEGREE OF FREEDOM the surface
// hangs from. An element may carry BOTH: a bone (a rigid segment) and a
// substance (the skin around it).
//
// The layout mirrors the frozen BoneRecord:
//   head -> bind_pos, tail -> head + length * axis, bend -> the direction
//   the segment deviates from straight (a joint).
// -----------------------------------------------------------------------------
struct Bone {
    std::string name;                      // "femur", "shaft", "rotor"
    int         parent = -1;               // index into Element::bones, -1 = root
    // The parent bone's NAME, as written in JSON. A model cannot know an
    // index that has not been assigned yet, so the parent is given by name
    // and resolved to an index after the whole tree is built. An unknown
    // name is REJECTED, not silently re-rooted.
    std::string parent_name;
    // The MATERIAL this part is made of, as a name a document would use:
    // "steel", "ABS plastic", "copper", "cortical bone".
    //
    // WHY A NAME AND NOT THE PROPERTIES: a document says "a steel
    // housing", not "mu_r = 200, sigma = 1e7". The name is the authored
    // thing; the NUMBERS come from the material registry
    // (physics/FieldProperties.h), so every steel part in the world gets
    // the SAME constants -- and a document can never state a physically
    // inconsistent material.
    //
    // An EMPTY material means "not stated" -- which is honest, and is NOT
    // the same as "generic plastic". The field layer treats a missing
    // material as no material at all rather than inventing one.
    std::string material;
    Vec3        head {0.f, 0.f, 0.f};       // pivot, in the element's LOCAL frame
    Vec3        tail {0.f, 0.f, 0.f};       // the far end of the segment
    Vec3        bend {0.f, 0.f, 0.f};       // the direction the segment deviates
    float       radius = 0.f;              // cross-section radius (0 = unspecified)
    float       mass = 0.f;                // mass contribution (0 = from volume)
    float       stiffness = 1.f;           // 0 = free to swing, 1 = rigid
    // The joints at this bone's ends, by name. An articulation is a POINT
    // in space where two bones meet -- named so a document can refer to
    // "the knee" rather than to an index.
    std::vector<std::string> joints;
};

// -----------------------------------------------------------------------------
// Element - one system in the tree. Either a pure grouping node (substance
// null, has children) or a leaf (substance set). Children OVERLAP: at a shared
// location all active children contribute (weighted).
// -----------------------------------------------------------------------------
struct Element {
    std::string kind;                       // category tag (categorizer-assigned)
    Vec3  origin{0.f, 0.f, 0.f};            // geometric location (world offset)
    Vec3  scale {1.f, 1.f, 1.f};            // local-frame scale (per axis)
    float weight  = 1.0f;                   // contribution strength at an overlap
    Vec3  color   {0.8f, 0.8f, 0.8f};       // this system's base color (blended)
    std::unique_ptr<ISubstance> substance;  // leaf material (null => grouping)
    // WHERE this system is active: world location -> [0,1]. Null => everywhere
    // (1.0). A categorizer uses this to localize a system to a region.
    std::function<float(const Vec3&)> presence;
    std::vector<std::unique_ptr<Element>> children;  // overlapping sub-systems

    // The rigid segment this element represents (optional). An element with
    // a bone AND a substance is a rigid part with skin; an element with a
    // bone and NO substance is a bare joint/axis.
    bool has_bone = false;
    Bone bone;

    // COARSENESS: the abstraction level this tree asks for, 0..3.
    //
    // WHY (Josias, 2026-11-08): "make them complete in representation,
    // denoting the coarseness for the sake of speedy computation". The
    // model should choose an ABSTRACTION LEVEL, not a pixel count. The
    // level then DERIVES the two budgets the engine actually spends:
    //
    //   lod 0 (fine)    grid_n 48   -- near-final surface
    //   lod 1 (medium)  grid_n 32   -- the default
    //   lod 2 (coarse)  grid_n 16   -- silhouette + mass only
    //   lod 3 (sketch)  grid_n 8    -- proportions and skeleton only
    //
    // The skeleton's own density follows from the same level: a coarser
    // tree keeps fewer bones, because a joint nobody can see is a joint
    // nobody can afford to solve.
    //
    // A tree with no stated level uses the default (1). The level lives on
    // the ROOT; a nested element may override it for its own subtree.
    int lod = -1;                           // -1 = not stated

    // The resolution this tree's coarseness implies, and the density of the
    // skeleton it keeps. Both are derived from `lod`; they are NOT chosen
    // independently, because the point is that ONE choice sets both.
    static int GridForLod(int lod) {
        switch (lod) {
            case 0:  return 48;
            case 1:  return 32;
            case 2:  return 16;
            case 3:  return 8;
            default: return 32;
        }
    }
    // The fraction of bones a tree at this level keeps: at lod 3 only the
    // ROOT bones survive (the proportions), at lod 0 every bone survives.
    static float BoneKeepForLod(int lod) {
        switch (lod) {
            case 0:  return 1.00f;
            case 1:  return 1.00f;
            case 2:  return 0.75f;
            case 3:  return 0.40f;
            default: return 1.00f;
        }
    }

    // Convenience builders (keep call sites readable).
    Element& AddChild(std::unique_ptr<Element> c) {
        children.push_back(std::move(c)); return *this;
    }
    Element& SetPresence(std::function<float(const Vec3&)> p) {
        presence = std::move(p); return *this;
    }
    // Attach a bone. The element must own it, so the segment's local frame
    // travels with the element's origin/scale.
    Element& SetBone(const Bone& b) { bone = b; has_bone = true; return *this; }

    // The name of this element's leaf substance ("plane", "ellipsoid",
    // "torus", "gaussian", "heightfield"), or "" when it is a grouping node.
    //
    // WHY: the Element stores the substance as an opaque ISubstance pointer,
    // so a caller cannot tell WHICH substance it holds. Without this, a
    // report of "the body came out flat" cannot distinguish "the model
    // chose a flat substance" from "the baker could not resolve the one it
    // chose" -- two completely different bugs with the same symptom.
    const char* SubstanceName() const {
        if (!substance) return "";
        const ISubstance* s = substance.get();
        if (dynamic_cast<const TorusSubstance*>(s))          return "torus";
        if (dynamic_cast<const EllipsoidSubstance*>(s))      return "ellipsoid";
        if (dynamic_cast<const GaussianSystemSubstance*>(s)) return "gaussian";
        if (dynamic_cast<const HeightfieldSubstance*>(s))    return "heightfield";
        if (dynamic_cast<const PlaneSubstance*>(s))          return "plane";
        return "unknown";
    }
};

// -----------------------------------------------------------------------------
// ElementSystem - the baker. Reduces ANY element tree to valid PSF records.
// Category-agnostic: it never inspects `kind`; it only composes the field and
// fits patches. This is the seam the categorization method drives.
// -----------------------------------------------------------------------------
class ElementSystem {
public:
    // The aggregate field at a world location: the weighted blend of every
    // active system's surface point / normal / color at that location.
    struct Field { Vec3 point; Vec3 normal; Vec3 color; float weight; };

    // Compose the tree's field at a single world location (exposed for tests
    // and for the future categorizer to query "what is at this point?").
    static Field FieldAt(const Element& root, const Vec3& world);

    // Bake the tree onto a grid_n x grid_n patch canvas of the given world
    // `extent` (canvas spans [-extent/2, extent/2]^2 in X/Y at Z=0, then
    // shifted by `world_offset`). Produces frozen 64-byte records in `out`
    // (ModelOutput). Returns false (with `why`) if the result would be
    // invalid (e.g. degenerate anchors).
    static bool Bake(const Element& root, int grid_n, float extent,
                     ModelOutput& out, std::string& why,
                     Vec3 world_offset = Vec3{0.f, 0.f, 0.f});

    // Bake + materialize into a live PSFBody (calls PSFBody::FromRecords).
    // Returns nullptr (with `why`) on failure.
    static std::unique_ptr<class PSFBody> BakeBody(const Element& root,
                                                   int grid_n, float extent,
                                                   std::string& why,
                                                   Vec3 world_offset = Vec3{0.f, 0.f, 0.f});

    // --- the COARSENESS-aware bake ----------------------------------------
    //
    // The tree's `lod` (0..3) DERIVES the surface resolution, so the model
    // chooses an abstraction level rather than a pixel count. `lod < 0`
    // (the tree states none) uses the default, grid 32.
    //
    //   grid_n < 0  ->  grid_n = Element::GridForLod(tree.lod)
    //   grid_n >= 0 ->  the caller's explicit resolution wins
    //
    // An explicit grid_n still overrides the tree, because a caller that
    // knows the budget it is paying for should not have to argue with the
    // model about it.
    static bool BakeLod(const Element& root, int grid_n, float extent,
                        ModelOutput& out, std::string& why,
                        Vec3 world_offset = Vec3{0.f, 0.f, 0.f}) {
        return Bake(root, grid_n < 0 ? Element::GridForLod(root.lod) : grid_n,
                    extent, out, why, world_offset);
    }
    // NOTE: defined OUT OF LINE (ElementSystem.cpp) because PSFBody is an
    // incomplete type here. An inline body would instantiate
    // unique_ptr<PSFBody>'s deleter, which needs sizeof(PSFBody) -- and
    // every translation unit including this header would fail to compile.
    static std::unique_ptr<class PSFBody> BakeBodyLod(
            const Element& root, int grid_n, float extent, std::string& why,
            Vec3 world_offset = Vec3{0.f, 0.f, 0.f});

    // The effective coarseness of a tree: the ROOT's stated level, or the
    // default (1, medium) when it states none.
    static int LodOf(const Element& root) {
        return root.lod < 0 ? 1 : root.lod;
    }
};

// -----------------------------------------------------------------------------
// ProceduralElementGenerator - builds GENERIC element trees from a semantic
// request. This is the stand-in for the future learned categorizer: today it
// maps a small set of semantic families to hand-authored (but generic) element
// trees; later a model emits the same trees. The baker is unchanged.
// -----------------------------------------------------------------------------
class ProceduralElementGenerator {
public:
    // Build an element tree for `semantic`. Always succeeds (unknown requests
    // fall back to a single plane substance -> a valid flat body).
    static std::unique_ptr<Element> Build(const std::string& semantic);
};

} // namespace DeliRium