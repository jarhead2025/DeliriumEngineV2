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
private:
    float sx_, sy_, sz_;
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

    // Convenience builders (keep call sites readable).
    Element& AddChild(std::unique_ptr<Element> c) {
        children.push_back(std::move(c)); return *this;
    }
    Element& SetPresence(std::function<float(const Vec3&)> p) {
        presence = std::move(p); return *this;
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