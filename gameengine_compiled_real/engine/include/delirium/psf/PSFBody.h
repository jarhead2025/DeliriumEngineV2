// =============================================================================
// PSFBody.h - THE central runtime object of the engine.
// A body is not a mesh: it is a living set of equations -- patches,
// shared anchors, a seam graph, and a quadtree that can split or merge
// patches DURING gameplay. Every shape/physics operation in the engine
// ultimately resolves to editing constants inside this class.
// =============================================================================
#pragma once
#include <vector>
#include <unordered_set>
#include <memory>
#include <CL/cl.h>
#include "Patch.h"
#include "Seam.h"
#include "AnchorPool.h"
#include "../core/Types.h"

namespace DeliRium {

struct SurfaceHit { uint32_t patch; float u, v, t; Vec3 p; };

// Device-side mirror of a body's 64-byte blobs, as owned by the renderer.
// Pass a pointer to this as the `cl_queue` argument of PSFBody::FlushToGPU
// so the body can upload ONLY its dirty records (the whole point: edits
// are tiny, so a full-blob re-upload is wasted bandwidth every frame).
struct PSFFlushState {
    cl_command_queue queue = nullptr;
    cl_mem  anchors_buf  = nullptr;   // CL_MEM_READ_WRITE, n_anchors*64
    cl_mem  surfaces_buf = nullptr;   // CL_MEM_READ_WRITE, n_surfaces*64
    cl_mem  stitch_buf   = nullptr;   // CL_MEM_READ_WRITE, n_surfaces*8
    size_t  anchors_bytes  = 0;       // capacity in bytes (may exceed live)
    size_t  surfaces_bytes = 0;
    size_t  stitch_bytes   = 0;
    uint64_t flush_gen = 0;           // anchor_gen last fully uploaded
};

class PSFBody {
public:
    // --- construction ----------------------------------------------------
    static std::unique_ptr<PSFBody> FromFile(const char* path);
    static std::unique_ptr<PSFBody> FromRecords(
        std::vector<AnchorRecord> a, std::vector<SurfaceRecord> s,
        std::vector<int8_t> stitch, Vec3 bbmin, Vec3 bbspan, float s_scale);

    // --- live topology (scale principle: refine only where needed) -------
    // Split one patch into 4 children: hanging anchors snapped onto the
    // parent surface (deduped by quantized position), children fitted by
    // least squares against the parent evaluation, neighborhood
    // restitched. Parent record is REPLACED by child 0 (+3 appended), so
    // the blob never has retired holes. Returns the 4 child ids.
    bool SplitPatch(uint32_t patch_id, uint32_t children_out[4]);
    // Inverse: collapse 4 siblings when their joint refit error < tol.
    void MergePatches(uint32_t parent_id, float tol);
    // Merge the 4-patch group sharing interior center anchor `ctr` back
    // into one parent. Guarded: rejected (returns false, no change) if the
    // refit would lose more than `tol` (fraction of diagonal) of detail.
    bool MergeGroup(uint16_t ctr, float tol);
    bool FindMergeGroup(uint16_t ctr, uint32_t grp[4], uint16_t outer[4]) const;
    // Coarsen a region: merge every clean, low-error 4-group near center.
    // Detach a set of patches into a NEW independent body (used by
    // Fracture). The patches + their anchors are copied with remapped
    // indices; the originals are removed from this body and the blobs
    // compacted. Returns the new fragment (nullptr if the set is empty
    // or all patches).
    std::unique_ptr<PSFBody> DetachPatches(const std::vector<uint32_t>& ids);
    // Coarsen a region: merge every clean, low-error 4-group near center.
    int  CoarsenWhere(const Vec3& center, float radius, float tol,
                      int max_merges = 64);
    // Split patches near `center` until their size <= target_size
    // (the live AMR loop; max_splits caps the work per call).
    void RefineWhere(const Vec3& center, float radius, float target_size,
                     int max_splits = 64);

    // Radial dent: anchors inside the radius move along their normals,
    // patch coefficients are refit against original+dent target, then
    // the neighborhood is restitched. depth > 0 dents inward (-normal).
    void ApplyRadialDent(const Vec3& center, float radius, float depth);

    // --- live shape ------------------------------------------------------
    // Re-fit the 8 equation constants of every patch whose center lies
    // within `radius` of `center`, against the body's CURRENT surface
    // (snapshot -> LSQ on the corner-vanishing basis). This is the hook
    // for "anchors were edited externally (dent, AI edit, skinning) and
    // the coefficients must be re-optimized to reproduce the intended
    // surface". Idempotent on a clean body (refit recovers the same
    // coefs up to LSQ noise); after anchor edits it pulls the surface
    // back onto the (new) anchor bilinear + refit residual.
    void RefitLocal(const Vec3& center, float radius);
    // Apply a precomputed deformation mode scaled by k (ROM hook).
    // Mode shapes are registered by ModalDynamics::LoadModes
    // (RegisterModeShapes); the shape layout is [n_anchor] anchor
    // displacements (along rest normals) + [n_surf*8] coefficient
    // deltas, exactly as written back by ModalDynamics::WriteBack.
    // With no modes registered this is a safe no-op.
    void ApplyMode(uint32_t mode_id, float k);
    // Register (or replace) the ROM mode-shape table. Each shape is
    // [n_anchor + n_surf*8] floats, PSM v2 layout.
    void RegisterModeShapes(std::vector<std::vector<float>> shapes);
    uint32_t ModeShapeCount() const { return (uint32_t)modes.size(); }

    // --- queries ----------------------------------------------------------
    float  Volume() const;                       // TODO analytic (divergence thm)
    // Gauss-Newton on (u,v), seeded by the K nearest patch centers.
    bool   ClosestPointEx(const Vec3& p, SurfaceHit& out) const;
    Vec3   ClosestPoint(const Vec3& p) const;
    // Newton on (u,v,t): P(u,v) = o + t d. First (smallest-t) hit.
    bool   Raycast(const Vec3& o, const Vec3& d, SurfaceHit& out) const;

    // --- GPU sync ----------------------------------------------------------
    // Upload ONLY dirty 64-byte records (the whole point: edits are tiny).
    // `cl_queue` is actually a PSFFlushState* (renderer-owned device
    // buffers + capacities + generation). nullptr / incomplete state is a
    // safe no-op. Grew-past-capacity => full re-upload; anchor-dirty =>
    // whole anchor blob (+stitch); otherwise one small write per dirty
    // surface record.
    void   FlushToGPU(void* cl_queue);

    const std::vector<Patch>& Patches() const { return patches; }
    std::vector<Patch>&       MutablePatches() { return patches; }
    SeamGraph&                Seams()         { return seams; }
    AnchorPool&               Anchors()       { return anchors; }
    AnchorPool&               AnchorsConst() const { return const_cast<AnchorPool&>(anchors); }
    int8_t*                   StitchOf(uint32_t s) { return &stitch_blob[size_t(s)*8]; }

    // Patch views hold pointers into the blobs; MUST be called after any
    // operation that can reallocate them (SplitPatch does internally).
    void RebuildViews();

    // anchor/stitch blobs changed (renderer must re-upload / re-create)
    bool AnchorsDirty() const { return anchors_dirty; }
    void MarkAnchorsClean()   { anchors_dirty = false; }
    void MarkAnchorsDirty()   { anchors_dirty = true; ++anchor_gen; }

    Vec3  BBoxMin()  const { return bbox_min; }
    Vec3  BBoxSpan() const { return bbox_span; }
    // Grow the bbox by `margin` (fraction of current span) on every side,
    // re-quantizing all anchors so positions are preserved. Needed before
    // deformations that move anchors outside the current box (skinning,
    // large dents) -- otherwise SetPosition clamps and truncates them.
    void  ExpandBBox(float margin);
    // Rigidly move the whole body by `m` (world units): shift the bbox and
    // re-quantize every anchor at its translated position. Geometry, normals,
    // and coefficients are unchanged -- only the body's placement in the
    // world. Needed to place a file-loaded body (which sits at its saved
    // position, usually the origin) into a scene without it interpenetrating
    // other bodies.
    void  Translate(const Vec3& m);
    float StitchScale() const { return s_scale; }

    // Precompute one smooth geometric normal per anchor ("check there
    // once"): for every patch corner, accumulate the patch's TRUE analytic
    // normal (EvaluateNormal) at that corner into the shared anchor, then
    // normalize. The result is C0-continuous across every shared edge, so
    // shading by interpolating these per-anchor normals has no facet seams
    // -- and it costs nothing per sample (computed once, reused). Returns
    // a const ref to a vector indexed by anchor id. The result is CACHED
    // and only recomputed when anchors have changed since the last call
    // (tracked by a generation counter bumped on MarkAnchorsDirty), so
    // static frames pay nothing and only the first frame after a
    // deformation pays the recompute.
    const std::vector<Vec3>& ComputeSmoothNormals() const;

    // Make anchor normals a consequence of the surface (refits coefs so
    // geometry is preserved). Fixes anchors whose stored normal contradicts
    // the true surface normal. Returns worst residual (fraction of diag).
    float RederiveNormals();

    // raw 64-byte blobs, exactly what the GPU consumes
    const std::vector<AnchorRecord>&  AnchorBlob()  const { return anchor_blob; }
    const std::vector<SurfaceRecord>& SurfaceBlob() const { return surface_blob; }
    const std::vector<int8_t>&        StitchBlob()  const { return stitch_blob; }
    bool AnyDirty() const {
        for (const Patch& p : patches) if (p.Dirty()) return true;
        return false;
    }
    void ClearDirty() { for (Patch& p : patches) p.ClearDirty(); }

private:
    std::vector<AnchorRecord>  anchor_blob;   // contiguous, GPU-mirrored
    std::vector<SurfaceRecord> surface_blob;  // contiguous, GPU-mirrored
    std::vector<int8_t>        stitch_blob;   // 8 per surface
    std::vector<Patch>         patches;
    AnchorPool                 anchors;
    SeamGraph                  seams;
    Vec3  bbox_min{}, bbox_span{};
    float s_scale = 0.0f;
    bool  anchors_dirty = false;
    uint64_t anchor_gen = 1;                       // bumped on dirty
    mutable std::vector<Vec3> smooth_n_cache;      // cached per-anchor normals
    mutable uint64_t smooth_n_gen = 0;             // gen the cache was built at
    std::unordered_set<uint16_t> split_centers; // merge lineage
    // ROM mode shapes (PSM v2): [mode][n_anchor + n_surf*8]. Populated by
    // ModalDynamics::LoadModes so PSFBody::ApplyMode can drive shape modes
    // without the body owning the .modes file.
    std::vector<std::vector<float>> modes;
};

} // namespace DeliRium
