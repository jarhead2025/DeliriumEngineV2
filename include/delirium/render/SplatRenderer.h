// =============================================================================
// SplatRenderer.h - OpenCL host for kernels/psf_splat.cl.
// One frame: fill fb -> psf_splat (one work-item per surface sample,
// anisotropic per-patch grids) -> resolve_fb -> readback -> SDL texture.
// 64-bit packed atomic path when cl_khr_int64_extended_atomics exists;
// otherwise the kernel's two-pass 32-bit fallback (same image).
// CPU reference: tools/psf_validate.py (pixel-for-pixel comparable).
// =============================================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <CL/cl.h>   // cl_float4 for h_snorm_staging member

namespace DeliRium {

class PSFBody;
class ModalDynamics;
struct Camera;

class SplatRenderer {
public:
    // sdl_renderer: an SDL_Renderer* passed as void* to keep this header
    // dependency-free. kernel_dir: where psf_splat.cl lives.
    bool Initialize(int width, int height, void* sdl_renderer,
                    const std::string& kernel_dir = "kernels");
    void Shutdown();

    // Milestone 3 scope: one body per frame (TODO scene batching).
    // clear=false skips the framebuffer clear so a second body (e.g. a
    // fracture fragment) can be composited into the same frame.
    // final=false defers the buf_fb->buf_img resolve + queue finish to the
    // LAST body of the frame (the caller passes final=true there), so a
    // multi-body frame pays the full-screen resolve and GPU sync ONCE
    // instead of once per body.
    //
    // `model` is the body's LOCAL->WORLD placement (row-major 4x4). The
    // 64-byte records stay in local space (they are FROZEN and quantized
    // over the body's own bbox), so moving an object costs NOTHING on the
    // CPU: no re-quantization, no anchor-blob re-upload. The transform is
    // applied on the GPU while evaluating the surface.
    bool RenderBody(PSFBody& body, const Camera& cam, bool clear = true, bool final = true,
                float depth_bias = 0.0f, const float model[16] = nullptr);
    void Present();                     // blit the resolved image

    // --- M2 scene-level culling -------------------------------------------
    // Frustum culling: tests a body's WORLD-space AABB against the view
    // frustum planes derived from the camera's MVP matrix. Returns true if
    // the body is at least partially visible (should be rendered).
    // Call BEFORE RenderBody to skip fully-outside bodies entirely.
    // The AABB is computed from the body's bbox transformed by `model`.
    static bool BodyVisibleFrustum(const PSFBody& body, const Camera& cam,
                                   const float model[16] = nullptr);

    // Occlusion culling: tests a body's world-space AABB against a coarse
    // depth tile buffer (16x16 tiles covering the viewport). Returns true if
    // the body is NOT fully occluded by nearer geometry already rendered this
    // frame. Call AFTER BodyVisibleFrustum passes and AFTER rendering nearer
    // bodies. The depth buffer is cleared each frame via RenderEmptyFrame()
    // or the first RenderBody(clear=true). The camera is used to project the
    // body AABB center to screen-space tile coordinates.
    // NOTE: coarse approximation — a body flagged visible may still be
    // mostly hidden; the GPU march/raymarch handles per-pixel occlusion.
    bool BodyVisibleOcclusion(const PSFBody& body, const Camera& cam,
                              const float model[16] = nullptr);

    // GPU modal writeback (M2): applies modal amplitudes K to the body's
    // coef[8] coefficients directly on the GPU (no CPU round-trip). The
    // mode_shapes buffer must already be uploaded (see SetModalShapesGPU).
    // Returns true if the GPU path ran; false if disabled (no device / kernel
    // build failed). After a successful call the body's coef are dirty and
    // must be re-uploaded before the next render (MarkSurfaceDirty).
    bool ApplyModesGPU(PSFBody& body, const std::vector<float>& K);

    // Upload modal shapes to GPU device buffer for ApplyModesGPU.
    // mode_shapes layout: [n_modes][n_surfaces][8] floats.
    bool SetModalShapesGPU(const std::vector<float>& mode_shapes,
                           uint32_t n_modes, uint32_t n_surfaces);

    // GPU-FIRST modal writeback setup: uploads REST state ONCE (rest anchors,
    // rest normals, rest coef, full shapes table) to device buffers. Called
    // when a modal body is first bound. Returns true if buffers allocated.
    bool SetupGPUModal(PSFBody& body, ModalDynamics& dyn);

    // GPU-FIRST modal writeback per frame: uploads only K (amplitudes), runs
    // modal_writeback_anchors + modal_writeback_surfaces kernels writing the
    // deformed anchors/surfaces DIRECTLY into the shared render buffers
    // (buf_anchors/buf_surfaces), then runs modal_snorm_pass1/2 writing the
    // smooth normals DIRECTLY into buf_snorm. bake_surfels reads those buffers
    // as usual. No host round-trip, no per-frame upload (~85KB/frame eliminated).
    // Returns true if GPU path ran; false if disabled.
    bool GPUModalWriteback(PSFBody& body, ModalDynamics& dyn, const float* K,
                           uint32_t n_modes);

    // Tear down GPU modal buffers (Shutdown path).
    void ShutdownGPUModal();

    // GPU anchor reflow (shape-preserving): runs the anchor_reflow.cl kernel
    // pipeline on the body's CURRENT GPU buffers (buf_anchors/buf_surfaces/
    // buf_stitch must already be uploaded). Redistributes anchor positions
    // along normals to reduce sampling error WITHOUT growing the pool, then
    // refits coef to reproduce the snapshot target over the new frame.
    // Returns true if the GPU path ran and updated the body; false if the
    // path is disabled (no device / kernel build failed) — caller falls back
    // to PSFBody::ReflowAnchors() CPU path. alpha/max_disp/max_iters match
    // the CPU signature. After a successful call the body's blobs are dirty
    // (anchors + coef changed) and must be re-uploaded before the next render.
    bool ReflowBodyGPU(PSFBody& body, float alpha = 0.01f, float max_disp = 0.01f,
                       int max_iters = 2);

    // GPU adaptive mesh refinement (live subdivision): splits a batch of
    // flagged patches (split_list) entirely on the GPU via psf_refine.cl.
    // The kernel evaluates the 5 hanging anchors on the parent equation,
    // fits each child by LSQ over a 9x9 grid, and appends new records via an
    // atomic free-list. Returns the number of patches actually split (0 if the
    // GPU path is disabled / no device / kernel build failed). The caller
    // rebuilds the seam graph afterwards (seams.Build). ONLY legacy
    // height-field patches (surface_def == SK_LEGACY) can be split on the GPU;
    // parametric families must be handled by the CPU SplitPatch path.
    // NOTE: the body's device buffers must be uploaded BEFORE calling this
    // (the renderer's shared buf_anchors/buf_surfaces/buf_stitch are reused,
    // grown to fit the post-split capacity, then read back into the body's
    // CPU blobs).
    int  SplitBodyGPU(PSFBody& body, const std::vector<uint32_t>& split_list);

    // Clear the framebuffer and resolve it to an empty (background) image.
    // Used when the scene has no bodies: without this the viewport would
    // keep showing the PREVIOUS frame's stale pixels.
    void RenderEmptyFrame();

    // Where the 3D image is drawn IN THE WINDOW. The renderer always
    // renders at its own (width x height) resolution -- which is the
    // VIEWPORT's resolution, not the window's -- so the aspect ratio and
    // the GPU work both match what the user actually sees. Present() then
    // blits that texture into this rectangle, leaving the rest of the
    // window free for editor panels (the Godot-like docked layout).
    //
    // Defaults to the full window (0,0,w,h) when no rect is set.
    void SetPresentRect(int x, int y, int w, int h);
    void ClearPresentRect() { present_x = present_y = 0; present_w = present_h = 0; }

    // ---- THE RENDER PATH (the editor's 1 / 2 / 3 switch) -----------------
    //
    // The engine has three render paths, and the user needs to compare
    // them live from the editor:
    //
    //   March   -- one ray per PIXEL, marched through a 3D surfel grid.
    //              The default; the most stable silhouette.
    //   Raymarch-- one ray per PIXEL, evaluating the surface equation
    //              directly (no grid). Emissive/glow look.
    //   Splat   -- one SURFACE SAMPLE per work-item, each landing on one
    //              pixel via an atomic min. This is the path whose
    //              primitive is the IMPACT: "smaller but more numerous
    //              pixel impacts", and therefore the path the user's
    //              distance-plane averaging method applies to.
    //
    // The Splat path has two compositing modes:
    //   Classic -- one winner per pixel (hard depth resolution).
    //   Layered -- the user's method: the impacts are bucketed into
    //              DISTANCE SHELLS along the view axis, averaged within
    //              each shell, and the shells are stacked back-to-front
    //              with the nearest winning.
    //
    // WHY THIS IS AN API AND NOT JUST AN ENV VAR: the env vars
    // (DELIRIUM_MARCH / DELIRIUM_RAYMARCH / DELIRIUM_SPLAT_LAYERS) are
    // read ONCE (a `static const int` initialiser), so changing them at
    // runtime has no effect -- an editor switch needs a value the
    // renderer reads EVERY FRAME.
    enum class Path { March, Raymarch, SplatClassic, SplatLayered, SplatArea, EquationLayered, TriInfill };
    // Select the render path.
    //
    // The path is a RUNTIME value the renderer reads EVERY FRAME, so the
    // switch is instant -- and it announces itself, because a path switch
    // that is invisible in the log is indistinguishable from a path
    // switch that did not happen (the same failure class as a silently
    // disabled kernel: the scene renders through the fallback and looks
    // plausible). The LAST "render path:" line in the log always names
    // the path currently in force.
    void SetPath(Path p);
    Path GetPath() const { return path_; }
    const char* PathName() const;

    // Save the resolved GPU framebuffer to a PPM (ground-truth capture).
    bool SaveFramePPM(const std::string& path) const;
    // Read the resolved GPU framebuffer into `out` (width*height*4 RGBA,
    // row-major, top row first -- the same layout Present() blits). Used by
    // the 10s recorder. Returns false on a readback failure.
    bool GrabFinalRGBA(std::vector<unsigned char>& out) const;
    void Resize(int width, int height);

    bool HasLongAtomics() const { return has_long_atomics; }
    const std::string& DeviceName() const { return device_name; }
    float LastSamplesPerPixel() const { return last_spp; }

    // --- TriInfill GPU path timing diagnostics (last frame, milliseconds) ---
    // The Profiler panel reads these to show the per-stage breakdown.
    float LastTriSampleMs()   const { return last_tri_sample_ms; }
    float LastTriBuildMs()    const { return last_tri_build_ms; }
    float LastTriRasterMs()   const { return last_tri_raster_ms; }
    float LastTriUploadMs()   const { return last_tri_upload_ms; }
    float LastTriTotalMs()    const { return last_tri_total_ms; }
    uint32_t LastTriSamples() const { return last_tri_samples; }
    uint32_t LastTriTris()    const { return last_tri_tris; }
    int      LastTriUVN()     const { return last_tri_uvn; }
    bool     LastTriLodMode() const { return last_tri_lod_mode; }

private:
    bool  BuildProgram(const std::string& kernel_dir);
    // anisotropic grids: ssu/ssv from projected edge lengths (x1.4 + 2,
    // clamp [2,192]); returns total sample count, fills packed ss + offsets
    uint32_t BuildSampleGrids(PSFBody& body, const float mvp[16]);

    int  width = 0, height = 0;
    // Present destination rectangle in WINDOW pixels. All zeros = full window.
    int  present_x = 0, present_y = 0, present_w = 0, present_h = 0;
    // The active render path. Read EVERY FRAME by RenderBody (the env vars
    // are read once, so they cannot express a runtime switch).
    Path path_ = Path::March;
    // True once SetPath() has been called -- an explicit choice SUPERSEDES
    // the env-var seed, even if the GPU is initialised afterwards.
    bool path_explicit_ = false;
    bool has_long_atomics = false;
    std::string device_name;
    float last_spp = 0;
    // TriInfill GPU path timing diagnostics (last frame, milliseconds).
    float last_tri_sample_ms = 0.f, last_tri_build_ms = 0.f, last_tri_raster_ms = 0.f;
    float last_tri_upload_ms = 0.f, last_tri_total_ms = 0.f;
    uint32_t last_tri_samples = 0, last_tri_tris = 0;
    int      last_tri_uvn = 0;
    bool     last_tri_lod_mode = false;

    // opaque CL/SDL handles (void* so the header stays clean)
    void *cl_ctx = nullptr, *cl_queue = nullptr, *cl_dev = nullptr;
    void *prog = nullptr, *k_splat = nullptr, *k_resolve = nullptr;
    // TriInfill GPU path (the optimized path): psf_sample evaluates PSF math
    // surfaces on a structured su×sv grid per patch, tri_build builds a
    // structured triangle list (2 per cell) with indexed output, tri_raster
    // rasterizes triangles with Gouraud shading + early-z rejection. See
    // kernels/psf_splat.cl (psf_sample, tri_build) and kernels/tri_infill.cl
    // (tri_raster).
    void *k_psf_sample = nullptr, *k_tri_build = nullptr, *k_tri_raster = nullptr;
    // The LAYERED distance-plane splat (the user's method): the impacts
    // are bucketed into distance shells, averaged within each shell, and
    // the shells are stacked back-to-front. See kernels/psf_splat.cl.
    void *k_splat_layered = nullptr, *k_layer_fill_band = nullptr,
         *k_layer_stack_band = nullptr;
    // The AREA-WEIGHTED splat (the pixel-footprint pyramid): each surface
    // sample is distributed over the pixels its footprint covers, weighted
    // by area, and the pixel's colour is the area-weighted mean of
    // everything that touched it. The weights are FIXED-POINT INTEGERS
    // summed with atomic_add and divided with an integer divide, so the
    // result is exact and deterministic. See kernels/psf_splat.cl.
    void *k_splat_area = nullptr, *k_area_resolve = nullptr;
    // Anchor reflow GPU path (shape-preserving): snapshot_error_accumulate
    // captures the current surface before anchors move AND computes per-patch
    // error indicators AND atomically accumulates error into per-anchor buffers;
    // compute_anchor_displacement redistributes anchors along normals using the
    // accumulated totals (O(n_anchors), no patch scan); update_anchor_positions
    // applies displacement; refit_coefficients reproduces the snapshot shape
    // over the new anchor frame (no drift). See kernels/anchor_reflow.cl.
    void *k_reflow_accum = nullptr,
         *k_reflow_disp = nullptr, *k_reflow_update = nullptr,
         *k_reflow_refit = nullptr;
    // GPU AMR (live subdivision) path kernels (psf_refine.cl): split_patches
    // evaluates hanging anchors on the parent equation, fits each child by LSQ
    // over a 9x9 grid, and appends records via an atomic free-list. See
    // kernels/psf_refine.cl for the algorithm and record layouts.
    void *k_split = nullptr;
    // The area accumulators, one slot per pixel:
    //   area_csum -- 3 uints per pixel (the colour sums, fixed point)
    //   area_wsum -- 1 uint per pixel (the accumulated area weight)
    //   area_key  -- 1 ulong per pixel (the nearest depth, as float bits)
    // They are reallocated when the render resolution changes.
    void *buf_area_csum = nullptr, *buf_area_wsum = nullptr,
         *buf_area_key = nullptr;
    uint32_t cap_area_px = 0;   // pixels currently allocated
    // The band buffers are sized nbands * W * H. They are reallocated
    // when the band count or the render resolution changes.
    //
    // Each slot is ONE 64-bit key: (depth_bits << 32) | rgba. The depth
    // and the colour travel together in a single atomic (atomic_min), so
    // "the nearest impact owns this pixel" and "this is its colour" are
    // one indivisible operation. With separate depth and colour slots
    // there was a window in which a preempted writer could overwrite a
    // nearer impact's colour with its own -- the render was
    // NON-DETERMINISTIC (five runs of the same scene -> five different
    // images).
    void *buf_band_key = nullptr;
    // The band's FILL is a ping-pong: the fill reads the source band's
    // values and writes the destination's, so the two sets must be
    // DIFFERENT buffers. An in-place fill (source == destination) is a
    // data race -- work-items read neighbours that other work-items in
    // the same dispatch have already overwritten -- and the result
    // depended on the GPU's scheduling. Measured: three renders of the
    // same scene gave three different md5s with the in-place fill, while
    // the classic path was bit-stable. This is the destination set.
    void *buf_band2_key = nullptr;
    uint32_t cap_band_slots = 0;   // band slots currently allocated
    int      cap_band_count = 0;   // band count the allocation was made for
    void *buf_fb = nullptr, *buf_img = nullptr, *buf_depth32 = nullptr;
    void *buf_tie_cnt = nullptr;   // SPLAT_TIE_DIAG depth-tie counter (1 uint)
    void *buf_anchors = nullptr, *buf_surfaces = nullptr, *buf_stitch = nullptr;
    void *buf_snorm = nullptr;   // smooth per-anchor shading normals
    // Anchor reflow temp buffers (per-body, allocated lazily on first reflow):
    //   buf_reflow_target -- float3[G*G] per patch (snapshot output)
    //   buf_reflow_error  -- float per patch (error indicator output)
    //   buf_reflow_sum_err -- float per anchor (accumulated error total)
    //   buf_reflow_count   -- uint per anchor (patch count total)
    //   buf_reflow_disp   -- float3 per anchor (displacement output)
    //   buf_reflow_coef   -- float[8] per patch (refit_coefficients output)
    void *buf_reflow_target = nullptr, *buf_reflow_error = nullptr,
         *buf_reflow_sum_err = nullptr, *buf_reflow_count = nullptr,
         *buf_reflow_disp = nullptr, *buf_reflow_coef = nullptr;
    uint32_t cap_reflow_patches = 0, cap_reflow_anchors = 0;
    // GPU AMR buffers: buf_amr_anchors/buf_amr_surfaces/buf_amr_stitch hold the
    // body's blobs during a split (READ_WRITE so the kernel can append new
    // records via the atomic free-list). buf_amr_alloc_heads holds the two
    // cursors [anchor_cursor, surface_cursor]; buf_amr_split_list is the input
    // flagged-patch ids. They are allocated lazily on first split and grown
    // to fit the body's post-split capacity.
    void *buf_amr_anchors = nullptr, *buf_amr_surfaces = nullptr,
         *buf_amr_stitch = nullptr, *buf_amr_alloc_heads = nullptr,
         *buf_amr_split_list = nullptr;
    uint32_t cap_amr_anchors = 0, cap_amr_surfaces = 0;
    void *buf_soff = nullptr, *buf_sss = nullptr;
    uint32_t cap_soff = 0, cap_anchors = 0, cap_surfaces = 0, cap_snorm = 0;
    // GPU-FIRST modal writeback buffers (resident on device).
    // buf_modal_rest_anchors/rest_normals/rest_coef hold REST state uploaded ONCE
    // at bind (read-only). buf_modal_shapes holds the full PSM v2 table
    // [n_modes][n_anchor + n_surf*8]; buf_modal_K holds amplitudes uploaded each
    // frame. buf_raw_n is the snorm pass1 accumulator scratch. The writeback
    // kernels read REST + K and write the deformed anchors/surfaces/snorm DIRECTLY
    // into the SHARED render buffers (buf_anchors/buf_surfaces/buf_snorm) that
    // bake_surfels reads — no host round-trip, no per-frame upload (~85KB/frame).
    void *buf_modal_rest_anchors = nullptr, *buf_modal_rest_normals = nullptr,
          *buf_modal_rest_coef = nullptr;
    void *buf_modal_shapes = nullptr, *buf_modal_K = nullptr, *buf_raw_n = nullptr;
    uint32_t cap_modal_anchors = 0, cap_modal_surfaces = 0, cap_modal_modes = 0, cap_modal_shapes_bytes = 0;
    // GPU modal writeback kernels (from psf_deform.cl).
    void *k_modal_writeback_anchors = nullptr, *k_modal_writeback_surfaces = nullptr;
    void *k_modal_snorm_pass1 = nullptr, *k_modal_snorm_pass2 = nullptr;
    // GPU modal writeback enabled flag (set from DELIRIUM_GPU_MODAL env var).
    bool gpu_modal_on = false;
    // GPU modal diagnostic logging flag (set from DELIRIUM_GPU_MODAL_DIAG env var).
    bool gpu_modal_diag = false;
    // Cached modal state for the currently-bound modal body (set by SetupGPUModal).
    cl_float3 modal_bbmin = {{0,0,0}}, modal_bbspan = {{0,0,0}};
    uint32_t modal_n_anchors = 0, modal_n_surfaces = 0, modal_n_modes = 0;
    const PSFBody* modal_body = nullptr;
    // TriInfill GPU path buffers (frame-sized):
    void *buf_verts_grid = nullptr, *buf_norms_grid = nullptr;   // float4/float3 per sample
    void *buf_pos_grid = nullptr;                                // float3 per sample (world-space position)
    void *buf_branch_trace = nullptr;                            // uint per sample: kernel-recorded surface kind
    void *buf_indices = nullptr, *buf_n_tri = nullptr;           // uint[3]/tri + counter
    void *buf_sample_off = nullptr, *buf_patch_uvn = nullptr;    // prefix sums + per-patch UVN
    void *buf_patch_cell_off = nullptr, *buf_patch_surf = nullptr;
    uint32_t cap_tri_samples = 0, cap_tri_tris = 0, cap_tri_patches = 0;
    // CROSS-BODY OWNERSHIP: buf_anchors/buf_surfaces/buf_snorm are SHARED
    // across every body rendered in a frame. The old upload gate
    // (grew || AnchorsDirty) let a smaller/clean body skip its upload and
    // render with the PREVIOUS body's data (the "everything is one color"
    // contamination). Tracking which body's data is actually in the buffers
    // forces a re-upload whenever the body changes, not just when it grows.
    const PSFBody* last_body = nullptr;
    // The record COUNTS that were last uploaded. A pointer comparison alone
    // is not a safe body identity: a destroyed body and a newly allocated
    // one can share the same heap address, and the renderer would then draw
    // the OLD body's geometry for the NEW body (measured: a 4-sheet gearbox
    // then a 2-sheet sphere at the same address -- the sphere rendered with
    // the gearbox's anchors and lost half its silhouette). These counts are
    // part of the identity, so a different-sized body always re-uploads.
    size_t last_anchor_count = 0, last_surface_count = 0;
    void *sdl_rnd = nullptr, *sdl_tex = nullptr;

    // host staging for per-frame sample layout
    std::vector<uint32_t> h_soff, h_sss;
    // persistent host staging for smooth normals. Used so the buf_snorm
    // upload can be NON-BLOCKING (CL_FALSE): the local staging vector in
    // RenderBody would be destroyed before the DMA completes, but a renderer
    // member survives until the next clFinish(q) at the end of the path.
    std::vector<cl_float4> h_snorm_staging;
    uint32_t cap_snorm_staging = 0;

    // --- M2 occlusion culling buffers -------------------------------------
    // Coarse depth tiles (16x16) storing the nearest depth (float) already
    // rendered this frame. buf_occlusion_depth is the device buffer;
    // cap_occlusion_tiles tracks the allocation size.
    void *buf_occlusion_depth = nullptr;
    uint32_t occlusion_tiles_x = 16, occlusion_tiles_y = 16, cap_occlusion_tiles = 0;

    // --- M2 GPU modal writeback buffers -----------------------------------
    // buf_modal_shapes holds mode_shapes [n_modes][n_surfaces][8] on device;
    // buf_modal_K holds the amplitude vector K [n_modes]. k_apply_modes is
    // the apply_modes kernel from psf_deform.cl.
    void *k_apply_modes = nullptr;
};

} // namespace DeliRium
