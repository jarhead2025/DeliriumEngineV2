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

namespace DeliRium {

class PSFBody;
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
    enum class Path { March, Raymarch, SplatClassic, SplatLayered, SplatArea };
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

    // opaque CL/SDL handles (void* so the header stays clean)
    void *cl_ctx = nullptr, *cl_queue = nullptr, *cl_dev = nullptr;
    void *prog = nullptr, *k_splat = nullptr, *k_resolve = nullptr;
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
    void *buf_soff = nullptr, *buf_sss = nullptr;
    uint32_t cap_soff = 0, cap_anchors = 0, cap_surfaces = 0, cap_snorm = 0;
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
};

} // namespace DeliRium
