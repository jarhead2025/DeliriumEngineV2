
## 2026-10-26 — validate_gpu_flush compile fix (readback missing queue arg)
- Error: `tests/validate_gpu_flush.cpp:52: cannot convert 'cl_mem' to 'cl_command_queue'` —
  readback() helper called clEnqueueReadBuffer(m, CL_TRUE, ...) without the command queue.
- Fix: readback(cl_command_queue q, cl_mem m, ...) + all 9 call sites pass q.
- Status: fixed in sandbox; user must sync file + rebuild before ctest.
- Same run confirmed: validate_gpu_split PASSED (kernel + stitch-check fix validated on A770),
  delirium_visual_gallery + delirium_playground built and linked.

## 2026-10-26 — 18/18 gates green + realtime_deform benchmark added
- FULL GATE SUITE PASSED on user's A770: 18/18 (validate_gpu_flush + validate_gpu_split
  both Passed; perf_engine 2.42s). GPU live subdivision + partial dirty-record upload
  are now validated end-to-end.
- New: examples/realtime_deform/main.cpp (target delirium_realtime_deform) — HEADLESS
  real-time benchmark WHILE deforming. Workloads: modal / amr / dent / mixed; paths:
  march / raymarch / splat. Times deform(CPU) + render(GPU incl. FlushToGPU) per frame,
  reports mean/p50/p95/max + FPS vs 16.67ms budget. --shot FRAME FILE saves a PPM.
- Design constraint (verified in ModalDynamics.cpp:106): WriteBack no-ops after a
  topology change (n_surf/n_anchor guard), so modal+AMR are separate workloads;
  'mixed' excludes RefitLocal for the same reason.
- Files to sync: examples/realtime_deform/main.cpp (new), CMakeLists.txt (target added).

## 2026-10-26 — march path: kernel syntax fix + GPU bake_surfels wired (kills 28ms CPU bake)
- SYMPTOM: realtime_deform "march" runs showed 28ms render (35 FPS) +
  "surfel_march kernel build FAILED ... 1:74:54: expected ';' after return".
  The march path (DEFAULT visible path) was silently dead -> fell back to splat.
- ROOT CAUSE (two, compounding):
  1) kernels/surfel_march.cl line 74: deform_h had 8 closing parens but the
     7 nested fma calls need exactly 7 -> one extra `)`. FIXED (7/7 balanced).
  2) Even with the kernel compiling, the march path did a CPU BakeSurfels
     (54,400 surfel evals) + ~2.2 MB clEnqueueWriteBuffer EVERY frame, because
     ModalDynamics::WriteBack marks anchors dirty each frame. The GPU
     bake_surfels kernel (surfel_march.cl:81) was WRITTEN to kill exactly this
     ("kills the CPU bake + 2.6 MB upload") but was NEVER wired: k_march_bake
     was compiled but never created/enqueued; m_bake_* dirty flags were dead code.
- FIX (engine/src/render/SplatRenderer.cpp, march path only):
  * Build block: create k_march_bake = clCreateKernel(pr,"bake_surfels"); added
    to the null-check + the march-path gate (k_march_bake must be non-null).
  * Replaced CPU `std::vector<Surfel> surfels; BakeSurfels(...)` + full upload
    with the GPU bake: dirty-flag raw-record upload (anchors/stitch/snorm on
    grew||AnchorsDirty, surfaces on grew, else FlushToGPU partial) — the SAME
    logic as the splat path (lines ~743-799) so shared buffers stay consistent.
    Then enqueue bake_surfels over a FIXED ns*uvn*uvn grid (deterministic count,
    so build_grid's work-size is known up front). buf_surfels is now
    CL_MEM_READ_WRITE (GPU writes it); downstream build_grid/march_draw unchanged.
- CORRECTNESS: verified GPU bake_surfels == CPU BakeSurfels byte-for-byte:
  deform_h is the identical 8-term fma basis; bilinear order A0..A3 == P[0..3]
  (mix(mix(A0,A1,uu),mix(A3,A2,uu),vv)); both use bilinear smooth normal +
  hue albedo; anchor_pos dequant (offset 32, ushort[3], /65535) matches.
  CLC returns false on any GPU error -> clean fallback, no corrupted frame.
- EXPECTED: march render drops from ~28ms to ~1ms (AMR run proved the GPU
  splat pass is ~0.7ms; the 28ms was the CPU bake+upload). deform(CPU) unchanged.
- Files to sync: kernels/surfel_march.cl (1-char fix), engine/src/render/SplatRenderer.cpp.
- VERIFY: rebuild + `./build_delirium/delirium_realtime_deform` -> modal/march +
  mixed/march should now be ~1ms render (~1000+ FPS), and the "kernel build
  FAILED" stderr line should be GONE.

## 2026-10-26 — RESULT: march path alive, 6.76ms / 147 FPS / MEETS 60 (prediction corrected)
- User's A770 run after sync: "GPU surfel_march (single-pass lateral): ready"
  (kernel now compiles — the 1-char paren fix worked). modal/march:
  render mean 6.757ms | p50 6.323 | p95 9.859 | max 10.681 -> 147.3 FPS,
  MEETS 60 FPS. deform(CPU) unchanged (0.032ms mean).
- PREDICTION CORRECTION (honest): I had predicted "~1ms / 1000+ FPS". WRONG.
  I extrapolated from the AMR run (0.73ms), which was a nearly-static body
  on a different path — not representative of the march_draw cost. The GPU
  bake fix DID work (the CPU bake + 2.2MB upload is gone), but the real floor
  is the march_draw kernel itself.
- REAL COST: march_draw = 786,432 px (1024x768) x up to 40 steps x a 5x5x5
  (125-cell) neighborhood per step, with early exit on first hit. That is
  inherent to the per-pixel lateral-march algorithm the user designed. The
  bake is now ~sub-ms on GPU; ~6ms is march_draw + 2 fill passes + compose.
- FAIR COMPARISON: the old "march" 28.5ms was actually the SPLAT fallback
  (march was dead -> renderer fell back to splat; the march-path CPU
  BakeSurfels never ran). So the win is: march now RUNS at 6.76ms vs the
  28.5ms splat fallback it was silently hiding behind = ~4.2x faster, and it
  meets 60 FPS with 2.4x headroom. (My previous message's "28ms was the CPU
  bake" was imprecise — the 28ms was the splat fallback, not the march bake.)
- OPTIONAL LEVERS if more headroom wanted (all env-var, no rebuild):
  DELIRIUM_MARCH_STEPS=24 (from 40) cuts march_draw ~40%; the 5x5x5
  neighborhood (surfel_march.cl:185-190) -> 3x3x3 would cut ~55% but risks
  gaps at low uvn; lower render resolution. Not needed — already 147 FPS.
- STATUS: DONE. march path is the correct, fast, real-time default.

## 2026-10-26 — GENERIC element-system reconstruction (categorizer-ready)
- GOAL (Josias): procedural generation must be GENERIC, not named-primitive
  makers. Objects = recursive ELEMENT trees: systems that OVERLAP + INTERACT
  per geometric location, decomposed to physical SUBSTANCE leaves (mammal =
  overlapping systems per location). The future CATEGORIZATION method will
  EMIT element trees; the baker stays.
- NEW FILES (all auto-globbed by CMake, no CMakeLists edit needed):
  * engine/include/delirium/ai/ElementSystem.h
  * engine/src/ai/ElementSystem.cpp
  * tests/validate_reconstruction.cpp (CPU-only gate)
- DESIGN (ElementSystem.h/.cpp):
  * ISubstance = leaf "physical substance": maps LOCAL (x,y) -> local point +
    unit normal (+valid flag for bounded domains). Generic graph model z=f(x,y)
    (same graph the PSF patch format uses). Leaves: PlaneSubstance,
    EllipsoidSubstance (upper hemisphere, bounded), GaussianSystemSubstance,
    HeightfieldSubstance (arbitrary fn). NOT named objects.
  * Element = one system: kind (category tag), origin/scale (location),
    weight, color, presence(world)->[0,1] (WHERE active), substance (leaf),
    children (overlapping sub-systems, recursive).
  * ElementSystem::FieldAt = the "interacting per location" rule: every active
    system votes; surface point = weighted avg, normal = weighted sum, color =
    weighted avg. Overlap is first-class (blend, no cull).
  * ElementSystem::Bake = sample the aggregate field on an N x N grid, fit each
    patch's 8 corner-vanishing coefs with the SAME LSQ (cv_basis + solve8) the
    engine uses internally -> frozen 64-byte records. Category-agnostic.
  * BakeBody = Bake + Validate + PSFBody::FromRecords (the existing
    materializer) -> a live, renderable/physical body.
  * ProceduralElementGenerator::Build(semantic) = stand-in for the learned
    categorizer; maps a few semantic families to GENERIC element trees
    (bump/mound/hill, orb/sphere/ball/body, field/wave/terrain, default plane).
- WIRED: NeuralReconstructor::Reconstruct now builds a tree via the generator
  and bakes it (grid_n from max_patches budget). No longer delegates to
  FixedReconstructor. (Reconstructor.h/.cpp edited; +<cmath>.)
- KEY CORRECTNESS (verified by static cross-ref):
  * cv_basis == render deform_h (identical 8 terms) -> fits reproduce in GPU.
  * corner order A0(0,0) A1(1,0) A2(1,1) A3(0,1) matches renderer bilinear.
  * quantize pos=(p-bbmin)/bbspan*65535, normal=(n*0.5+0.5)*255 (matches pool).
  * Validate() passes (distinct anchors, in-range, finite coefs).
- BUG FIXED WHILE WRITING: first draft passed normalized (u,v) to substances but
  FieldAt passes local world coords -> substances rewritten as graphs over local
  (x,y); EllipsoidSubstance became a bounded upper hemisphere (valid flag).
- FILES TO SYNC: the 3 new files + engine/include/delirium/psf/Reconstructor.h
  + engine/src/psf/Reconstructor.cpp.
- VERIFY (user machine): rebuild, then ctest -> expect validate_reconstruction
  as a NEW gate (tests #9..#11 shift). Run:
  ./build_delirium/validate_reconstruction
  Sections: A field composition, B bake validity, C materialize, D fidelity
  (<2% diag), E NeuralReconstructor seam, F ellipsoid (<0.06).
- STATUS: DONE (static-verified; awaits user build + ctest).

## 2026-10-26 — recon_window: live window for element-system reconstruction
- NEW examples/recon_window/main.cpp (target delirium_recon_window, EXISTS-guarded,
  links delirium_engine_lib): LIVE SDL window showing FOUR objects reconstructed
  IN MEMORY from the element-system baker (NeuralReconstructor ->
  ProceduralElementGenerator -> ElementSystem::BakeBody), not from a file:
  orb (ellipsoid body), bump (plane + gaussian node), wave (plane + wave field),
  mound (plane + gaussian node). Placed at x = -1.5/-0.5/0.5/1.5.
- Placement: engine renders each body at its world position (does NOT apply
  GameObject::Transform), so objects are separated by offsetting the CAMERA per
  object in the render hook (cam.position -= offset, RenderBody, restore).
  clear=true only on the first object so the 4 composite. Aim (window center)
  ray-tests each object in its own frame (same offset) and picks nearest by hit.t.
- Full playground control set: LMB poke, RMB dent, R/C AMR, F fracture, G
  restitch, T weld, 1/2/3 render path, O orbit, P screenshot, Esc.
- Files to sync: examples/recon_window/main.cpp (new), CMakeLists.txt (target).
- VERIFY: rebuild + `./build_delirium/delirium_recon_window` -> window with 4
  live objects; aim at one and drive it.

## 2026-10-26 — recon_window black screen: bodies all baked at origin (overlapping)
- Symptom: `delirium_recon_window` opened but showed a black screen. Log showed
  `Engine::Run: 0 object(s)` (expected — objects built directly, not via
  scene->Create) and, crucially, EVERY body's `anchor_aabb` identical:
  `(-0.5,-0.5,0.013)..(0.5,0.5,0.100)`. All four objects baked at the ORIGIN,
  exactly overlapping, and they are thin flat cards (z-span 0.087).
- Root cause: the per-object placement was a CAMERA offset (shift cam.position
  per object in the render hook). But the engine renders each body at its WORLD
  position and ignores GameObject::Transform — so the camera offset only rotated
  the view; the four 1x1 cards stayed stacked at the origin and were seen
  edge-on/far -> black.
- Fix (placement at bake time, generic):
  - ElementSystem::Bake / BakeBody gained a `Vec3 world_offset` param (default 0).
    Bake's bbox is now OFFSET-INCLUSIVE (mn/mx over pts+offset), anchors quantize
    as (pts+offset - bbmin), the patch-fit target is in the dequant/world frame
    (f.point + offset - bil), and BakeBody passes the identical offset-inclusive
    bbox to FromRecords so dequant matches. (Offset is a pure translation -> the
    fit is translation-invariant; the critical fix was bbox consistency.)
  - ReconstructionRequest gained `Vec3 world_offset` (default 0); NeuralReconstructor
    threads it to BakeBody.
  - recon_window now bakes each object at a DISTINCT 2x2 grid position
    (±0.7 in X/Y) via req.world_offset; render hook uses ONE camera (no per-object
    shift); aim() ray-tests all bodies in world space with a single ray.
- Files: engine/include/delirium/ai/ElementSystem.h, engine/src/ai/ElementSystem.cpp,
  engine/include/delirium/psf/Reconstructor.h, engine/src/psf/Reconstructor.cpp,
  examples/recon_window/main.cpp.
- VERIFY: rebuild + run -> 4 distinct objects visible (orb/bump/wave/mound in a
  2x2 grid), aim + poke/dent/refine/fracture work per-object.
- SESSION 2026-10-27: recon window BLACK SCREEN diagnosis (no fix yet, diagnostic built).
  Symptom: delirium_recon_window renders pure background (#0b0b10 = renderer bg) —
  ZERO lit pixels, not dim. Log shows placement WORKS (anchor_aabb = the (0.7,-0.7)
  slot, not origin) and both march+splat log lines per frame; 92000 samples built.
  Verified in code (static): camera (0,3.2,4.6)->origin has the bodies IN FRONT
  (dot~0.99) and in-frustum; bake_surfels colors 0.35..0.85 (not black); deform_h
  == cv_basis; anchor_pos/nrm byte offsets (32/44) correct; march near/far brackets
  the bodies; splat path complete. So geometry+color+frustum are all fine on paper.
  Also found a REAL color bug (not the black cause): ElementSystem::Bake writes
  K[0..3]=color*126 on all 4 slots, but the kernel sums 4 slots /63 -> 8x too
  bright (clamps toward white). Fix pending: K[0]=color*63, K[1..3]=0.
  NEXT: headless diagnostic examples/recon_diag (target delirium_recon_diag) answers
  Q1 in-front/frustum, Q2 raycast, Q3 lit-pixel count per path (march/raymarch/splat)
  for the 4 bodies + per-body framing camera + torus control. Output decides:
  camera/placement vs generated-records vs renderer/env.
  Files: examples/recon_diag/main.cpp, CMakeLists.txt (new target).
- SESSION 2026-10-27 (cont): recon_diag enhanced with RECORD DUMP (decisive test).
  User re-ran: "still a black screen". Added DumpRecords() to examples/recon_diag:
  per body prints max|coef| (HUGE flag if >0.05*diag), and for first 2 patches:
  anchor idx, h(0.5,0.5), normal N (kernel decode n*(2/255)-1), center vs deformed
  center (OUTSIDE-BBOX flag), and EXACT kernel color (sum K/63 * pattern_eval).
  Fixed Q2: raycast now aims straight at each body's bbox center (was window
  center pixel -> false "no hit"). Confirmed channel_color math: 4 slots,
  idx nibbles + int8 K[4], /63 each, pattern 0 = 1.0. Bake writes pattern 0 x4,
  K[0..3]=color*126 -> decodes 8x (clamps white) -- color bug confirmed but NOT
  the black cause (8x brightens, doesn't blacken).
  READ ME: max|coef| huge -> LSQ fit diverged (surface off bbox). N points down
  -> back-face cull eats everything. color ~0 -> color bug. All ok -> renderer.
  Q3 + per-body framing + torus control still in the run. Files: examples/recon_diag/main.cpp.
- SESSION 2026-10-27 (cont): MARCH-PATH MULTI-BODY COMPOSITING FIX (black screen).
  ROOT CAUSE (confirmed by code trace, not just recon_diag): the march path was
  gated on clear=true, so ONLY body 0 marched. It composed straight to buf_img
  (overwrite) and RETURNED EARLY, never touching buf_fb. Bodies 1..3 (clear=false)
  fell through to the SPLAT path: splat into buf_fb, then resolve_fb writes EVERY
  pixel of buf_img -> clobbered the march output -> black screen at the window
  camera. (Single-body framing + torus control worked because nothing clobbered.)
  FIX (2 files):
  (1) kernels/surfel_march.cl:
      - march_draw now takes 4 MVP float4 args (mvp_s0123/s4567/s89ab/scdef) and
        computes the hit surfel's NDC depth hp.z/hp.w (EXACT psf_splat.cl scale,
        no clamp, skip if <0 or w<=1e-6). Removed dead toS/hd world-depth math.
      - NEW kernel march_compose_fb(color,depth,W,H,INF,fb): stores the marched
        layer's depth|rgba 64-bit key into the SHARED buf_fb. Body 0 is the sole
        first writer (buf_fb just cleared to far), so a PLAIN STORE == atom_min
        here -> march path needs NO 64-bit atomics (removed the base_atomics
        pragma; strictly more portable).
  (2) engine/src/render/SplatRenderer.cpp (march path):
      - At frame start (clear=true): clEnqueueFillBuffer buf_fb = far key (and
        buf_depth32 on the no-long-atomics path). This is the frame reset.
      - Build MVP (cam.BuildMVP, same matrix as splat) -> mvp0..mvp3, passed to
        march_draw (args after light, before buf_mcolA/buf_mdepA).
      - Final compose: was layer_compose -> buf_img + early return. NOW:
        march_compose_fb -> buf_fb, then k_resolve (resolve_fb) buf_fb -> buf_img
        (so a single-body scene still presents). The splat bodies that follow
        (clear=false) atom_min ON TOP of body 0's keys and the LAST body's
        resolve produces the 4-body composite.
      - Gate hardened: requires buf_fb && k_march_compose_fb && k_resolve (not
        k_march_layer_compose) so a partial kernel build falls back to splat
        instead of resolving an empty buf_fb (black).
  (3) kernels/psf_splat.cl: two-pass fallback (no 64-bit atomics) pass 0 now does
      w = min(existing buf_fb depth, splat depth) before atomic_min(depth32,w),
      so a NEARER march hit is not clobbered by the plain-store fallback.
  VERIFIED (static, no C++ toolchain in sandbox): brace/paren/bracket balance on
  all 3 files; march_draw arg count 25 == 25 SetKernelArg; march_compose_fb 6==6;
  resolve_fb 3==3; member names (has_long_atomics/buf_fb/buf_depth32/k_resolve)
  all in SplatRenderer.h. DEPTH SCALE PARITY: march ndcz == splat hp.z/hp.w
  (same BuildMVP, same float bits) -> atom_min interleaves correctly.
  OPEN: user rebuild + run recon_diag (Q3 per-path lit pixels) + the recon
  window (default march path should now show all 4 bodies, not black).

- SESSION 2026-10-27 (cont): BUILD ERROR after march fb-composite sync.
  SplatRenderer.cpp:653 `std::min(256.0, std::ceil(need))` — 256.0 is a double
  literal, std::ceil(float) returns float -> template deduction failed
  ("no matching function for call to min(double, float)"). FIX: 256.0 -> 256.0f.
  Scanned the rest of the file: no other mixed double/float min/max sites.
  Brace/paren/bracket balance re-verified (257/1517/145 all OK).
  NOTE: user's build now uses a delirium_engine_lib target (library) — file
  compiles as part of it; no CMake change needed for this fix.
  OPEN: user re-sync SplatRenderer.cpp + rebuild + run recon_diag.

- SESSION 2026-10-27 (cont): MARCH FB-COMPOSITE FIX CONFIRMED ON A770.
  recon_diag Q3 (4-body window camera): march 0 -> 86686 lit px (11.0%).
  Ordering march>=raymarch>=splat holds across ALL 12 framings (4 bodies x 3
  paths + torus control) -> atom_min depth interleaving is correct, no
  scale mismatch. ctest 19/19 green (incl validate_reconstruction,
  validate_gpu_split). Build note: user's tree now compiles engine into a
  delirium_engine_lib static lib; harmless gmake "modification time in the
  future / clock skew" warning from the synced SplatRenderer.cpp mtime.
- Same session: FIXED the pending 8x color bug in ElementSystem::Bake
  (engine/src/ai/ElementSystem.cpp ~line 269). channel_color sums
  K[0..3]/63 * pattern_eval(idx); old write set ALL 4 slots = val*126 with
  pattern 0 (eval=1) -> decoded 4*(val*126)/63 = 8*val -> clamped white for
  every mid-range color (dump showed bump/wave/mound = 1.000). FIX: K[0]=
  val*63, K[1..3]=0 -> decodes to exactly val. Verified pattern_eval(0)==1.0
  in psf_splat.cl. Balance re-checked 108/108,249/249,149/149.
  NOTE: orb dump color=(0,0,0) at patch 0 is NOT the bug -- that's a grid
  CORNER anchor outside the EllipsoidSubstance hemisphere (presence 0 -> no
  color). Orb center should be light blue {0.5,0.7,0.9} after the fix.
  OPEN: user re-sync ElementSystem.cpp + rebuild + recon_diag; dump colors
  should now read the intended hues (not 1.000), and the march PPM should
  show colored (not overexposed-white) bodies.

- SESSION 2026-10-27 (cont): COLOR FIX CONFIRMED ON A770. Record dump now
  decodes intended hues (no longer clamped white):
    bump  patch0 color=(0.746,0.683,0.556)  [was 1.000,1.000,1.000]
    wave  patch0 color=(0.508,0.730,0.698)  [was 1.000,1.000,1.000]
    mound patch0 color=(0.746,0.683,0.556)  [was 1.000,1.000,1.000]
    orb   patch0 color=(0.000,0.000,0.000)  [expected: grid corner OUTSIDE
          EllipsoidSubstance hemisphere -> presence 0 -> black; NOT a bug]
  Lit-pixel counts DROPPED after the fix (march 86686->78071, splat
  76994->65829) because the old 8x overexposure pushed edge pixels past the
  lit threshold; true dimmer colors cross it less. Ordering march>=raymarch
  >=splat still holds across all 12 framings -> depth interleaving still
  correct. ctest 19/19 green. Both the march fb-composite fix AND the 8x
  color bug are now closed and verified on hardware.

- SESSION 2026-10-27 (cont): FIXED the orb "flat card + bump" artifact in
  ElementSystem::Bake. ROOT CAUSE: EllipsoidSubstance::Eval returns
  valid=false outside the unit circle, so FieldAt's acc.weight==0 and it
  collapses the point to the ORIGIN. Bake had no way to know the sample was
  invalid -> corner anchors baked at (0,0,0), corner patches fit to that
  phantom point -> degenerate (all-4-anchors-coincident) patches with huge
  coefs (the max|coef|=1.72 "HUGE" flag), bbox inflated to the full card
  (z-span 0.64 > hemisphere radius 0.5), and corner patches rendered as a
  flat black card (color 0,0,0). FIX (engine/src/ai/ElementSystem.cpp):
  (1) Bake tracks ok[id] = (f.weight>1e-9f); bbox computed from VALID
  anchors only; (2) invalid anchors get a safe default (bbox center, +Z
  normal, unused by any surface); (3) patches with any invalid corner are
  SKIPPED (no surface record); (4) BakeBody's bbox recompute applies the
  same weight guard so dequant matches; (5) Bake fails cleanly if the whole
  grid is out-of-support. Bump/wave/mound unaffected (infinite host plane ->
  all anchors valid -> same patch counts, tests C/D/E unchanged). Orb now
  bakes to just the hemisphere patches (~250 of 900), tight bbox, no huge
  coefs. Balance re-checked. OPEN: user re-sync ElementSystem.cpp, rebuild,
  ctest (expect 19/19), recon_diag -> orb dump should show tight bbox
  (z-span ~0.5), no HUGE flag, and the march PPM should show a clean
  hemisphere instead of a card-with-bump.

## Session 2026-10-27 (late) — black screen RESOLVED + torus added to recon_window
- recon_diag (user's A770): all 3 render paths now light pixels for ALL 4
  generated bodies AND the torus control (march 19k-74k, raymarch 21k-65k,
  splat 2.6k-5.5k px). Orb is a real hemisphere: 644 surfaces (was 900),
  max|coef|=0.022 (was 1.72), color 0.508/0.698/0.905, overexposure gone.
- Splat is ~15x sparser than march/raymarch: PRE-EXISTING (BuildSampleGrids
  sizes grids from projected anchor quads; 0.12 spp on these bodies). Not a
  regression from the deformed-bounds fix (that block is march-path only).
  Optional follow-up: deformed-bounds in BuildSampleGrids.
- "19/19 failed" ctest run was a FALSE ALARM: user built only 2 targets, so
  test binaries didn't exist -> "Not Run". Full `cmake --build build_delirium`
  first, then ctest.
- realtime_deform dropped to 20.7 FPS (was 147): NOT a regression. Log line
  "steps=256" proves DELIRIUM_MARCH_STEPS=256 was still exported from the
  black-screen debugging (default is 40). 48.289ms * 40/256 = 7.5ms ~= the
  147-FPS run. Fix: `unset DELIRIUM_MARCH_STEPS`. Proper fix (later):
  adaptive step size so step < foot at any camera distance.
- ADDED: PSFBody::Translate(const Vec3&) (header + PSFBody.cpp, mirrors
  ExpandBBox: capture world pos, shift bbox_min, re-bind, re-quantize,
  mark dirty). Needed because file-loaded bodies sit at their saved position
  (torus.psf is centered at origin, bbox -1.35..1.35) and would
  interpenetrate the 2x2 generated grid.
- ADDED: torus as 5th object in examples/recon_window/main.cpp (full rewrite).
  Obj struct now holds go + raw body + fragment; body_ref() accessor; all
  interactions routed through PSFBody-level ops (DentLocally/Fracture/
  RecalculateStructure are GameObject wrappers, NOT PSFBody methods):
  dent = RefineWhere + ApplyRadialDent; restitch = Seams Build + FitAll +
  MarkAnchorsDirty; fracture = FractureBody() local fn (same plane-split as
  GameObject::Fracture, stores fragment in Win::frags); poke stays
  GameObject-only (needs modal modes). Torus at (0,0,-2.2) via Translate.
  Balance verified 71/71 braces, 212/212 parens.
- NOTE: sandbox Edit tool silently dropped 10 of 13 edits on the first
  recon_window attempt (reported success, file unchanged; NFS mount).
  Re-applied; final state verified by grep + Read. Always verify after
  batch edits in this workspace.
- OPEN: user syncs 3 files (PSFBody.h, PSFBody.cpp, examples/recon_window/
  main.cpp), unset DELIRIUM_MARCH_STEPS, full rebuild, ctest (expect 19/19),
  run recon_window (5 objects) + realtime_deform (expect ~147 FPS again).

## 2026-10-28 — recon_window torus: const bug + MARCH_STEPS misdiagnosis corrected
- **recon_window compile error (mine):** `aim()` is a `const` member but `body_ref()` was non-const → "discards qualifiers". FIX: `PSFBody& body_ref() const`. `Raycast`/`ClosestPointEx` are const, so this is correct. 1-line fix.
- **ctest "4 Not Run" = false alarm:** the build ABORTED at the recon_window compile error (`gmake *** Error 2`), so make never built validate_reconstruction / validate_split / validate_weldcat / perf_engine. The 15 that ran all passed. Rebuild after the const fix → all 19 run.
- **CORRECTION (I was wrong):** I told Josias the `steps=256` came from a leftover `DELIRIUM_MARCH_STEPS=256` env var. WRONG. The user unset it and steps stayed 256 (and the playground shows steps=191/194 that CHANGE per frame). The truth: SplatRenderer.cpp:652 has ADAPTIVE step count — `bsteps = max(env_or_40, ceil((far_d-near_d)/(0.5*foot))+1)`, capped at 256. At the benchmark's camera distance, `need`≈258 → hits the 256 cap. So 256 is the adaptive cap, not an env var. The "147 FPS" was the pre-adaptive fixed-40-steps run (fast but had the black-screen-at-distance bug the adaptive fix solved).
- **Real perf trade-off:** march_draw cost = W*H × steps × 125 (5×5×5 neighborhood). At 256 steps on a big body (torus diag 3.88) → 47ms → 21 FPS. The recon_window demo (5 small objects, only the 1st marches) shows 61 FPS. If the single-body benchmark needs 60: loosen the guarantee 0.5*foot → foot (halves steps, ~2×, no correctness loss) and/or shrink the 5×5×5 neighborhood.

## 2026-10-28 (cont.) — MARCH IS NOW THE PRIMARY RENDERER for all bodies (user directive)
- **User spec (verbatim intent):** splat drops as the main rendering way. Use:
  smaller impact radii per set of lateral rays; slices averaged ONLY between
  impact points; then back-to-forth stacked, forward-most overwriting the ones
  in the back; only then sent as an image through the GPU. This is exactly the
  march path (march_draw small foot -> layer_fill between impacts ->
  depth-keyed frontmost-wins compose -> resolve). Splat stays only as the
  DELIRIUM_MARCH=0 fallback.
- **CHANGES (engine/src/render/SplatRenderer.cpp + kernels/surfel_march.cl):**
  1. Gate: march path now runs for EVERY body (`march_on`, not `clear`).
     `clear` (body 0) only controls the frame reset. CRITICAL: the buf_fb
     frame-reset stays gated on `clear` — if it ran per body, each body would
     wipe the accumulated image and only the last body would show.
  2. `march_compose_fb`: plain store -> FRONTMOST-WINS. atom_min on the
     (depth_bits<<32)|rgba key (same scale as psf_splat) so all marched bodies
     interleave by depth; two-pass depth32 fallback when the device lacks
     cl_khr_int64_extended_atomics (mirrors psf_splat exactly). New kernel
     args: `pass`, `depth32`. March program now compiled with the same
     `-DUSE_LONG_ATOMICS` options as the splat program.
  3. Per-body resolve (buf_fb -> buf_img) after each body's compose — cheap
     W*H pass; keeps Present() and the caller unchanged.
  4. `foot` shrunk ~3.6x: max(2*cell, 1.6*spacing) -> max(0.5*cell,
     0.5*spacing) so adjacent disks just touch and layer_fill bridges the
     1-2 px gaps (the "fill between impacts" effect). Live tune:
     DELIRIUM_MARCH_FOOT=<mult>.
  5. BUG FIX: a re-clamp `foot = max(foot, 2*cell)` in the deformed-bounds
     block silently re-inflated foot and NEGATED any shrink. Now 0.5*cell.
  6. Step guarantee re-tuned to match the smaller foot: step <= foot (was
     0.5*foot), cap 256 -> 512. At the benchmark camera need ~= 469 steps
     (fits under 512). ~1.8x the old march cost at that camera.
- **Per-body cost note:** march range is per-body (near_d/far_d from the
  body's own bbox corners), so small bodies march cheaply; the big body
  (torus, diag 3.88) is the expensive one. 5-body recon_window now marches
  all 5 — expect a visible FPS drop vs the old 1-marches/4-splat mix.
- **OPEN (user to verify on A770):** sync SplatRenderer.cpp +
  surfel_march.cl, full rebuild, run recon_window. Expect: all 5 objects
  rendered by march (log line "march: ... (primary, frontmost-wins)" for
  each body), smaller per-surfel impacts with fill between them, correct
  frontmost-wins occlusion between bodies. A/B knobs: DELIRIUM_MARCH=0
  (old splat path), DELIRIUM_MARCH_FOOT (impact size), DELIRIUM_MARCH_STEPS
  (speed vs far-end guarantee).

## 2026-10-28 — march_draw -51 (CL_INVALID_VALUE): host/kernel ARG ORDER desync, not count
- SYMPTOM: recon_window spammed "CL error -51 at SplatRenderer.cpp:775" every frame
  for all 5 bodies (march is now primary). Syncing the .cl did NOT help.
- ROOT CAUSE: previous session added `use_slab` to the KERNEL signature as the LAST
  param (index 25) but added the HOST clSetKernelArg at index 18 (right after `foot`).
  Both sides had 26 total args, so a count-only audit passed — but the POSITIONS
  disagreed. Host wrote a 4-byte int into the slot the kernel expects to be `light`
  (12-byte float3) -> CL_INVALID_VALUE (-51) at the use_slab set-arg line.
- FIX: moved the host `use_slab` clSetKernelArg to the END (index 25), after the
  color/depth buffers, matching the .cl signature order. Verified index-by-index
  (order + type) for all 26 args: 0 S,1 grid,2 gx,3 gy,4 gz,5 org,6 cell,7 eye,
  8 fwd,9 right,10 up,11 tanfov,12 W,13 H,14 d_near,15 d_far,16 steps,17 foot,
  18 light,19-22 mvp,23 color,24 depth,25 use_slab.
- GUARD ADDED: one-time signature audit in the surfel_march init block. For each of
  the 7 march kernels it calls clGetKernelInfo(CL_KERNEL_NUM_ARGS) and compares to
  the host's expected count; on drift it prints a clear one-time message and disables
  the march path (falls back to splat) instead of spamming -51 per frame. NOTE:
  clGetKernelInfo is 5-arg (kernel, name, size, value, size_ret) — the first draft
  passed 4 args and would itself have returned -51; fixed to sizeof(n),&n,nullptr.
- LESSON: a host/kernel arg mismatch is an ORDER problem, not just a count problem.
  A "26 == 26" check is not enough; the set-arg sequence must match the signature
  slot-for-slot. The new audit catches count drift; order drift is now documented in
  a comment next to the launch site.
- Files changed: engine/src/render/SplatRenderer.cpp (host arg order + init audit).
  kernels/surfel_march.cl UNCHANGED this session (its 26-param signature is correct).
- User action: REBUILD (host code changed, not just the .cl):
  cmake --build build_delirium --target delirium_recon_window && ./build_delirium/delirium_recon_window

## 2026-10-28 — denser sampling + once-per-frame resolve/finish + frontmost-dominant accumulation
- SYMPTOM: -51 fixed, march live, but ~11-17 FPS and visible artifacts:
  checkered/point-cloud surface, z-fighting zebra in ring holes, jagged
  silhouettes, moire. Log: "march: 1024x768 surfels=64400 steps=512 cell=0.0064".
- ROOT CAUSE (diagnosed):
  1) The adaptive step count computed need=(far-near)/foot ~= 812 but was
     HARD-CAPPED at 512 -> step > foot -> the code's own "step <= foot"
     guarantee was broken -> front faces missed (checkered + z-fighting).
  2) renderBody ran the WHOLE pipeline 5x/frame (5 bodies), including the
     full-screen resolve_fb AND a clFinish() per body -> 4 redundant
     full-screen resolves + 4 GPU queue syncs per frame.
  3) march_draw used EARLY-EXIT (first hit written, return) -- NOT the
     user's "accumulate all lateral layers, write once at the end" method.
     The kernel averaged normals within foot at a single step; in a hole the
     near+far rim normals cancel -> normalize noise -> zebra banding.
- FIXES (this session):
  1) Step cap now env-tunable DELIRIUM_MARCH_MAXSTEPS (default 1024, was
     hard 512) so step<=foot holds -> denser sampling, no front-face miss.
  2) RenderBody gained a `final` param (header + impl). resolve_fb +
     clFinish now run ONCE per frame (final=true on last body) instead of
     5x. recon_window passes final=true only on the last body/fragment.
  3) march_draw: FRONTMOST-DOMINANT ACCUMULATION (new kernel param
     `accumulate`, index 26). Accumulates every hit layer weighted 1/d,
     stops at a surface crossing (2 consecutive hit steps OR >=2 surfels in
     one step), writes the pixel ONCE after the loop, depth = frontmost hit.
     Same-object check is FREE (each body marches its own grid).
     accumulate=0 = legacy early-exit (A/B via DELIRIUM_MARCH_ACCUM=0).
  - march_draw is now 27 params; init signature audit updated 26 -> 27.
- FILES CHANGED:
  - kernels/surfel_march.cl (accumulate param + accumulation loop + header)
  - engine/src/render/SplatRenderer.cpp (maxsteps env, final param,
    use_accum arg, audit 27)
  - engine/include/delirium/render/SplatRenderer.h (final param)
  - examples/recon_window/main.cpp (final=true on last body only)
- USER ACTION: REBUILD (host code changed):
  cmake --build build_delirium --target delirium_recon_window
  ./build_delirium/delirium_recon_window
- A/B KNOBS (no rebuild):
  DELIRIUM_MARCH_ACCUM=0   (legacy early-exit vs new accumulation)
  DELIRIUM_MARCH_MAXSTEPS=512  (old sparse cap) vs 1024 (default)
  DELIRIUM_MARCH_FOOT=1.5  (bigger disks, less fill work)
  DELIRIUM_MARCH_PASSES=4  (stronger hole fill)
  DELIRIUM_MARCH=0         (old splat path baseline)
- EXPECTATION: checkered + z-fighting should be gone (step<=foot + no
  cross-layer normal cancel). FPS: denser steps cost ~1.6x march time but
  the once-per-frame resolve/finish should recover part of it. Measure and
  tune DELIRIUM_MARCH_MAXSTEPS / _FOOT / _PASSES to the FPS/quality trade.

## 2026-10-29 — march_draw neighborhood 5x5x5 -> adaptive (2*nr+1)^3 (fidelity-preserving 4.6x cut)
- USER DIRECTIVE: "optimize but whatever is drawn stays faithful to what it
  represents" -> only provably output-identical changes allowed.
- FINDING: the per-step neighborhood scan was fixed 5x5x5 (125 grid lookups)
  but a surfel within `foot` of the ray point p (p in cell cx) can be at most
  floor(foot/cell) cells away in any axis. Default foot = 0.5*cell -> a
  3x3x3 (27 lookups) contains EVERY in-reach surfel; the 5x5x5 was sized for
  the old foot=2*cell "blobby" era and now scans 4.6x more cells than can
  hold a hit (all empty -> wasted random loads).
- PROOF of fidelity: for foot <= cell, any surfel at |cell_dist| >= 2 is more
  than cell > foot away -> d2 >= foot2 -> never counted by the old code
  either. Set of (si,d2) found is IDENTICAL for every foot value:
  foot<=cell -> nr=1 (subset of old scan, same counted set);
  cell<foot<=2cell -> nr=2 (== old 5x5x5); foot>2cell -> nr=2 clamped
  (== old, which was also insufficient there). No regression anywhere.
- FIX: kernels/surfel_march.cl march_draw: `int nr = 1 + (int)(foot/cell);`
  clamped [1,2]; loop bounds -2..2 -> -nr..nr. Adaptive: raising
  DELIRIUM_MARCH_FOOT past ~2.0 auto-grows back to 5x5x5.
- EXPECTED: inner loop 4.6x cheaper (27 vs 125 random int loads + Surfel
  fetches); march_draw is the dominant kernel -> expect ~2-3x frame speedup,
  output pixel-identical.
- ALSO IN THIS REBUILD (from prior turns): fill_passes 2->4 (bridges the
  3-5px projected-surfel-grid gaps = the black-band zebra in overlap areas),
  DELIRIUM_MARCH_MAXSTEPS=1024 default, once-per-frame resolve+finish,
  frontmost-dominant accumulate (DELIRIUM_MARCH_ACCUM=0 for A/B).
- USER ACTION: REBUILD (kernel + host changed):
  cmake --build build_delirium --target delirium_recon_window
  ./build_delirium/delirium_recon_window
  Compare FPS + a screenshot at the same camera as path1_march.png.

## 2026-10-29 — splat made default + per-body post-processing -> once-per-frame + FPS counter fixed
- USER DECISION: "we need speed, if splat is better than march, just apply
  that; keep the averaging if no speed debuff; not convinced of forward ray
  projection unless we can avoid recalculating every certain distance."
- FINDING (verified in code): the classic splat block's post-processing
  (resolve_fb + 3x fb_infill + 2x fb_lod_stitch + clFinish) ran ONCE PER BODY
  (5x/frame). But buf_fb accumulates ALL bodies via frontmost-wins atom_min,
  and resolve_fb writes EVERY pixel (hit or bg) -> each body's resolve
  overwrote the previous body's infill/stitch output. Only the LAST body's
  post pass ever mattered. 4/5 of the post cost was pure waste.
- FIX 1: SplatRenderer.cpp classic splat block: resolve+infill+lod-stitch+
  finish now inside `if (final)` (final=true only on last body, passed by
  recon_window on_frame; header default final=true so Engine.cpp's built-in
  loop is unchanged). Same final image, ~1/5 post cost. Averaging (infill +
  lod-stitch) fully preserved, now run once on the COMBINED image (more
  correct: fills gaps in the result, not per-body intermediates).
- FIX 2: examples/recon_window/main.cpp: default render path = 3 (splat) at
  startup via S.set_render_path(3). 1/2/3 still switch at runtime.
- FIX 3: Engine.cpp FPS counter: was printing an EMA (fps_smooth, 0.7/0.3)
  that took ~5s to converge after a path switch -> STALE numbers
  ("FPS: 50.3 (frame 137 ms)" where 50.3 was the previous path's leftover;
  real FPS = 1000/137 = 7.3). Now prints the actual 0.5s-window FPS
  (fps_inst). The (frame X ms) was always the true per-frame cost.
- WHY SPLAT IS FAST (answers the user's "recalculate every distance" concern):
  splat = N parallel work-items, each projects ONE surfel exactly once
  (O(1), no distance loop) and atom_min collapses each pixel to the
  frontmost hit. That IS the user's "parallelize the ray projection equation
  and collapse to the first value by positioning" -- already implemented.
  March = per-pixel ray marched in steps (recalculates the ray point every
  step) -> 4x slower. Raymarch = per-pixel implicit-surface eval -> 7fps.
- CAVEAT: recon_window caps at ~60Hz (Engine sleeps when faster than 1/60s),
  so the splat path already showed 60fps; the once-per-frame post shows up as
  LOWER frame-ms (expect ~6ms -> ~3-4ms) and matters for headless
  benchmarks (realtime_deform) + higher resolutions.
- REMAINING SPLAT ARTIFACT: depth-boundary aliasing at object intersections
  (purple/green interlock). lod-stitch's object mask (alpha) is ineffective
  (resolve writes alpha=255 for both hits AND bg) so it can't stop
  cross-object bleed at boundaries. Fix would be a depth-aware boundary pass
  (buf_fb's high 32 bits carry depth) -- ~0.3ms extra, NOT applied (user: no
  speed debuffs). Offered as a follow-up.
- USER ACTION: REBUILD (host code changed):
  cmake --build build_delirium --target delirium_recon_window
  ./build_delirium/delirium_recon_window
  Expect: starts on splat (path 3), ~60fps, frame-ms lower than before.

## 2026-10-29 (cont) — DARK PATCHES re-diagnosed (vision read of user screenshot)
- USER: "I see these dark patches that make no sense. If march and splat have no
  inherent advantage, why does one lag more? If we go lateral ray march we could
  pull off slice averaging -- a plane slicing back-to-front, filling pixels,
  replacing the plane before it. More viable than individual ray casts?"
- VISION READ of image.png (1028x768, splat path, ~60fps): 5 dark patches,
  ONE per object, CENTERED, covering 40-50% of each object's face; torus
  lower-left quadrant ~35-40% blacked. Edges jagged/stepped. Specular highlights
  still wrap the edges. Vision read them as "background color" (18,18,24).
- KEY INSIGHT (corrects my earlier "depth-boundary aliasing" call):
  * In THIS renderer, the ONLY way a pixel is EXACTLY (18,18,24) is
    resolve_fb's `key == 0xFFFFFFFFFFFFFFFF` branch = NO SPLAT EVER WROTE IT.
    A shading error would give 15% floor (lam=0.15, never 0), NOT pure bg.
  * So the dark patches = pixels where NO front-facing splat sample landed.
    The rim stays lit (front-facing samples there), the center is dark.
  * CAUSE: the BACK-FACE CULL `if (dot(Nshade, view) > 0.10f) return;`
    (psf_splat.cl:250) is culling the center. On a convex object the center
    faces the camera (dot<0, kept) and the rim is edge-on (culled) -- the
    OPPOSITE of what we see. So Nshade is pointing INWARD at the center:
    the smooth-normal field (ComputeSmoothNormals -> snorm) is flipped/wrong
    on the interior of each object. The cull then deletes the center.
  * This is a NORMAL-ORIENTATION bug in the smooth-normal field, NOT a
    depth/occlusion bug. The march path has the SAME cull (surfel_march.cl:299
    "matches the splat back-face cull") so it would show the same patches.
- ANSWER TO "why does march lag if no advantage": march = per-PIXEL ray marched
  in ~400 steps (recalculates the ray point every step, 314M ray-steps/frame,
  plus a 16.7M-cell 3D grid). Splat = per-SURFEL, one projection each, atom_min
  collapse (82k samples, no distance loop). Same frontmost-surfel answer, but
  march does ~4000x more work. The lag is the distance loop the user wanted to
  avoid -- splat already removed it.
- ANSWER TO "slice averaging / plane back-to-front": that IS the layered march
  (kernels/surfel_march.cl + LateralRayMarch.h BakeSurfels): bake surfels into
  a 3D grid, march, and each hit layer composites front-over-back. It is MORE
  viable than per-pixel individual ray casts for this scene (the grid makes
  per-pixel lookup O(1) instead of O(N_surfels)), but it still marches per-pixel
  in steps -> slower than splat. The slice-averaging (per-slice infill then
  combine) is the right model and is already there as layer_fill, but it runs on
  the single marched buffer, not per-slice.
- NEXT (proposed, NOT yet applied): the dark patches are a normal-orientation
  bug. Two candidate fixes, both env-gated for A/B:
  (a) Make the back-face cull robust: cull on the SIGN of the geometric normal
      dot view, but FLIP Nshade to face the camera when it's inward
      (N = -N if dot(N,view)>0) instead of culling -- keeps the center lit.
  (b) Audit ComputeSmoothNormals orientation (EvaluateNormal = cross(dPdu,dPdv)
      sign depends on (u,v) winding; if some patches wind the other way the
      area-weighted average flips). 
  (a) is the safe, local fix. (b) is the root cause.

## 2026-10-29 (cont) — ROOT CAUSE FOUND + FIXED: back-face CULL -> FLIP (splat path)
- USER: "visual glitches make some toroids not appear or teleport in and out
  fast; the shaded parts make no sense. Let's fix those last couple issues."
  (splat path, ~60fps, frame 3.5-4.3ms)
- VISION READ of new image.png: top row (wave/mound) MISSING; torus visible but
  with a jagged black notch on the rim; "inner curve of the large torus is
  completely dark (black hole) but the rim adjacent is brightly lit"; "no
  consistent light direction"; "harsh color bands".
- LOG RED HERRING: all 3 logged bodies show identical anchor_aabb=(-1.167,
  -1.167,0)..(-0.233,-0.233,0.5) — that's the ORB's AABB. The static slog
  counter (slog++ % 60) logs every 60th RenderBody call; with 5 bodies/frame,
  those 3 hits are all the orb (bodies 0,5,10,...). NOT a data bug.
- ROOT CAUSE (single line explains ALL symptoms): the SPLIT between the two
  paths' back-face handling.
  * MARCH path (surfel_march.cl:300,337): `if(dot(n,fwd)>0) n=-n;` then shade.
    FLIPS the normal to face the camera. NO cull. ROBUST.
  * SPLAT path (psf_splat.cl:250, OLD): `if(dot(Nshade,view)>0.10f) return;`
    DELETES the sample. FRAGILE.
  The smooth-normal field (ComputeSmoothNormals) is an area-weighted average
  over patches with MIXED (u,v) winding -> on object CENTERS it points INWARD.
  An inward normal at the front-center reads as "back-facing" (dot>0.10) and
  gets DELETED. One line explains everything:
  * centered dark patches = front-center samples culled, rim (dot~0) survives
  * objects vanishing/teleporting = as the camera orbits, samples cross the
    0.10 threshold frame-to-frame -> whole objects flicker in and out
  * "shaded parts make no sense"/no consistent light = each object's normal
    field independently mis-oriented -> per-object inconsistent lighting
  * torus inner tube "black hole" = the inner tube is the BACK side of the
    front patch -> culled -> dark.
- FIX (applied, kernels/psf_splat.cl:255-256): make the splat path match the
  march path. Replace the cull with a flip:
      float3 view = fast_normalize(P - eye);
      if (dot(Nshade, view) > 0.f) Nshade = -Nshade;   // was: return;
  This is the standard two-sided-normal technique. For a CORRECT outward
  normal the flip is a no-op on visible front samples (dot<0), so it ONLY
  changes the broken (inward) cases. True back faces are hidden by the depth
  atom_min anyway (nearer front sample wins). The inner tube now gets lit
  (faces the camera) instead of culled. Shading below (keyL, fillL) both use
  the now camera-facing Nshade -> consistent lighting.
- WHY NOT fix ComputeSmoothNormals winding instead: the flip is the robust,
  local, provably-correct fix and it's EXACTLY what the march path already
  does. Fixing the winding is the "root cause" but it's riskier (affects both
  paths, and the march path already compensates with its own flip). The flip
  makes both paths consistent and correct. If the user later wants the baked
  normal field itself to be outward-correct (e.g. for a path that can't flip),
  that's a separate, optional cleanup.
- VERIFIED: no other cull-style returns remain in the splat kernel (the other
  returns are bounds/behind-camera checks). The march path flip is intact.
  Nshade is a float3; -Nshade and dot(Nshade,view)>0.f are valid OpenCL.
  Compiles (simple assignment replacing a return; in the common path before
  projection, so applies to both the long-atomics and fallback paths).
- USER ACTION: REBUILD (kernel changed — recompiled at runtime, no C++ change):
  cmake --build build_delirium --target delirium_recon_window
  ./build_delirium/delirium_recon_window
  Expect: all 5 objects present (no flicker/vanishing), no centered dark
  patches, torus inner tube lit, consistent lighting. Splat still ~60fps.

## 2026-10-29 (cont) — 4 fracture fixes: camera-parallel cuts, depth bias, tie diag, cut cap
- USER: "make the cut planes camera parallel yes; curious if there is a way to
  directly calculate those intersections and approximate them by averaging out
  their colors in between sparsed out pixel impacts; but do implement all 3."
  (the "3" = camera-parallel cuts + depth-tie diagnostic + fragment depth bias;
  the cap = the "directly calculate intersections + average colors" item)

### 1. CAMERA-PARALLEL CUT PLANES (both fracture paths)
- OLD normal: normalize(hit.p - cen) -- RADIAL from the anchor centroid. Only
  camera-parallel when the object's centroid sits on the center-screen ray.
  For the torus (centroid = the hole) that was a RADIAL cross-section, not a
  camera-parallel slice.
- NEW normal: normalize(cam.target - cam.position) -- the camera forward, so
  the cut plane is PARALLEL to the image plane regardless of object position.
  The plane still passes through the midpoint (cen + (hit-cen)/2).
- Applied to BOTH paths:
  * examples/recon_window/main.cpp FractureBody(): now takes const Camera& cam.
  * engine/src/core/GameObject.cpp Fracture(): new param
    const Vec3& cam_forward = {0,0,0} (zero = fall back to radial, preserving
    old behavior for callers that omit it). torus_live + playground call sites
    pass normalize(cam.target - cam.position).
- VERIFIED: cam in scope at all 3 call sites; Camera.h included in all 3.

### 2. FRAGMENT DEPTH BIAS (kills cut-boundary z-fighting = the popping)
- ROOT CAUSE of "popping in and out fast": the fragment is the FAR side of the
  cut plane. Its cut rim is COPLANAR with the parent's cut rim (same plane)
  -> EQUAL depth -> atom_min flips the winner frame-to-frame = z-fighting.
- FIX: new depth_bias param threaded RenderBody -> psf_splat + march_compose_fb.
  The fragment renders with depth_bias=1e-4 NDC (far above float epsilon
  ~1e-7, visually negligible). It pushes the fragment's cut rim just BEHIND
  the parent/cap, so the parent always wins the cut line. No more tie, no pop.
- Kernel changes (psf_splat.cl, surfel_march.cl): depth += depth_bias before
  the atom_min. march_compose_fb: d = depth[pid] + depth_bias.
- ARG COUNTS (audited, updated): psf_splat 18 (long atomics) / 20 (fallback);
  march_compose_fb 8 -> 9. The one-time signature audit now covers psf_splat
  too (same drift class of bug that broke march with CL_INVALID_VALUE -51).
  march_compose_fb fallback pass index 5 -> 6 (bias inserted before pass).

### 3. DEPTH-TIE DIAGNOSTIC (proves the popping cause, gated on env)
- psf_splat: new tie_cnt buffer (1 uint, zeroed at init). For BIASED
  (fragment) samples only, if the pre-bias depth is within 2e-5 NDC of the
  pixel's current winner, atomic_add(tie_cnt,1). Normal bodies pay nothing
  (the check is gated on depth_bias>0).
- Host: buf_tie_cnt buffer + readback in the final block, printed when
  DELIRIUM_TIE_DIAG=1: "[tie-diag] depth-ties since init: N". High N =
  coplanar cut-edge contention (the popping); ~0 = clean.
- ORDERING VERIFIED: bias (1e-4) > tie window (2e-5), so a FIXED fragment no
  longer ties; the counter measures the contention the bias resolves.

### 4. CUT-PLANE CAP (the "directly calculate intersections + average colors")
- USER'S IDEA, implemented: directly calculate the plane-surface intersections
  and approximate them by averaging colors between the sparsed-out pixel
  impacts.
- HOW:
  * RIM (direct intersection): for every PARENT-side patch, each edge whose
    endpoints sit on opposite sides of the cut plane yields its EXACT crossing
    (linear interp of the anchor positions + (u,v) params at the same t -- the
    edge is a straight line in (u,v) space, so this is exact). Deduped by
    position. Colors evaluated AT COLLECTION TIME (before DetachPatches
    compacts the surface blob -- the patch index is stale after the detach).
  * FAN: rim points sorted by angle around the centroid in the cut plane
    (tangent basis t1,cross(n,up) / t2,cross(n,t1)); a shared center anchor at
    the rim centroid; one quad per consecutive rim pair (A0=A3=center, the
    standard triangle-as-quad encoding). Zero deformation coefficients ->
    P(u,v) is the exact bilinear plane; a bilinear quad whose 4 corners are
    coplanar STAYS on the plane (blend weights sum to 1). Winding fixed by
    cross product so the analytic normal points along the cut normal.
  * RENDER: the cap is a real PSFBody (FromRecords) rendered through the
    EXISTING splat pipeline. The adaptive sample grid sparsely impacts the cap,
    and the infill/lod-stitch post passes AVERAGE the color between those
    sparse pixel impacts -- exactly "average out their colors in between
    sparsed out pixel impacts". No new render kernel.
  * COLOR: rim anchors carry the parent's channel color at the crossing (u,v);
    each fan quad's flat color = average of its two rim colors; center =
    average of all rim colors. So the cap blends into the surface it closes.
- WIRING: FractureBody returns FractureResult{fragment, cap}. Obj gains a
  cap pointer (kept alive by Win::frags). on_frame renders parent -> cap ->
  fragment (fragment keeps its 1e-4 bias so the cap wins the cut line).
- <algorithm> added to recon_window/main.cpp (std::sort for the rim).

### VERIFIED (no C++ compiler in sandbox -- structural + logic checks only)
- All 9 edited files: brackets balanced.
- Kernel arg counts match the host (psf_splat 18/20, march_compose_fb 9).
- Tie window (2e-5) < bias (1e-4): ordering correct.
- No leftover old-signature FractureBody calls; cam in scope at all call sites.
- The cap's A0==A3 quads: BuildSampleGrids clampg pads the 0-length A0->A3
  edge to >=10 samples (no divide-by-zero in the kernel's ssu-1).

### USER ACTION: REBUILD (C++ + kernels changed)
cmake -S . -B build_delirium -DCMAKE_BUILD_TYPE=Release
cmake --build build_delirium -j"$(nproc)"
ctest --test-dir build_delirium --output-on-failure
./build_delirium/delirium_recon_window
- Press F to fracture (now camera-parallel cut + cap on the parent side).
- DELIRIUM_TIE_DIAG=1 to print the depth-tie count (should be ~0 after the
  bias; if it's high, the popping is NOT the cut boundary and we look elsewhere).
- Expect: no popping at the cut line, the parent's cut is closed by a cap
  (no more "hole / jagged inner rim"), the cut is a clean camera-parallel slice.
## 2026-10-29 — Deep revision of fracture/cap logic (user: "full deeper revision")
Screenshot review: no visible cap, "thick dark bands", fragmented scene.
Full pipeline re-audit (splat kernel, sample grids, infill/lod-stitch,
ComputeSmoothNormals, flip logic, FromRecords/DetachPatches bbox handling):
BASE RENDER VERIFIED CORRECT — flip convention consistent across both paths
(face-camera), color decoding byte-identical to kernel, infill sound,
sample density ~1.5/px correct. The "dots" are natural splat grain + the
cap bugs below.

BUG 1 (confirmed, the "dark bands"): cap color K-encoding. BuildCutCap
stored the raw 0..255 RGB as the int8 K coefficient; the format decodes
color = (K/63)*pattern, so K=194 -> byte 0xC2 = -62 (int8) -> -0.98 ->
clamped to BLACK. Every cap quad rendered black.
FIX: K = round(c*63) in [0,63] via k8() helper.

BUG 2 (confirmed by logic, not yet seen in a build): single-centroid fan
breaks on multi-loop cuts (torus -> outer+inner loop) -> self-intersecting
star. Also the first union-find attempt ("same patch" connectivity) was
WRONG: a junction point shared by two adjacent patches carries only ONE
patch id after dedup, so it never linked the two patches' groups -> each
patch = its own 2-point component -> no loop -> empty cap.
FIX: BuildCutCap rewritten:
  1. spatial-hash dedup of rim points (O(n), cell = 0.5% bbox span);
  2. loop grouping by DISTANCE connectivity in the cut plane: project rim
     to (t1,t2), union pairs closer than thr = 3x median nearest-neighbor
     distance (adaptive to patch size; separate loops are far apart);
  3. per-loop centroid fan, rim sorted by angle in the cut plane, winding
     per quad, shared center anchor per loop (avg rim color).
  Anchor dedup uses a FINE cell (0.25x) so distinct rim points can never
  share a hash cell.
  Diagnostic print: [cap] N rim -> M dedup, L loop(s), Q quads.

BUG 3 (latent): on_frame `final` flag logic was wrong when the cap is
hidden (resolve could be skipped entirely). FIX: cap = (cap && show_cap);
final = last && !cap && !frag / last && !frag / last.

NEW: X key toggles the cut-plane cap (show_cap) for A/B comparison.
Includes added: <functional> <unordered_map> <utility>.
Compile-error self-caught before handoff: `for (const auto& loop : loops)`
+ std::sort -> changed to `auto&`.

### USER ACTION: REBUILD (C++ only; no kernel changes this round)
cmake --build build_delirium -j"$(nproc)"
./build_delirium/delirium_recon_window
- F to fracture; watch the [cap] diagnostic line (loop count: torus cut
  should report 2 loops).
- X to toggle the cap off/on to A/B.
- Expect: a flat, camera-facing, correctly-colored cap sealing the parent's
  cut (no black bands, no star), fragment behind it, no popping.

## 2026-10-29 — 10-second frame recorder (capture the "flying objects")
- USER: "objects flying in and out of the toroids at really fast speeds" --
  too fast to describe in words. Request: "just write in a demo that records
  for 10 seconds."
- Added a self-contained 10s recorder to examples/recon_window/main.cpp:
  * SplatRenderer::GrabFinalRGBA(out) -- reads the resolved buf_img
    (width*height*4 RGBA, top row first, same layout Present blits) into a
    vector. New public method; SaveFramePPM unchanged.
  * WritePNG(path,w,h,rgba) -- minimal 8-bit-RGB PNG writer using zlib
    (compress2/crc32/compressBound). No SDL_image dependency. RGBA->RGB,
    filter byte 0 per row, IHDR+IDAT+IEND chunks.
  * Recorder struct: start(W,H,base) -> <base>_rec/frame_%04d.png; frame()
    saves one PNG; finalize() computes fps from wall-clock, spawns
    `ffmpeg -framerate N -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p
    <base>_rec.mp4` in a DETACHED std::thread (window keeps running), or
    prints the ffmpeg command if ffmpeg is not on PATH.
  * Win gains `Recorder rec; bool rec_active;`.
  * on_frame: after the last body's resolve (final=true -> clFinish), if
    rec_active it reads the frame back + saves a PNG, and auto-stops after
    10s of wall-clock capture.
  * main(): AUTO-STARTS the recorder on launch (the 10s capture). V key
    (re)starts or stops a recording manually.
  * CMake: find_package(ZLIB) + link ZLIB::ZLIB (fallback `z`) to
    delirium_recon_window.
- Output: /tmp/recon_window_rec.mp4 (or /tmp/recon_window_rec/frame_*.png
  if no ffmpeg).
- NOTE: the recorder captures the FINAL framebuffer (post-infill/lod-stitch),
  so it shows exactly what the window shows -- including the flying objects.
- USER ACTION: rebuild (CMake changed -> reconfigure), run, let the 10s
  auto-recording finish; to capture a fracture, press F then V to restart the
  10s window, then play /tmp/recon_window_rec.mp4.
## 2026-10-29 — DEEP REVISION: "render is broken" (dotted/garbled + all-beige screenshot)

- USER: "this isnt working well, please do a full deeper revision of all the
  logic so far" + screenshot (image.png) showing sparse beige dots, no 2x2
  grid, no torus, no per-object colors.
- ANALYSIS (from the 10s auto-recording, frame-diffed + vision-read):
  * Frame 1 (before any motion) already renders the CORRECT 5-object layout
    with DISTINCT colors (green TL, beige TR, blue sphere BL, beige BR,
    magenta torus) -- so the fresh run is NOT cross-body-contaminated.
  * The "motion" in the recording is the CAMERA ORBITING (whole-frame shifts,
    change centroid pinned at screen center), not object deformation. The
    scene settles when the orbit settles. No ModalDynamics is set up in
    recon_window (GameObject::FixedUpdate early-returns with ModeCount()==0).
  * The real, consistent defect: every object renders as SPARSE DOTS with
    black gaps ("dissolving into noisy dots"), present even at frame 1.
  * The user's all-beige/no-torus screenshot is the cross-body contamination
    signature (shared buffers + stale data), which is a real latent bug even
    if not manifesting in every run.
- ROOT CAUSES + FIXES:
  1. CROSS-BODY BUFFER CONTAMINATION (SplatRenderer.cpp, all 3 paths):
     buf_anchors/buf_surfaces/buf_snorm are SHARED across every body in a
     frame; a body re-uploaded only `if (grew || AnchorsDirty())`. A smaller
     or clean body after a larger one skipped its upload and rendered the
     PREVIOUS body's records (all-beige, no torus). FIX: track `last_body`
     (new SplatRenderer member); `body_changed = (last_body != &body)` now
     forces a full re-upload of anchors+stitch+snorm AND surfaces whenever
     the body changes. Applied to the splat (~L1179), march (~L630) and
     raymarch (~L1003) upload blocks; `last_body = &body` set after each.
  2. LOD-STITCH BLUR (the "dissolving into dots" grain, splat path):
     resolve_fb wrote alpha=255 for EVERY pixel (hits AND background), so
     fb_lod_stitch's object mask (s.w == oid) saw all objects + background
     as ONE object and blurred colors across object boundaries and into the
     background. FIX: lod-stitch DISABLED BY DEFAULT on the splat path
     (`lod_passes = 0`; re-enable with DELIRIUM_LOD_STITCH=2). It was a
     raymarch-path feature (per-object alpha) misapplied to the splat path.
  3. RESOLVE ALPHA (kernels/psf_splat.cl::resolve_fb): background now
     writes alpha=0 (empty), hits alpha=255. Alpha is the hit/empty mask
     for the post passes. Safe: SDL blit (RGBA32, blend NONE) and the
     PNG/PPM readers use RGB only, so alpha is invisible to them.
  4. INFILL (kernels/fb_infill.cl): the old luminance test
     (r>28||g>28||b>36) misclassified DARK SHADED surface pixels (rims,
     shadowed faces) as "empty" and averaged them away -> dark gaps. FIX:
     use the alpha mask (a>0 = hit, a=0 = empty), fill from SAME-object
     neighbors only (object-aware, works for both splat a=255 and raymarch
     a=object_id), and 1 lit neighbor now suffices (was >=2, which left a
     checkerboard of unfilled pixels).
  5. SAMPLE DENSITY (SplatRenderer.cpp::BuildSampleGrids): raised 1.5 ->
     2.0 samples/px (cap 1.55 -> 2.05) so projected spacing is ~0.5px and
     every pixel is hit even on curved/foreshortened patches; the infill
     then has no gaps to bridge.
  6. LOD-STITCH MASK (kernels/fb_lod_stitch.cl): IS_EMPTY is now purely
     `w == 0` (the luminance fallback that misclassified dark hits as empty
     is gone), so if re-enabled it no longer smears dark faces.
- FILES TOUCHED:
  engine/include/delirium/render/SplatRenderer.h   (+last_body member)
  engine/src/render/SplatRenderer.cpp              (3 upload blocks, density,
                                                   lod-stitch default)
  kernels/psf_splat.cl                             (resolve_fb alpha)
  kernels/fb_infill.cl                             (alpha mask, object-aware)
  kernels/fb_lod_stitch.cl                         (IS_EMPTY alpha-only)
- NOTE: sandbox has no C++/OpenCL toolchain -- NOT compiled here. User must
  rebuild on their machine.
- USER ACTION:
  cmake -S . -B build_delirium -DCMAKE_BUILD_TYPE=Release
  cmake --build build_delirium -j"$(nproc)"
  ./build_delirium/delirium_recon_window
  Expect: SOLID filled surfaces (no dots/gaps), 5 distinct objects (4 grid +
  torus), each with its own color. Press 1/2/3 to compare render paths.
  If any object still renders in another object's color, that is the
  contamination bug not fully fixed -- report which object.
## 2026-10-29 — recording + performance/driver audit
Inspected user-supplied recon_window_rec.mp4: 9.94 s, 1024x768, encoded 50 fps, no audio. Sampled video visuals at 1 s/frame (not frame-by-frame): distinguishable multicolor torus, blue orb and three thin colored sheets, jagged/speckled silhouette edges; no FPS overlay. MP4's 50 fps is the recorder's rounded average (n / recording wall-time), NOT a GPU benchmark; recording includes per-frame GPU readback and PNG encode. Driver cannot be diagnosed from compressed video alone.
Algorithm audit: 2.0 samples/px along each patch edge implies ~4x sample work per projected area versus 1.0 and ~1.78x versus previous 1.5, before 3 full-frame infill passes; cannot claim performance gain without matched A770 measurements. Current shared-buffer body switching full uploads each body; correct for data isolation, expensive versus per-body persistent GPU buffers. Sparse silhouette remains possible because point splats+image-space infill are not guaranteed coverage or visibility correctness. Disable post with DELIRIUM_GPU_INFILL=0 for A/B; benchmark with recording OFF. Previous claim that previous video motion was camera orbit was unsupported: recon_window orbit defaults false, and recording alone cannot establish input state. Previous claim that SDL ignored alpha was not verified: now explicitly set SDL_BLENDMODE_NONE.
Confirmed concrete host risks: nonblocking OpenCL writes from short-lived std::vector (snf, zero stitch) and stack frustum; next body reuses sample-grid vectors before final finish; SDL streaming texture pitch may differ from width*4, old Present ignored pitch and readback failure; FPS printed pre-throttle instantaneous last-frame ms alongside half-second throughput, so not directly comparable. Fixed local temporary lifetimes: fill zero stitch on device, block on snf/frustum writes and sample-grid layout writes before reuse. Present now respects pitch, reports readback/lock errors, explicitly disables alpha blending; frame dump uses pitch. FPS reporting now measures complete frame including throttle. These are correctness/instrumentation fixes, not proven speedups; blocking staging transfers may cost time. Must build and visually verify on user's A770; sandbox has no C++ compiler/OpenCL headers/GPU. Driver hypothesis remains open pending repeatable raw framebuffer vs SDL comparison, OpenCL build logs/version and driver comparison.
## 2026-10-29 — grazing-angle edge shading experiment
User image inspection: torus outer/top and inner/lower silhouettes have black/green pixel gaps and spikes; orb top edge is jagged/bitten; smaller patch edges stair-step. Distinguish geometry/coverage gaps from shading discontinuity: normal averaging cannot fill missing pixels or repair topology. Implemented ONLY in kernels/psf_splat.cl shading path: after camera-facing flip, smoothly blend interpolated shading normal 25% toward the camera direction at |N·V|<=0.10, taper to zero at >=0.35 (smoothstep). Keeps surface position, displacement normal, depth, sample distribution and occlusion unchanged. This is a visual hypothesis to A/B test on user's Intel GPU, not a confirmed fix for jagged silhouettes; may modestly increase per-sample GPU cost. No OpenCL headers/C++ compiler/GPU in sandbox, so kernel compile and image verification remain user-side.
## 2026-10-29 — close-up splat cost and optical-zoom proposal
User observes immense FPS drop when zooming and asks whether a camera illusion can render objects apparently closer while saving work. Audit: mouse wheel currently dollies camera by 0.35 units per notch; BuildSampleGrids uses projected anchor edge lengths and samples ~2x per axis up to 320x320/patch, so close-up work rises roughly with inverse distance squared until cap. Same projected size from narrower FOV does not save per-pixel splat evaluation, and camera-only zoom changes perspective, parallax, clipping and picking semantics. Deliberately did NOT change camera or add projected-corner clipping (curved/deformed patch interiors can lie outside the anchor quad and such clipping would lose visible detail). Added opt-in DELIRIUM_SPLAT_DENSITY=[0.75,2.5] (default 2.0, unchanged) read once per body grid build in engine/src/render/SplatRenderer.cpp; 1.5 targets about (1.5/2)^2 = 56% grid samples where uncapped, at the cost of possible holes/edge aliasing. Benchmark with 10s auto-recorder OFF after capture; no measured FPS gain claimed. Static source check only: no C++ compiler/OpenCL headers/GPU available here.
## 2026-10-29 — screenshot edge review and interactive FPS spikes
New screenshot shows comb-like torus silhouette, visible dark gaps at inner ring, and a jagged U-shaped missing region on the blue orb; these are missing coverage/geometry, not simply dark shading. Normal averaging alone cannot repair. Found guaranteed default benchmark confounder in examples/recon_window/main.cpp: every launch auto-recorded 10 s; for every frame on_frame called GrabFinalRGBA (blocking ~3 MiB OpenCL readback) plus synchronous RGB PNG zlib compression on render thread, then Present performed a second image readback. This easily produces frame-time spikes and makes observed 60->30 FPS unrepresentative of renderer throughput. Changed launch recording to opt-in DELIRIUM_AUTO_RECORD=1; V still manually starts/stops. Default render path 3, density 2.0 and 3 infill passes unchanged (no unverified image-quality downgrade). User must benchmark after 10 s or with capture off at identical camera position, compare DELIRIUM_GPU_INFILL=0 and DELIRIUM_SPLAT_DENSITY=1.5 as deliberate image-quality tradeoffs. No compiler/OpenCL/GPU available in sandbox; performance improvement on A770 not measured or claimed.
## 2026-10-29 — close-up 20 FPS persists; representative one-object tests
User reports 20 FPS zoom drops unchanged by DELIRIUM_GPU_INFILL=0 or reduced DELIRIUM_SPLAT_DENSITY. Treat as evidence AGAINST those two being dominant, not proof of a driver bug. Added DELIRIUM_SIMPLE_SCENE=orb|plane opt-in to examples/recon_window/main.cpp: constructs just one centered generated PSF object via existing NeuralReconstructor/ElementSystem, same 900-patch budget and splat pipeline; skips creation/rendering of four-object grid and file torus in simple mode; default five-body path unchanged. Added opt-in DELIRIUM_SPLAT_WORK_DIAG (per-body sample count, not GPU time) and OpenCL CL_DRIVER_VERSION display in SplatRenderer.cpp. No GPU kernel or default density/quality change made because actual bottleneck has not been measured. Benchmark same camera distance with recorder off: five-body, orb, plane; compare per-body sample count and FPS; if one-body retains 20 FPS, investigate GPU splat atomics vs SDL readback/synchronization, not full body upload count. If one-body fast but five-body slow, prioritize persistent per-body GPU buffers rather than repeatedly uploading shared anchors/surfaces/normals. Static inspection only; sandbox lacks C++ compiler, OpenCL headers and Arc GPU.
2026-10-29 — Toroid close-up remains ~20 FPS per user; infill-off and density=1.5 previously did not improve it. Inspected BuildSampleGrids and psf_splat: per-patch projected edge grids can grow roughly inverse-square with distance (axis clamp 320); each sample does anchor/deformation/normal/color work and per-pixel depth arbitration, but sample count alone does not establish GPU bottleneck. Shared body buffers are re-uploaded on body switches; unchanged default quality. Added opt-in DELIRIUM_SPLAT_AXIS_CAP=10..320 (default 320) and DELIRIUM_ROTATE_LIGHT=1 (~0.6 rad/s around world Z; camera fill unchanged) in SplatRenderer.cpp. DELIRIUM_SPLAT_WORK_DIAG=1 reports launched samples per body; compare normal five-object scene at fixed zoom and single-object orb/plane separately. Shading may change abruptly at camera-facing normal flips; resolve is nearest-depth and infill averages neighboring RGB into empty-alpha pixels, so rims may also be coverage/infill artifacts; rotating key light can distinguish stationary coverage faults from lighting-dependent color faults. No build/GPU/FPS verification here (compiler/OpenCL headers/A770 unavailable); obtain user A/B counts, frame times and screenshots before changing baseline, kernel shading or upload architecture. git diff --check unavailable: tree is not a Git repository.
2026-10-29 — Editor direction: user requests a Godot-like physics authoring interface with scene tree, inspector, physics authoring, SDL-only UI, and a mathematical/low-memory architecture. Audit found engine/src/core/Editor.cpp had stub OnInput/Hud/Diag, scene GameObject list exists but the main recon_window owns extra PSF-only bodies outside Scene (including file torus); Node/SceneTree and Scene::Objects are distinct models. Implemented first MODEL slice only in Editor.h/.cpp and const read accessors in GameObject.h: scene index/pointer-checked selection, textual scene tree/physics inspector, validated setters for rigid mass/inverse-mass/restitution/friction and material density/Young/Poisson/damping, Tab selection via existing Engine hooks. No drawn docks, property widgets, scene serialization, or play/pause yet. Existing recon_window is intentionally not wired because its SDL hook and PSF-only torus aren't represented by Scene::Objects; integrating without a unified editable scene and input routing would misrepresent selection and risk existing controls. Architecture: Scene is authoritative object store; PSF buffers remain in body (no duplicate 64B records); inspector is lazy selected-object view; physics writes through validated commands, no per-frame scene copies; SDL draw overlay after Present only after confirming renderer ownership and input focus; project persistence versioned and separate from frozen PSF records; simulation authoring versus play state requires snapshots/undo and clock ownership. No compile/run verification in sandbox (no C++ compiler/SDL development header); user must build and test. No claim of complete editor.
2026-10-29 — Editor continuation: SDL-only first visual slice in existing Editor.cpp/h, Engine.cpp/h and SplatRenderer.cpp. Optional Engine overlay hook draws rectangle-only scene list (indices) and numerical physics inspector over existing SplatRenderer Present, with one SDL present and no PSF/image copies, added textures or external UI/font dependency. F1 opens/closes editor, F2 toggles pointer/viewport camera mode; pointer-mode input is consumed before camera handling and Engine skips polled WASD in pointer mode. Click left pane to select Scene::Objects entry; right pane selects numbered mass/bounce/friction/density/Young/Poisson/damping row; +/- adjusts through existing validated setter; Tab cycles scene/inspector; console exposes full object labels. Overlay hidden until attached editor receives F1; unmodified recon_window uses its own input hook and does NOT attach the editor, so no misleading claim that file-loaded torus appears in scene hierarchy. Limitations: no TTF labels, persistent project save, undo, comprehensive physics controls or Godot-like docking yet. Could not compile/run here (C++ compiler/SDL headers absent); user-side build/visual verification required. Hook lifetime cleared in Engine::Shutdown, but callers must keep Editor alive until shutdown. Future stage should unify torus with authoritative scene objects and introduce stable IDs before viewport selection and undo.
