# Fixes Register — Delirium Engine

Every bug found and fixed in the editor / AI / rendering work, with the
symptom, the root cause, the fix, and the evidence that the fix holds.

Each entry is written so it can be **re-derived** if the code changes
again: the symptom is what you would see, the cause is what you would
find, and the evidence is what proves the fix.

---

## 1. `Transform::ToMatrix()` wrote translation to the wrong slots

**Symptom** — every object placement was silently ignored. Moving an
object did nothing, with no crash and no NaN. The transform looked like a
no-op for any point near the origin.

**Cause** — `ToMatrix()` wrote translation into `m[12], m[13], m[14]`.
That is the **w-ROW** of a row-major 4×4, not a translation slot. Both the
CPU point transform (`Mat4_TransformPoint`) and the GPU's `xform_point`
read translation from the last **COLUMN** — `m[3], m[7], m[11]`. The two
disagreed about where translation lives, so the offset went where nothing
looked.

**Fix** — translation in the last column, and `ToMatrix` made to agree
with `InversePoint` (which already assumed `M = T·R·S`).

**Evidence** — `tests/validate_move.cpp` (29 checks): moving an object
leaves the frozen 64-byte records **bit-identical** while the rendered
position changes.

**The lesson** — in a row-major `M·(p,1)` convention, translation belongs
in the last COLUMN. Cross-check the CPU transform and the kernel's
transform against each other.

---

## 2. Every body rendered with `clear=true` — only the last was visible

**Symptom** — a multi-object scene showed only one object.

**Cause** — `Engine::Run` called `RenderBody(body, cam)` with the default
flags for every body. The defaults are "clear" and "final", so each body
wiped the framebuffer before drawing, and each body ran its own resolve.

**Fix** — build the draw list first, then assign the flags from the list's
length: body 0 clears, every body composites (`clear=false`), only the
last sets `final=true`. An empty scene gets an explicit clear + resolve.

**Evidence** — the workspace renders the orb and the torus together.

**The lesson** — in a shared-framebuffer renderer, "clear once, composite
all, resolve once" is a contract, and the natural-looking per-body default
call is exactly wrong.

---

## 3. Fracture fragments were drawn after the resolve

**Symptom** — a fractured piece vanished.

**Cause** — detached fragments were drawn after the full-screen resolve,
so the resolve overwrote them.

**Fix** — fragments are in the draw list, with the 1e-4 NDC depth bias the
recon harness uses (so a coplanar cut rim does not flip the depth winner
frame to frame).

---

## 4. The arg-count audit silently disabled the entire march path

**Symptom** — "render path 1" produced **byte-identical output** to path 3
(splat), while still printing the `march:` banner. Path 1 was rendering as
the fallback all along.

**Cause** — the init-time audit expected `layer_draw_band` to take **28**
kernel arguments (as if the four model-matrix rows were parameters). The
kernel signature and the host's `clSetKernelArg` sequence both used
**26**. The audit therefore mismatched on every frame and disabled the
path. The disabling message went to **stderr** while the success banner
went to **stdout**, so a probe checking only stdout saw the banner and
concluded the path had run.

**Fix** — corrected the expected count; the audit's disable message and
the path's banner now go to the same stream.

**Evidence** — scene A's and scene C's PPM md5s now differ.

**The lesson** — an expected-arg-count literal is a **third** thing that
can drift (kernel signature, host arg sequence, audit literal). When one
changes, update the other two. And never let "path N" degrade to "path N-1"
without the pixels saying so.

---

## 5. A failed kernel compile silently swapped the render path

**Symptom** — a broken `.cl` build made the scene render through the
fallback, which still looked plausible enough to fool an A/B comparison.
The gallery still reported "19 ok".

**Cause** — the build error printed to stderr and disabled the path; the
harness exited 0, wrote its files, and reported success.

**Fix** — always assert, for any kernel change: (1) the path's own "ready"
line on stdout, (2) **no** build-failure text, (3) the target scene's
bytes differ from a known fallback's.

**Evidence** — hit again while adding the area-weighted path: `abs` on a
float is **ambiguous in OpenCL C 1.2** (it resolves to the integer
overload), the whole program failed to build, and the area path was
silently off. Fixed with `fabs`. The giveaway was the missing banner.

---

## 6. The layered splat: four type / precedence bugs

**Symptom** — the user's distance-plane method produced a white wash, then
an empty image, then black pixels.

**Cause** — four separate bugs, each producing a different extreme:

| # | Bug | Symptom |
|---|---|---|
| 1 | Band assigned by **radial distance** instead of **view-axis depth** | Rainbow torus averaged to white — points at very different depths shared a band |
| 2 | Colour buffer declared `uint*` by the writer, `float*` by the reader | Integer sum 4096 reinterpreted as a denormal (~5.9e-41); division produced garbage; `clamp` saturated it — **88% pure white** |
| 3 | The write pointer advanced **4 floats** per slot (`band_color + bslot*4u` on a `float*` base) | Every impact landed 4× past where the readers looked — **image empty** |
| 4 | **Cast precedence**: `(uint)clamp(rgb.x) * 255.0f` binds the cast tighter than `*` | Truncated to an integer *before* scaling — 0.42 → 0 → **black**; with the denormal, → 1.0 → 255 → **white**. The same line produced both extremes |

**Fix** — band by `dot(P - eye, view_fwd)`; one packed 64-bit
`(depth_bits << 32) | rgba` key per slot (so the nearest-wins claim and
the colour are one indivisible `atom_min`); all three kernels agree on
`uint`; and multiply in float, cast **last**:
`(uint)(channel_color(...) * lam * 255.0f)`.

**Evidence** — classic 16,044 lit / 0 white; layered 20,773 lit / 0 white;
visually a solid torus with a clean open hole.

**The lesson** — when a shader produces "all white" or "all black", check
the **cast position** before anything else. It is invisible in a
screenshot and produces the *opposite* symptom from a small numeric error,
which misdirects the search.

---

## 7. The band range was derived from the anchors, not the deformed surface

**Symptom** — the layered path lost a whole shell of surface at a curved
body's extremes (~450 lost lit pixels).

**Cause** — the rendered sample is `P + h·Nd` (displaced along the anchor
normal), so a curved body rises above its anchor plane. A range derived
from the anchor positions cut through the displaced extremes and those
samples were dropped.

**Fix** — compute the range from the **deformed** surface using the march
path's proven 5-point patch math; and `band_of` now **clamps** to the
nearest band instead of returning −1, so a sample is never silently
discarded.

---

## 8. The editor grabbed the mouse pointer and never released it

**Symptom** — the mouse **disappeared**, was **captured** (motion events
became relative to a virtual centre), and dragging "outward" ran into the
window edge and started interacting with whatever was under it.

**Cause** — `Engine::Initialize` grabbed the pointer in relative mouse
mode **before** the workspace decided it wanted a cursor. The grab was
never released.

**Fix** — `SetCameraControl` re-applies the mode **at the point of
change** (`Engine.cpp`), releasing the grab and calling `SDL_ShowCursor`.

**The lesson** — anything applied once at init (pointer grab, fullscreen,
vsync, cursor visibility) has the same trap: the code that wants the
opposite mode arrives later and finds the init-time state still in force.

---

## 9. Dragging a slider dragged the *next* slider under the cursor

**Symptom** — "when I drag around more outwardly I end up clicking and
dragging sliding bars, overlapping them by simply moving it around."

**Cause** — `NumberField`/`Slider` scrubbed on `mouse_down && hover`, so
ownership of the drag belonged to "whatever is under the cursor" **every
frame**. One gesture could drive several widgets. There was also a
`static int last_x` inside `NumberField` — shared across **every
instance** of the widget.

**Fix** — **drag capture** in the UI toolkit: the widget that was pressed
claims the drag; while a capture is live only that widget responds,
anywhere in the window; the capture is released on button-up; scrubbing
uses the **delta from the drag's origin**, not the absolute cursor
position, so the value does not jump at the boundary; and a `drag_moved`
flag keeps a click distinguishable from a drag.

**The lesson** — in an immediate-mode toolkit, a **click** is instantaneous
(the widget under the cursor is right), but a **drag** is a gesture that
belongs to one target. The immediate-mode equivalent of focus-with-capture.

---

## 10. Orbiting the camera also applied the sculpt tool

**Symptom** — one gesture did two things.

**Cause** — the viewport used the same button for "look" and "act".

**Fix** — separate them **by construction**, not by modifier: a sculpt
click applies the tool and **never** orbits; the camera moves only via the
nav ball, the keyboard, or Navigate mode. The nav ball lives in its own
rect, hit-tested before the viewport, so it can never land on the
geometry.

---

## 11. The editor starved the local model of CPU

**Symptom** — the Generate button looked broken. A 7B model's ~25 s reply
stretched past **300 s**.

**Cause** — the editor rendered every frame at full resolution while the
model generated; both wanted the same cores.

**Fix** — throttle the render loop to 10 FPS while the model is busy; echo
the streaming partial reply into the console (a console that shows nothing
until the reply completes is indistinguishable from a hung app); never
block the render thread on the model (the client runs on a worker thread);
and keep the reachability probe off-thread too.

---

## 12. A single infinite plane drowned the dome — "the sphere only shows one face"

**Symptom** — the model asked for "a smooth glass ellipsoid" and produced a
**flat card**. The user reported it as "the sphere isn't there, it only
shows one face".

**Cause** — the model emitted an ellipsoid **with a plane child**, and the
field composition used a **flat weight**. The plane is infinite at full
weight, so it drowned the bounded dome.

**Fix** — **depth-weighted contribution**: a nested child's weight halves
per level, so a parent keeps a majority voice and a deeper system
**refines** a shallower one instead of replacing it.

| | centre z | flat samples |
|---|---|---|
| flat weight (broken) | ~0.25 | ~100% |
| depth-weighted (fixed) | **0.333** (67% of the pure dome's 0.500) | 84% |

**The lesson** — "overlapping surfaces average" only holds if the
overlapping systems contribute **comparably**. A flat weight lets the
widest / most infinite system dominate, which is the opposite of what
"averaging" means. Nesting depth is the natural proxy for specificity.

---

## 13. A graph-based substance cannot make a closed sphere

**Symptom** — the rendered "sphere" is a **dome cut off flat at the
equator**.

**Cause** — **architectural, not a bug**. `ISubstance::Eval(x, y)` maps
local `(x,y)` to a surface point + normal — a graph `z = f(x,y)`. That
cannot represent a closed surface; it can only describe the **top sheet**.
`EllipsoidSubstance::Eval` returns `z = sz·√(1 − x²/sx² − y²/sy²)`, which
is the upper hemisphere *by definition*. The same limit applies to the
torus (upper half of the tube — which is why its hole reads correctly but
its underside is missing).

**Status** — **SOLVED by fix 26** (the multi-layer bake). A substance is
still a graph — that property is unchanged and correct — but the baker now
emits each sheet as its OWN layer, so two graphs compose into a closed
surface. The substance vocabulary still has no first-class closed-body
helper, but the *baker* no longer collapses a multi-sheet body into one
dome.

---

## 14. The baker chorded bounded features at coarse grid resolutions

**Symptom** — "make a torus" produced a flat slab with no hole.

**Cause** — a PSF patch is reconstructed as **bilinear corner positions +
a normal-direction displacement** (`P = Pb + h·Nd`) — a **tangent plane**
through the four corners. A feature whose size approaches the sample
spacing is **chorded**. The sample spacing is `extent/(N−1)`, so a bounded
substance whose radius is comparable to that spacing loses its defining
shape.

| grid_n | spacing | field z-span | body z-span | body keeps |
|---|---|---|---|---|
| 12 | 0.091 | 0.0749 | 0.0146 | **19%** |
| 16 | 0.067 | 0.0844 | 0.0343 | 41% |
| 32 | 0.032 | 0.0991 | 0.0590 | 60% |
| 48 | 0.021 | 0.1080 | **0.0720** | **67%** |
| 64 | 0.016 | 0.0999 | 0.0746 | 75% |

**Fix** — bake at a resolution chosen from the **smallest feature** the
substance must express, not from a default (48+ for a torus).

**The lesson** — bake cost is not a reason to stay coarse. 8 ms at
`grid_n=48` vs 0.3 ms at 12 — irrelevant next to a model's reply time.
Coarser is not "faster and fine"; it is a different, wrong shape. Assert
the bake preserves the **FIELD** as a *fraction*, not an absolute.

---

## 15. The model's JSON was rejected on a trailing comma

**Symptom** — the full prompt → geometry pipeline failed on a reply that
looked fine.

**Cause** — after a comma, the array parser called the value parser, which
saw the `]` and reported "expected number". **Trailing commas are the
single most common LLM JSON defect.**

**Fix** — accept a trailing comma in both arrays and objects; add explicit
regression tests.

**Evidence** — the parser gate went from 84 checks / 1 failure to
**98 checks / 0 failures**.

---

## 16. The nlohmann optional dependency broke the whole library build

**Symptom** — `LLMClient.cpp` included `<nlohmann/json.hpp>` unconditionally
while CMake declared `nlohmann_json` **OPTIONAL**. The file sits in the
`engine/src/*.cpp` GLOB, so on a machine without nlohmann the **entire
engine lib and every CTest gate** failed to build — not just the LLM
client.

**Fix** — use the engine's own strict, dependency-free reader
(`ai/ElementTreeParser.h` → `DeliRium::AI::ParseJson`). The optional
dependency is off the critical path entirely.

**Evidence** — the whole tree builds and **21/21 tests pass** with
nlohmann genuinely absent:

```bash
cmake -S . -B /tmp/no-nlohmann -DCMAKE_DISABLE_FIND_PACKAGE_nlohmann_json=ON
cmake --build /tmp/no-nlohmann --parallel 8 && (cd /tmp/no-nlohmann && ctest)
```

**The lesson** — the guard must be on the **SOURCE**, not just the target.
A `GLOB_RECURSE` source list picks up a file that needs an optional
package and breaks the whole library.

---

## 17. The layered path's in-place fill was a data race

**Symptom** — three renders of the same scene produced three **different**
md5s.

**Cause** — the band's hole fill read neighbours from the buffer it was
writing, so a work-item's neighbours could already have been overwritten
by others in the same dispatch. The result depended on the GPU's
scheduling.

**Fix** — ping-pong between two distinct band buffers, and mirror the
classic path's atomics.

---

## 18. The grid's one-surfel-per-cell silently dropped 34% of the surfels

**Symptom** — the march path rendered a noisy, salt-and-pepper speckled
torus: **52% edge density, 853 holes**.

**Cause** — `build_grid` stores **one** surfel index per cell, so a later
surfel overwrites an earlier one. On the Delirium torus (uvn=10): 54,400
surfels → 35,696 occupied cells → **34.4% of surfels lost**. The
survivors are irregularly spaced, so the hit radius sized from the
*nominal* spacing was far smaller than the real gaps, and rays slipped
between surfels.

**Fix** — size the hit radius from the **grid cell**, the scale at which
information is actually lost:
`foot = max(2·cell, 1.6·spacing)`. Plus the hard invariant
`nr × cell >= foot` (the neighbourhood search radius must cover the hit
radius; when `nr` is clamped, clamp the **effective** radius too — and
`foot` is `const` in OpenCL, so clamp into a local).

**Evidence** — holes 853 → **20**, edge 52.5% → **7.0%**, and it was
*faster* (43.1 ms vs 45.0 ms — fewer march steps needed).

---

## 19. The editor's UI toolkit had a memory leak and a placeholder

**Symptom** — `*(new JsonValue())` passed as an out-parameter leaked once
per call, and a placeholder label sat in the workspace's draw code.

**Cause** — scaffolding residue from an in-progress file.

**Fix** — a stack object; the placeholder removed.

**The lesson** — a scaffold that **compiles** is not a scaffold that
**links**. A `GLOB_RECURSE` picks up a half-written file, compiles it into
the library, and passes every test while the feature is incomplete. Audit
a picked-up component against its own header before writing anything.

---

## 20. A render-path comparison averaged over scenes the path never touched

**Symptom** — a probe reported "holes 3380 → 4571" — a catastrophe.

**Cause** — **13 of the 19** gallery scenes use the **march** path, where
the splat-layer setting has no effect, so those rows compared march against
march. The real splat-path scenes showed the **opposite**: dither improved
2.4–3.3%.

**Fix** — a per-scene A/B must **skip** any scene whose two renders are
byte-identical — that is the proof the path under test did not run there.

---

## 21. The LLM client's JSON parse needed the in-tree reader

*(Covered by fix 16 — the header comment still claimed nlohmann after the
implementation had moved. Fixed, and the whole-tree build without
nlohmann is now the regression test.)*

---

## 22. The area-weighted splat: `abs` on a float is ambiguous in OpenCL

**Symptom** — the area-weighted path produced **no output at all**, while
the gallery still reported "1 ok". The path was silently disabled.

**Cause** — `abs(dot(Nshade, view))`. OpenCL C 1.2's `abs` resolves to the
**integer** overload for a float argument, and the compiler rejects it:
`error: call to 'abs' is ambiguous`. The whole program failed to build, so
the kernel was never created.

**Fix** — `fabs` for float arguments.

**Evidence** — the gallery now prints its own banner
(`splat-area: 640x480 samples=191839 footprint=area-weighted`), and the
ad-hoc probe asserts on that banner.

**The lesson** — this is fix 5 recurring. A kernel compile failure is
**silent** from the harness's point of view: it exits 0, writes its files,
and reports success. Always assert the path's own banner AND the absence
of build-failure text.

---

## 23. The area path resolved straight into the image, breaking multi-body compositing

**Symptom** — caught during implementation, before it shipped.

**Cause** — the first implementation resolved the area accumulators into
`buf_img` directly, bypassing the shared `(depth|rgba)` framebuffer. That
framebuffer is the **multi-body compositing surface**: every body writes
its pixels there and the nearest wins. Writing the result straight to the
image would have made the area path single-body only — the last body
would overwrite the others.

**Fix** — `area_resolve` writes the same `(depth|rgba)` key into the
shared framebuffer via `atom_min`, exactly as the classic path does. The
area accumulators stay **per body** and are reset per body, which is
correct: the area weighting answers *"how much of THIS surface covers THIS
pixel"*, and the depth test answers *"which body wins"*.

---

## 24. The document learner counted tree-leaves instead of sheets

**Symptom** — the gate rejected a perfectly good closed body: *"the tree
has only ONE leaf"*.

**Cause** — the sheet count walked the tree and counted elements with **no
children**. But an element that has a substance **and** children (a
housing with a rotor inside it) contributes a **sheet** while also being a
tree-internal node — so the two-sheet housing was counted as one.

**Fix** — count the **sheets**: every element that has a substance counts
one, regardless of whether it has children. A tree with fewer than two
sheets cannot be a closed body.

**Evidence** — `validate_doc_learner`: 47 checks, 0 failures, including
the round-trip "a two-leaf tree bakes into a real body with patches and
anchors".

---

## 25. A render-path switch that is invisible in the log

**Symptom** — the editor's path buttons and the app's `--render-path`
flag appeared to work (the right banner appeared, the right pixels
appeared), but there was no way to assert from a script **which** path was
in force. The renderer printed its path only at init, so a later switch
left the log claiming the original path.

**Cause** — `SetPath` was a one-line assignment in the header. It changed
the path and said nothing. That is the same failure class as a silently
disabled kernel (fix 5): a switch that is invisible is indistinguishable
from a switch that did not happen.

**Fix** — `SetPath` moved out of the header and now **announces itself**:

```cpp
void SplatRenderer::SetPath(Path p) {
    path_ = p;
    std::printf("render path: %s\n", PathName());
}
```

The **last** `render path:` line in the log always names the path
currently in force, whatever triggered the change (the UI buttons, the
app's flag, or a gallery scene).

**Evidence** — the ad-hoc probe drives all five paths through the live
editor and asserts the reported name for each: `march`, `raymarch`,
`splat`, `splat-layered`, `splat-area`.

**The lesson** — a state change that affects what the user sees should
**announce itself**, especially when the failure mode is a silent
fallback that still looks plausible.

---

## 26. The baker collapsed a multi-sheet body into one dome — "the sphere only shows one face"

**Symptom** — a body built from an upper and a lower hemisphere rendered
as a **single dome with a flat bottom**. The five-body scene showed four
half-ellipsoids and one torus; every "sphere" had one visible face and a
hard flat cut at the equator.

**Cause** — the field composer blends every active sheet at a location into
**one** surface point, and the baker sampled that blended field on a
**single 2D grid at z=0**. A substance is a graph `z = f(x,y)`, so the
query's `z` never places the surface — it only selects which sheet.
Two sheets at the same `(x,y)` therefore **averaged into one point**
instead of forming two surfaces that meet at a seam.

Measured proof (the field's `point` is completely independent of the
query's `z`):

```
A upper + lower(origin=-0.5, scaleZ=-1)
  z=-0.90 -> point.z=+0.0000     z=+0.90 -> point.z=+0.0000
C single ellipsoid
  z=-0.90 -> point.z=+0.5000     z=+0.90 -> point.z=+0.5000
```

**Fix** — the **multi-layer bake** (`engine/src/ai/ElementSystem.cpp`).
Each leaf sheet (an element with a substance) is baked as its **own
layer**: its own anchor grid, its own patch set, its own surface. The
layers are concatenated into one anchor/surface list, so the renderer
composites them by depth and the two hemispheres MEET at the seam. Three
details had to move with it:

- the patch-fit target uses the **sheet's own** surface, not the blended
  field — fitting against the blend would drag the upper layer back down
  toward the average, reproducing the flat-bottomed dome;
- the colour indices are **layer-local** (the global anchor index is
  layer-offset);
- `BakeBody`'s bbox recompute must match the new bake **exactly** — the
  stored positions are dequantized with that box, so a different box
  dequantizes every position wrong.

The anchor index space is `uint16`, so the total across all layers is
checked and reported rather than silently truncated.

**Evidence** — the gearbox housing went from **17,772 → 239,396 lit
pixels** (13×) and now renders as a closed solid with a visible
underside; a sphere (upper + flipped lower at the same origin) renders
as a complete ball. Patch counts confirm the layers are both present:

```
A single ellipsoid (1 sheet)            patches=569   z-span=0.182
B upper + lower origin=0 scaleZ=-1      patches=1707  z-span=0.540
C upper + lower origin=-0.25            patches=1707  z-span=0.707
D upper + lower origin=-0.5             patches=1707  z-span=0.939
```

**The lesson** — "average the overlaps" is right for *co-located* detail
but wrong for *stacked* sheets. Averages belong between things that occupy
the same place; surfaces that occupy different places must stay separate.
When a representation cannot express a thing, the answer is usually a new
**layer**, not a smarter average. Also: the composition rule for a closed
body is **origin=0 + negative z-scale**, not a stacked offset — the
flipped sheet must share the origin it is mirroring.

---

## 27. The scene probe reported "rendered nothing" when it had never written the file

**Symptom** — `complex_scene` reported `lit=0 clusters=0` and FAILED for
both scenes, on a renderer that was demonstrably working (the gallery
rendered 21/21 in the same session).

**Cause** — the probe never created its **output directory**. `SaveFramePPM`
fails when the directory is missing, and the failure was silent — the
probe then measured a file that had never been written. A missing
directory therefore looked exactly like a rendering failure.

**Fix** — create the directory up front (`mkdir_p`, the gallery's
established pattern), and make a failed save **loud**: `RenderAssembly`
prints to stderr and returns false, and the caller counts it as a
failure instead of measuring a phantom file.

**Evidence** — the same binary, unchanged renderer:

```
before:  gearbox: lit=0 (0.00%) clusters=0   -> FAILED
after:   gearbox: lit=239396 (34.63%) clusters=9
         five bodies: lit=24699 (3.57%) clusters=24
```

**The lesson** — a measurement that reads a file must distinguish "the
file is empty" from "the file is not there" from "the file was never
written". Check the **write**, not just the read. This is the same family
as fix 4 and fix 5: a silent failure that produces a plausible-looking
result is worse than a crash.

---

## 28. The complex-scene probe never exercised the render paths

**Symptom** — `complex_scene` rendered the default path only. A path that
was silently disabled (fix 4) or that broke on multi-body compositing
would not have been caught by a single-body scene at all.

**Cause** — the probe had two scenes (one body, five bodies) and no path
coverage. The gallery's per-path scenes each render ONE body, so the
multi-body compositing contract was never tested against a non-default
path.

**Fix** — a third scene renders the five-body scene through **all five**
paths and measures each one: lit pixels and, crucially, the number of
**distinct lit regions** (a flood-fill cluster count). A path that falls
back produces the fallback's numbers; a path that renders nothing
produces zero; a path that breaks the compositing contract produces one
region instead of five.

**Evidence** — every path shows ≥4 separate regions:

```
path march          lit=27565  clusters=23
path raymarch       lit=8600   clusters=34
path splat          lit=21771  clusters=19
path splat-layered  lit=28023  clusters=42
path splat-area     lit=22240  clusters=12
```

**The lesson** — a path switch is only proven by **measuring the output
of each path**, on a scene that exercises the contract you care about.
A banner line says the path was *selected*, not that it *worked*.

---

## 29. The document pipeline's ceiling is model latency, not the seam

**Symptom** — the live document probe produced 1 valid tree from 3
documents; the other 2 failed.

**Cause** — **not a seam failure**. Both failures were at **transport**:
the model returned nothing before the 300 s timeout. The one document that
did come back produced a complete, valid, **baked** body (4 sheets, 2209
patches) — so parse → validate → bake works end to end against the real
local model.

**Evidence** — `doc_learner_probe` against Ollama
(`qwen2.5:7b-instruct-q8_0`):

```
--- [2/3] M-220 hydraulic pump ---
    attempt 1: OK (274.9s)
    sheets=4  patches=2209  anchors=2304
    GEOMETRY OK
documents: 3 | valid trees: 1 | baked bodies: 1
failures logged: 2 | successes logged: 1
```

**Status** — the failure log is populated and the escalation ladder is
wired, but has **not** been driven against a real larger model (it is
unit-tested with a stub callback).

**The lesson** — distinguish *transport* failure from *semantic* failure.
"2 of 3 failed" means something very different when the 2 are timeouts
(the model is slow) than when they are parse rejections (the model is
wrong). Log the stage, not just the failure.

---

# Verification status

| Suite | Result |
|---|---|
| `ctest` (canonical, 22 gates) | **22/22 passed** (8.83 s) |
| `validate_ai_parser` | 120 checks, 0 failures |
| `validate_doc_learner` | 47 checks, 0 failures |
| `validate_move` | 29 checks, 0 failures |
| Area-weighted splat probe | deterministic (identical md5 across runs), 0 white pixels, not a silent fallback, banner present |
| Area vs layered rim A/B (live editor) | area rim wiggle **1693** vs layered **1819** — the area path is **7% smoother**; the residual stair-stepping belongs to the **baked geometry** (fix 14), not to the area weighting |
| Render-path switch probe (live editor) | all five paths run, each announces its own name (`march` / `raymarch` / `splat` / `splat-layered` / `splat-area`); an out-of-range index is ignored without crashing |
| Complex-scene probe (5 bodies × 5 paths) | every path renders ≥4 separate regions; lit px 8,600–28,023 |
| Closed-body bake (fix 26) | gearbox 17,772 → **239,396** lit px (13×); sphere renders as a complete ball with an underside |
| Live document pipeline (real local model) | 1/3 documents → valid tree → **baked body** (4 sheets, 2209 patches); the other 2 failed at **transport** (timeout), not at parse/validate |
| Whole-tree build **without** nlohmann | builds clean, 21/21 tests pass |

CTest exercises CPU topology and physics — it renders **nothing**. Every
render-path claim above is backed by a focused pixel probe against the
live binary, not by the suite.

---

## 30. A missing brace made the anchor loop run once per sheet

**Symptom** — a 2-sheet body produced **9,216 anchors** where 4,608 were
expected. The extra blocks were never referenced by any surface, so the
blob was twice the size it should be and the measured z-extent of a baked
body was wrong (a radius-0.5 sphere read `[-0.4452, +0.7206]` instead of
`[-0.5000, +0.5000]`).

**Cause** — in `ElementSystem::Bake`, the per-sheet **sampling** loop was
never closed. The brace that ends it was missing, so the **anchor-writing**
loop below was nested inside it and ran once per sheet:

```
for (size_t si = 0; si < nsheets; ++si) {   // sampling loop
    for (int j...) for (int i...) { ... }   // <- only the INNER loop closed
    // (the sampling loop's own '}' was missing)
    for (size_t si...) for (int j...) for (int i...) { ...push anchor... }
```

The **surface** count looked correct because the surface loop has its own
headers — so the only visible symptom was a bloated anchor blob. That is
why it survived: the count that would have exposed it was the one count
nobody was checking.

**How it was found** — the counts were *mathematically impossible* (2 sheets
but 4 anchor blocks). A return-address trace showed the same call site
entering the block twice with stale anchors already in the vector, which
ruled out a stale build and pointed at loop nesting. A brace-depth analysis
of the source located the imbalance exactly. (Guessing at braces by eye cost
several wasted iterations; writing the depth analyzer was what solved it.)

**Fix** — close the sampling loop before the anchor loop, and give every
loop header its own explicit brace so the structure cannot silently nest
again. `out.anchors` is now exactly `sheets × n²`.

**Evidence** — `tests/validate_closed_body.cpp` (21 checks) asserts
`anchors == sheets × n²` at grid_n 16/32/48, and that every anchor is
either referenced by a surface or an out-of-support placeholder.

**The lesson** — when a count is "obviously 2×", the loop structure is
wrong, not the arithmetic. Write a brace-depth check before you start
editing braces by hand.

---

## 31. The closed-body rule was taught wrong — and measured wrong

**Symptom** — a "closed sphere" built from an upper and a lower hemisphere
came out with a gap, or closed geometrically but rendered inside-out (the
lower sheet's normals still pointed up, so it drew as a hollow shell).

**Cause** — the doc-learner system prompt taught **two different wrong
rules**: the prose said "origin z = -0.5" and the worked example used
`"origin":[0,0,0.4]`. Neither is right. The rule is a consequence of the
sheet's world height:

```
world.z = origin.z + local.z * scale.z      local.z ∈ [0, sz]
  scale.z = +1  ->  sheet spans [origin.z, origin.z + sz]   (UP)
  scale.z = -1  ->  sheet spans [origin.z - sz, origin.z]   (DOWN)
```

A negative z-scale **already** flips the sheet downward from the seam
plane it shares with its parent, so the lower half must keep the **same
origin** and only flip the scale. Offsetting the origin shifts the lower
sheet off the seam (a gap); leaving the scale positive closes the geometry
but turns the sheet inside-out.

**Fix** — the prompt now states the derivation, names both failure modes,
and gives one correct example (`"scale":[1,1,-1],"origin":[0,0,0]`). The
complex-scene gearbox tree was corrected to match.

**Evidence** — `tests/validate_closed_body.cpp` measures all three variants
on the baked anchors:

| variant | z-extent | verdict |
|---|---|---|
| same origin, scale −1 | `[-0.4998, +0.4998]` | **closed + outward** |
| origin −0.5, scale −1 | `[-0.9998, +0.4998]` | gap |
| origin −0.5, scale +1 | `[-0.4588, +0.4998]` | closed but **inside-out** |

**The lesson** — a prompt that teaches a rule is an **assertion about the
implementation**. It must be measured against the baker, not written from
intuition about the shape.

---

## 32. The renderer keyed its GPU anchor cache on a POINTER

**Symptom** — rendering a 4-sheet body and then a 2-sheet body made the
second body draw with the **first body's** anchors. The sphere's
silhouette lost roughly half its area (43,218 → 22,909 background pixels),
and it only happened when another body had rendered first — so the sphere
rendered correctly on its own and wrong in a sequence.

**Cause** — `SplatRenderer` decided "is this a different body?" with
`last_body != &body`. **A pointer is not an identity**: a destroyed body
and a newly allocated one routinely land on the **same heap address**, so
the comparison said "same body" and the GPU kept the previous body's
anchors.

**Fix** — the body identity now includes the record **counts**
(`last_anchor_count` / `last_surface_count`) as well as the address, so a
different-sized body always re-uploads. A body deformed in place keeps its
count but sets `AnchorsDirty()`, so it still re-uploads.

**Evidence** — an isolated reproduction (render a 4-sheet body, then the
sphere) went from 43,210 vs 22,932 (ratio **1.884×**) to 43,210 vs 43,210
(ratio **1.0000×**).

**The lesson** — any GPU-side cache keyed on "which object is this" must
key on **content**, because allocators reuse addresses. A pointer
comparison is only safe if the object's lifetime is strictly longer than
the cache's.

---

## 33. `build_grid` claimed a cell with an unguarded store — a GPU race

**Symptom** — the **march** path was **non-deterministic**: two runs of the
same binary on the same input differed by **1.09% of pixels** (7,534 of
691,200), with only 9 pixels differing by more than 16/255. The other four
paths were byte-identical. It read as faint per-frame noise rather than a
stable artefact, so it was easy to mistake for a rendering style.

**Cause** — `build_grid` in `kernels/surfel_march.cl` claimed a grid cell
with a **plain store** (`grid[cell] = i`). When several surfels land in the
same cell (which is common — the grid is coarser than the surfel spacing),
the winner is whichever work-item happens to write last. That order is not
defined, so the rendered image was not a function of the input.

**Fix** — `atomic_max` against the `-1` "empty cell" sentinel, so the
winner is a function of the data. **Not `atomic_min`**: the sentinel is -1,
and `atomic_min(-1, i)` = -1 for every valid index, so the grid would never
receive a surfel and the march would find nothing.

**Evidence** — all five paths are now byte-identical across two runs, and
the new `render_determinism` CTest gate asserts it permanently. The fix was
**verified negatively** too: reintroducing the racy store makes
`path_march.ppm` differ between runs while the other four stay identical.

**The lesson** — a plain store into a shared buffer is a race unless the
work-items provably cannot collide. "The grid is coarse so collisions are
rare" is exactly the assumption that makes it invisible.

---

## 34. A brightness threshold was standing in for a geometry measurement

**Symptom** — the closed-sphere A/B reported the body was **not closed**,
at a 2.909× side imbalance, for a body whose anchors measured
`[-0.4998, +0.4998]` (perfectly closed, correctly centred, correctly
outward-facing).

**Cause** — the test's `lit` metric is a **luminance threshold**
(`lum > 28`), so it measures how much of the surface faces the key light.
The lower hemisphere and the upper hemisphere are lit very differently: the
below-seam camera read 2.909× *dimmer* than the above-seam camera even
though their **silhouettes were identical** (168,903 vs 168,884
non-background pixels — a ratio of 1.0001×). The test was measuring
illumination and calling it geometry.

**Fix** — added a `nonbg` metric (pixels differing from the modal
background colour) and the A/B now compares **silhouette area**, which is
the correct invariant: a closed sphere is a disc of the same area from any
direction.

**Evidence** — `render_sphere` (CTest) now reports
`side balance (geometry): 1.0002x`.

**The lesson** — pick the metric that matches the question. "Does this
surface exist?" is a coverage question (area); "is this surface lit?" is a
brightness question. Using the second to answer the first produces a
confident, wrong answer.

---

## 35. Not every render path can resolve a hole

**Symptom** — the workspace render gate reported "the torus has a HOLE"
failing, at 0.518 background against a 0.70 threshold, for a torus whose
surface is provably correct (the baker keeps 952 patches at grid 48 with
the centre out of support).

**Cause** — not a bug in the geometry, and not a regression: the gate was
measuring the hole on the **layered splat** path, which is the workspace's
default. The splat-family paths project **SURFELS** to pixels, so a hole
survives only if no surfel lands in it — and a torus's **inner wall** lands
right there. The march path marches **RAYS**, so the gap is preserved.

Measured on the same torus, same camera, same window:

| path | centre-background | hole resolved? |
|---|---|---|
| march | **0.884** | **yes** |
| area-weighted splat | **1.000** | **yes** |
| splat-layered | 0.520 | no |
| raymarch | 0.526 | no |
| classic splat | 0.466 | no |

**Fix** — the gate now measures the hole on the march path (which resolves
it) and asserts **separately** that the workspace's default path still
renders the torus. The control (the orb) runs on the same path as the
torus, so the A/B is apples-to-apples.

**Evidence** — `tests/validate_workspace_render.py`, registered as the
`validate_workspace_render` CTest gate.

**The lesson** — "does this shape have a hole?" is a question about the
**surface**, and the answer depends on the **path** you asked it through.
Before asserting a shape property, check that the path under test is one
that can express it; otherwise the gate reports a correct model as broken.
This is the sibling of fix 34: both are "the measurement answered a
different question than the one being asked".

---

## 36. A wall-clock quit made every pixel measurement irreproducible

**Symptom** — the workspace render gate was **flaky**: it passed on one run
(centre-background 0.894) and failed on the next (0.465), for the same
binary, the same scene, the same camera. The orb control swung too
(0.092 → 0.326 lit).

**Cause** — the gate launched the app with `--auto-quit SECONDS`, and the
app quit on the **wall clock**. But the render is **not frame-stable**: the
march path's adaptive step count is recomputed per frame from the camera's
near/far range, so a clock-defined frame lands on a different *sample*
each run. Measured on the same torus, same camera, same window:

| auto-quit | centre-background | lit |
|---|---|---|
| 4 s | 0.868 | 0.220 |
| 6 s | 0.436 | 0.109 |
| 8 s | 0.496 | 0.120 |
| 12 s | 0.148 | 0.106 |
| 16 s | 0.853 | 0.222 |

The object is identical; the **sample** is not. A gate that samples the
render at an arbitrary frame is a gate that samples noise.

**Fix** — added `Engine::SetAutoQuitFrames(N)` and the workspace's
`--auto-frames N`: the app closes after N **rendered frames**, which is
exactly reproducible. Five consecutive runs at 60 frames produced an
**identical viewport md5** every time. The gate now uses `--auto-frames`,
and it asserts its own reproducibility (C3: the same frame must render the
same pixels).

**A second, subtler cause** — even with a fixed frame count, the gate was
*still* flaky: the first run in a sequence measured 0.761 while the repeat
measured 0.807. The app alone is perfectly reproducible (5/5 identical
md5); the instability came from running the app instances **back to back**
on the same GPU. The previous instance's last frames are still in flight
when the next instance creates its context and competes for the device.
Fixed with a **settle delay** between launches.

**Evidence** — `tests/validate_workspace_render.py` (CTest gate
`validate_workspace_render`), which now passes 21/21 including the
reproducibility check. Plus a focused ad-hoc probe
(`/tmp/hermes-verify-auto-frames.py`, 20 checks, 0 failures) that asserts
the mechanism directly:

- four runs at `--auto-frames 60` → an **identical viewport md5**
  (`e7167a08b7b9`) every time;
- three runs at `--auto-quit 4/6/12 s` → **three DIFFERENT** md5s, which
  is what proves the frame count is doing real work rather than the
  measurement happening to be stable;
- two launches with the settle pause → **agree**;
- the canonical suite → 27/27.

**The lesson** — two layers:

1. A measurement that samples a **continuous** process needs a **discrete**
   stopping point. "Stop after 12 seconds" is not a stopping point; "stop
   after frame 60" is. When a gate is flaky, suspect the *sampling* before
   you suspect the thing being sampled.
2. A measurement that samples a **shared device** needs the device to be
   **idle**. The same binary, same input, same frame — reproducible in
   isolation, different when preceded by another run. If a gate launches
   the app several times, it must let the GPU drain between launches, or
   it is measuring contention, not the code.

---

## 37. The bone hierarchy flattened — pointer arithmetic between unrelated types

**Symptom.** `validate_bone_bake` failed: "the rotor's parent index points at
the SHAFT" — the rotor came out a root instead of the shaft's child. The
skeleton still had the right *number* of bones, the right pivots and the
right lengths; only the hierarchy was gone. A mechanism whose rotor has no
parent cannot articulate, so this was not cosmetic.

**Cause.** `ElementSystem.cpp`, in the BoneRecord emission loop:

```cpp
const size_t bi = size_t(&b - bones.data());
```

`b` is a `const Bone*` — a pointer to a Bone **object**. `bones` is a
`std::vector<const Bone*>` — a vector of **pointers to** Bone. So
`&b` is a `Bone**` and `bones.data()` is a `Bone**`: the subtraction is
pointer arithmetic between two unrelated pointer types. That is undefined
behaviour, and in practice it evaluated to **0 every time** — so every bone
read `parent_of[0]`, the shaft's parent (-1), and the hierarchy silently
flattened into a forest of roots.

**Fix.** Find the bone's index by **searching** the vector for the pointer:

```cpp
size_t bi = SIZE_MAX;
for (size_t i = 0; i < bones.size(); ++i)
    if (bones[i] == b) { bi = i; break; }
const int par = (bi != SIZE_MAX && bi < parent_of.size()) ? parent_of[bi] : -1;
```

**Why the search and not a stored index.** The bone list is pruned by the
coarseness budget, so an index recorded at collect time goes stale the
moment the budget drops a bone. Searching for the *pointer* is immune to
re-ordering — and pointer identity is the one thing that survives a prune.

**Evidence.** `validate_bone_bake` — 25 checks, 0 failures.

**Lesson.** *A pointer into a container is not an index into that
container.* `&b - vec.data()` is only meaningful when `b` is an **element**
of the vector. When the container holds pointers, the element is
`vec.data()[i]`, and the index must be found by comparison. The failure was
silent because undefined arithmetic that happens to yield 0 is a perfectly
valid-looking index.

---

## 38. The coarseness gate asserted the opposite of its own stated invariant

**Symptom.** `validate_coarseness` failed 4 of 45 checks: "lod=1 has MORE
patches than the coarser level" (and lod 2, lod 3), plus "the coarsest level
keeps FEWER bones".

**Cause — two separate things, one real and one not.**

The *patch* failures were the **test** being wrong, not the engine. The
gate's own header states the invariant it means to check:

> B) The level DERIVES the surface resolution: **a finer level bakes more
> patches.** The relationship is monotonic — that is the whole point.

But the loop ran `for (int lod = 0; lod <= 3; ++lod)` — fine to coarse —
and asserted `patches > prev_patches`. Since `GridForLod` maps 0→48 and
3→8, the patch count **falls** as lod rises. The test asserted the
opposite of its stated intent. Fixed by running the loop coarse-to-fine
(`lod = 3 → 0`), which matches the invariant as written.

The *bone* failure was a **stale library**, not a logic error: the probe
was linked against a `libdelirium_engine_lib.a` built before the
`parent_of` prune landed. Rebuilt, lod 3 correctly keeps 1 bone of 2.

**Measured, coarse to fine** (3 sheets, 2 bones):

| lod | grid | patches | anchors | bones |
|---|---|---|---|---|
| 0 (fine) | 48 | 1123 | 6912 | 2 |
| 1 (medium) | 32 | 467 | 3072 | 2 |
| 2 (coarse) | 16 | 95 | 768 | 2 |
| 3 (sketch) | 8 | 14 | 192 | **1** |

**Lesson.** *When a gate fails, first ask whether the gate or the code is
wrong — and check that the binary you measured is the binary you built.*
Both answers were hiding here: a reversed comparison in the assertion, and
a library that predated the fix. Rebuilding before re-measuring is not
optional; the stale-binary trap has now cost this project several rounds.

---

## 39. The bake sampled the shared CANVAS, not each substance's own support

**Symptom.** The inertia tensor of a rod was **25x too small**. The body
looked correct — the patches were there, the extent was right — but its
MASS sat at a single point.

**Cause.** The baker sampled every sheet on the shared canvas:

```cpp
const float lx = (u - 0.5f) * extent;   // extent = 1.0 for the whole tree
```

A substance much smaller than the canvas therefore got almost NO samples.
Measured on a rod (`sx=0.5, sy=0.02`, canvas 1.0, grid 32):

| grid | samples in support |
|---|---|
| 8 | **0** (0.0%) |
| 16 | **0** (0.0%) |
| 32 | 36 of 1024 (3.5%) |
| 64 | 116 of 4096 (2.8%) |
| 128 | 496 of 16384 (3.0%) |

And the samples that DID land were clustered: the anchor histogram showed
**990 of 1024 anchors at x = -0.019** (one bin) and only ~34 spread along
the rod's length.

**Fix.** `ISubstance::Domain(half_x, half_y)` — each substance states the
region it occupies, and the baker stretches that sheet's grid over it:

- `EllipsoidSubstance` → `(sx, sy)` — the footprint IS the domain
- `TorusSubstance` → `(major + minor, major + minor)` — the ring's outer rim
- `GaussianSystemSubstance` → `(3 sigma, 3 sigma)` — where the bump fades
- `PlaneSubstance`, `HeightfieldSubstance` → the default canvas (unbounded)

The domain is **clipped to the canvas**: the canvas is the world the tree
occupies, so a support reaching past it is not represented anyway — and
clipping stops the grid step from getting *coarser* than it was before
(the gaussian's 3 sigma = 0.9 would otherwise stretch the grid over 1.8
units and halve the resolution inside the canvas).

**Measured after:** the rod's extent became 0.935 (was 0.548 — it was
being clipped by the canvas), and `mean x^2` over the mass-carrying anchors
went from 0.000985 to **0.061277**, against the exact ellipse value
`sx^2/4 = 0.062500` — **98% of the analytic answer**.

**Lesson.** *A grid sized to the container samples the container, not the
content.* When the content is much smaller than the container, most of the
grid is spent on empty space and what remains is wherever the content
happens to cross the grid lines — which looks like a shape but is not one.

---

## 40. The inertia counted the baker's PLACEHOLDER anchors as mass

**Symptom.** After fix 39 the rod's mass was spread correctly along its
length, but the inertia was still ~25% of the analytic value.

**Cause.** The baker writes an anchor for **every** grid cell, and the
cells outside a bounded substance's support get a well-formed placeholder
record parked at the bbox centre (the baker's own comment: "purely a
well-formed unused record"). Those placeholders are the **absence** of
skin, but the inertia code counted every anchor with a bone index — so
**97% of the rod's mass sat at the bbox centre**.

**Fix.** Build a `referenced` mask from the surface blob (the anchors the
patches actually point at) and give mass only to those:

```cpp
std::vector<char> referenced(n, 0);
for (const SurfaceRecord& s : body.SurfaceBlob())
    for (int k = 0; k < 4; ++k)
        if (s.anchor_idx[k] < n) referenced[s.anchor_idx[k]] = 1;
```

**Measured after:** the rod's transverse moment went from 0.021067 to
**0.061297**, against the exact ellipse value 0.062800 — **97.6%**.

**Lesson.** *A record that exists is not a record that means something.*
The baker's placeholder anchors are well-formed and validate — they are
simply not part of the surface. Anything that reasons about the body's
EXTENT or MASS must ask the surface which anchors are real, not count the
blob.

---

## 41. The bbox was re-derived with a DIFFERENT sampling than the bake

**Symptom.** The rod's dequantized extent was 0.548 when the analytic
extent is 0.935, and its positions were subtly wrong.

**Cause.** `ElementSystem::Bake` and `ElementSystem::BakeBody` each
computed the quantization box independently — and after fix 39 the baker
sampled each sheet over its **own domain** while `BakeBody` still sampled
the **shared canvas**. Two different grids, two different boxes, and every
stored position was dequantized against the wrong one.

**Fix.** `BakeBody` re-derives the box with the **same** per-sheet domain
(and the same clip to the canvas). The comment now says why the two must
match.

**Lesson.** *Duplicated geometry is two geometries.* The bbox, the
sampling and the quantization must be derived from ONE place. The
deeper fix is for `Bake` to publish its box in `ModelOutput` so there is
only one derivation at all — noted as the remaining structural improvement.

---

## 42. A fidelity gate with an ABSOLUTE threshold across resolutions

**Symptom.** `validate_reconstruction` failed: "body near analytic
ellipsoid (<0.06)" — the measured residual was 0.07524.

**Cause.** The residual is a **discretisation** property, and the gate's
fixed threshold was calibrated at the old sampling. Measured:

| grid | residual |
|---|---|
| 16 | 0.07524 |
| 24 | 0.05435 |
| 32 | 0.04879 |
| 48 | 0.04013 |

The error grows with `r` (it is the RIM, where the ellipsoid turns nearly
vertical and the tangent-plane patch fit chords it) and **converges with
resolution** — the textbook signature of a discretisation error rather than
a defect.

**Fix.** The gate now asserts:
- the residual is **O(cell)** — a fixed multiple of the grid cell, so an
  error that does NOT scale with the cell (a real defect) still fails;
- the residual **CONVERGES** with resolution — the stronger statement, and
  the one that separates a discretisation error from a defect.

**Lesson.** *An absolute threshold is only meaningful at one resolution.*
Prefer a bound expressed in the discretisation's own unit (the cell), plus
a convergence property. A defect does not improve when you add samples.

---

## 43. The patch-fit target sampled the CANVAS while the anchors sampled the sheet

**Symptom.** A baked body showed **radial "dandelion" spokes** splaying out
of it — and the user read it as "only front forward faces are displayed",
because the spokes were all the outward-facing parts and the body's own
surface was lost inside them.

**Cause.** `ElementSystem.cpp:742` evaluated the patch-fit target over the
**canvas**:

```cpp
const float lx = (wu - 0.5f) * extent;      // extent = the whole tree
```

while the anchors were sampled over the **per-sheet domain** (`hx`, `hy`).
Fix 39 changed the anchors to the sheet's own domain but did not change the
target. So each patch was fitted against a *different part of the sheet* than
its own corners occupied, and the fitted displacement was wrong by the
difference between the two domains.

**Fix.** One `SheetDomain(el, extent, hx, hy)` helper, used at **all three**
sites — the anchor sampling, the patch-fit target, and `BakeBody`'s bbox
recompute — so the rule cannot drift again.

**Measured after** (azimuth sweep, 24 angles):

| | before | after |
|---|---|---|
| closed sphere, equator | 1.43x | **1.13x** |
| closed sphere, polar | 1.24x | **1.004x** |
| splat silhouette bbox @az0 | 292x292 | **226x226** (square at every azimuth) |
| dandelion spokes | present | **gone** |

**Lesson.** *A rule expressed three times is three chances to drift.* When a
fix changes one site's rule, every site that shares the rule must be found
by search, not by memory — and the rule belongs in one named helper.

---

## 44. Two of my own probes measured the wrong thing

**Symptom.** The closed-sphere A/B reported the below-seam view **2.909x
dimmer** than the above-seam view, which read as "the sphere is not closed".

**Cause.** The metric was `lit`, and `lit` is a **luminance threshold**
(`lum > 28`). A surface turned away from the key light counts as *absent* —
so the measurement was of **illumination**, not geometry. The world-space key
light is correct physics, but it made the coverage curve track the light
rather than the shape.

A second probe, the azimuth sweep, had the same class of error in a
different disguise: it orbited in the **XY plane with `up = +Y`**, which is
**parallel to the view at azimuth 90** — a gimbal-lock artifact that looked
exactly like a cull.

**Fix.**
- The A/B compares **silhouette area** (pixels differing from the modal
  background), which is the correct invariant: a closed sphere is a disc of
  the same area from any direction. Result: **1.0002x**.
- The sweep's orbit plane is now perpendicular to the view axis, and the
  gate carries a **control** — a half dome, which must collapse when seen
  from below.

**And a third trap, in the control itself.** The control swept elevation only
from **+70**, so the camera never went *below* the body and the dome could
not collapse. The control therefore could not fail, which made it worthless
as a control.

**Lesson.** *A control that cannot fail is not a control — and a metric that
answers a different question than the one asked will fail in the direction
that looks most like the bug you are hunting.* Two of my three "the engine is
broken" conclusions this session were measurement errors.

---

## 45. A hard-coded threshold in a control rotted when the geometry improved

**Symptom.** `validate_azimuth_sweep` failed on its control: "the half dome
collapses when viewed from below (spread 1.3920x)" against a required
`> 1.50x`.

**Cause.** The threshold was a **guess** about the dome's geometry, written
before the multi-layer bake and the rim margin existed. After those landed,
the dome's real polar spread is **1.39x** — the geometry got *better*, and
the constant threshold did not move with it.

**Fix.** The assertion is now a **relation**: the control must be at least
**20% worse than the closed body it controls for**, comparing the two
*measured* spreads (`half_polar > 1.20 * closed_polar`). A constant
threshold rots the moment the geometry improves; the relation cannot.

**Lesson.** *Prefer an assertion that states the invariant over one that
states a number.* The invariant here is "the control is measurably worse
than the body" — that is what makes the sweep a valid detector. The number
1.50 was never the point.

---

## 46. A square grid over an annulus wastes the cells that matter

**Symptom.** The torus's tube was only **6.1 grid cells across** — below the
8 needed for a clean inner rim — and its patch count was **612 against the
sphere's 3250** on the same grid.

**Cause.** The baker's grid is a square of `N x N` cells over the
substance's domain. A torus's support is an **annulus**, so the square spends
most of its cells on the hole. What matters is the cells that land on the
**tube**, and the tube is the minority of the square.

The grid step over the 0.92-wide square at N=48 is 0.0196, and the tube is
0.12 wide: `0.12 / 0.0196 = 6.1 cells`.

**Fix.** `ISubstance::SamplingRefinement()` — a substance can ask for a
finer grid when its own shape needs one. The unit is "how many cells must
span this substance's smallest feature".

```cpp
// TorusSubstance
float SamplingRefinement() const override {
    const float need = 8.5f * (major_ + minor_) / std::max(1e-6f, minor_);
    return std::max(1.f, need / 48.f);
}
```

The factor is the **MAX over the sheets** (the grid is global — one `N x N`
block per sheet), and it scales with the caller's coarseness budget rather
than overriding it. The workspace's fat torus (minor 0.12) needs **no**
refinement; the thin one (minor 0.06) gets a 28% finer grid.

**Measured after:** N 48 → **64**, tube **6.1 → 8.2 cells**. The closed
sphere's azimuth spread is **1.020x** and its polar spread **1.016x**.

**Lesson.** *A grid sized to the bounding box spends its cells on the box.*
When the content is a fraction of its container, ask the content how fine it
needs to be — the substance knows its own smallest feature, and the baker
does not.

---

## 47. The gate asserted against a grid size the baker is free to change

**Symptom.** The same torus-tube assertion failed at exactly 8.0 cells
against a `>= 8.0` requirement — the comparison was on the wrong side of a
floating-point rounding.

**Cause.** Two things at once. The assertion **hard-coded** `N_GRID = 48`,
but the baker now *refines* the grid, so the requested `grid_n` is not the
grid that was used — the gate was asserting against a number the baker is
free (and expected) to change. And the target was exactly 8.0, which lands
on 7.99... after the baker rounds to an integer N.

**Fix.** Both sides.
- The gate **derives** the grid from the baked body: one `N x N` block per
  sheet means `N = sqrt(anchor_count)`, which is ground truth rather than an
  assumption.
- The requirement is **8.5 cells**, so the invariant is true rather than
  borderline.

**Lesson.** *A gate that hard-codes a value the implementation is allowed to
choose will fail for the right reason and be fixed for the wrong one.* Derive
the value from the artifact.

---

## 48. The field layer's first three failures were all in the TEST (2026-11-08)

G11 (material properties → magnetic / electric / radiative field physics)
landed with a new gate, `validate_field_properties` (74 checks). Its first
run failed **5 of 67**. Every one of the five was a mistake in the TEST or
in my stated expectation, not in the implementation — and each is a lesson
worth keeping, because each would have "fixed" correct code.

**1. The expected number was wrong by 10^6.** The gate asserted copper's
magnetic diffusion time at 1 cm should be ~7e-9 s. The formula
`tau_m = mu * sigma * L^2` with `mu_0 = 1.26e-6`, `sigma = 5.8e7`,
`L = 1e-2` gives **7.3e-3 s** — the code was right and my expected value
was not. A centimetre of copper excludes a changing field in about **7
milliseconds**, not 7 nanoseconds.

**2. The test was accidentally testing the DEFAULT.** The blackbody test
wrote `const FieldProperties black;` — but the default `emissivity` is
**0.9**, so it was measuring a 90% emitter and comparing it to a perfect
one. The gate now constructs the blackbody explicitly. *(Lesson: a test that
means "the extreme case" must SAY so. A default is never the extreme.)*

**3. The JSON schema was wrong.** The fixture wrote `"bones": [...]`; the
schema is `"bone": { ... }` — **singular**, one bone per element, with
`head` and `tail` REQUIRED and the parent given by NAME. The parser
silently ignored the unknown key, and the test reported "the tree did not
produce two bones" — which reads exactly like a parser bug.

**4. A material alias did not resolve.** `"A36 steel"` → `a36steel`, and
the prefix rule required the character after the alias to be non-alphabetic
— but `s` is a letter. The rule now accepts a **digit** after the alias (a
grade continues the token: `a36`) and falls back to a longest-alias-first
pass for compounds (`absplastic housing`).

**5. `absorptivity` is a METHOD, not a field.** The implementation tried to
assign `out.lumped.absorptivity = ...`, which does not compile. The
resolution is not just to add the parentheses: **the lumped medium's
absorptivity is not stored at all.** Kirchhoff's law makes the mixture's
absorptivity equal the mixture's emissivity, so the derived rule already
produces the right number — storing it separately would give the struct
two sources of truth for one quantity.

**Lesson.** *A gate that fails on its first run is telling you about the
gate.* Four of the five failures here would have been "fixed" by editing
correct code, and each would have baked a wrong number into the suite.

---

## 49. Permeability and conductivity are INDEPENDENT axes (2026-11-08)

The field layer's material presets are chosen so that **no two are
identical in every axis**, and so that the two "obvious" pairings are
demonstrably WRONG:

| material | permeability | conductivity | why it is there |
|---|---|---|---|
| copper | **1** (not magnetic) | **5.8e7** (best) | a great conductor that is NOT ferromagnetic |
| mu-metal | **20000** (most magnetic) | 1.1e6 (poor) | a great magnetic shield that is a POOR conductor |
| mild steel | 200 | 1.0e7 | ferromagnetic and conductive, but worse at both than copper/mu-metal |
| ABS | 1 | **0** | an insulator and not magnetic |
| water | 1 | 5e-4 | a HIGH permittivity (80) and a weak conductor |
| neodymium | 1.5 (recoil) | 8e5 | a permanent magnet — a linear mu_r is a poor model, and the recoil value is the honest one |

**Why this matters.** A layer that derived permeability from conductivity
(or treated "material" as one number) would make mu-metal and copper
interchangeable — and they are opposite in every field property. The gate
asserts the independence explicitly, and also asserts that **every pair of
presets differs in at least one axis** (otherwise one of them is
pointless).

**Lesson.** *Model the physical constants, not the material's name.* The
name is for authoring; the solver reads the numbers.

---

## 50. The material had to be stored as a NAME, not as its properties (2026-11-08)

G12 (the doc-learner at bones) needed a document's "steel housing" to reach
the physics layer. The obvious implementation — store the `FieldProperties`
on the bone — does not fit: the struct is far larger than the frozen
`BoneRecord`'s reserved space.

**What was done.** The bone stores the MATERIAL NAME, NUL-terminated, in
`reserved[6..27]` (22 bytes). The registry stays the single source of
truth for the numbers, so:
- two "steel" bones in two different documents get **identical** constants;
- the material can be **re-resolved** if the registry is extended;
- a document can never state a physically inconsistent material.

**Why the name is truncated rather than rejected.** 22 bytes is 21
characters. A longer name is truncated, because the name is *descriptive*
— the physics comes from the registry, and a truncated name still identifies
the material's first words. (The registry strips separators, so "cortical
bone" fits at 13 bytes and survives intact.)

**Why an unknown material is rejected LATER, not at parse time.** The
parser's job is to parse; the registry's job is to know what exists. An
unknown material is a **DOCUMENT problem**, not a JSON problem — so it
belongs in the document pipeline's error log, which is exactly what the log
exists to collect, and it is what lets the escalation prompt tell the model
which word it got wrong. The parser checks the material's *shape* (a
non-string is a schema error) but not its *existence*.

**The gate** (`validate_bone_materials`, 46 checks) covers the whole chain:
document → JSON → parser → baker → frozen record → resolver, and asserts
the name survives byte-for-byte, resolves to the same constants, two
materials stay distinguishable, the mass/radius/stiffness bytes are NOT
disturbed by the new field, an unknown material is rejected, a bone with no
material is not an error, and **every material the prompt teaches actually
resolves in the registry**.

**Lesson.** *A name in the frozen record plus a registry is how a bounded
record carries unbounded data.* The record stays 64 bytes, and the data
stays consistent across every document in the world.

---

## 51. The corpus was impossible to build — the log had no DOCUMENTS in it (2026-11-09)

G13 turned the document learner's log into a retrieval corpus. The first
implementation attempt was a ranking function over the successes. It could not
be written, and the reason was not the ranking.

**Symptom.** `DocAttempt` stored a `document_id` and a `raw_reply` — but not
the document's **text**. "Find the past success most similar to THIS
document" is therefore not computable: an id is a label, not a content. The
only selection rule available was **recency**, which is the weakest possible
one — it picks the example that happened to be logged last, which has no
relationship to whether it teaches anything.

**What was done.** `DocAttempt` gained `document_text` and `domain`, and the
log's on-disk format gained them as **optional trailing fields** (a log
written by an older build still loads — otherwise the corpus would be
unreadable until it was rebuilt, and every accumulated failure would be
lost).

**Lesson.** *Before designing a ranking, check that the thing being ranked
over contains the thing being ranked BY.* The interesting defect here was not
"the ranking is bad" but "the ranking has no input". A feature that sounds
like it is about algorithms is sometimes about data that was never captured.

---

## 52. The corpus must reach the FIRST attempt, not only the retry (2026-11-09)

**Symptom.** The learner's escalation prompt already carried the failure log
and the recent successes. The retrieval corpus was initially wired into the
same place. That would have left the system a **one-shot generator with a
retry button**: attempt 1 always starts from nothing, and only attempt 2
knows anything.

**What was done.** The **normal** attempt now carries the retrieved worked
examples too — the prior document and the tree that the validator accepted.
The escalation path is then what it should be: the response to the corpus
being insufficient, not the mechanism that does the learning.

**Why this matters more than it looks.** A worked example is worth far more
than a description of one. The retrieval carries the prior document's
**text** and the prior **accepted tree** — the payload — not a summary of
them. The failure log, by contrast, carries a reason, because that is all a
failure has.

**Lesson.** *A learning loop that only teaches on retry is not a learning
loop.* If the first attempt cannot benefit from what the system has already
learned, the corpus is decorative.

---

## 53. Two of the retrieval gate's first failures were the GATE's fault (2026-11-09)

`validate_retrieval` (43 checks) failed twice on its first run. Both failures
were **wrong expectations**, and both are worth recording because the wrong
expectation was itself a false claim about the system.

**Failure 1 — "a gearbox must be more similar to a femur than to a recipe".**
The two scores were **both exactly 0.0**, because neither document shares a
single content word with a gearbox. A lexical-overlap measure **cannot**
distinguish "about a femur" from "about a recipe"; asserting a strict order
between them demanded a discrimination the measure does not have. The
assertion was replaced with the claim that actually matters: both unrelated
pairs sit **below** the related pair, and the related pair scores strictly
between 0 and 1.

**Failure 2 — "an empty corpus produces empty context".** The test called
`InterpretDocument` first — which **succeeds** and populates the corpus — and
then asserted the context was empty. That is only true if the *retrieval*
fails, not if the corpus is *empty*. The two were conflated, and the test was
split into two: an empty corpus, and a corpus holding only a **failure**
(which must also retrieve nothing, because a worked example must be a tree
that passed).

**Lesson.** *A failing assertion is a claim about the system. When it fails,
decide which is wrong — the code or the claim.* Both of these were claims
about what lexical overlap *can* do, and both were wrong in a way that would
have forced the implementation to fake a capability it does not have.

---

## 54. A transport failure is a statement about the MACHINE (2026-11-09)

G14's live escalation run failed **6 of 6 attempts, all at the `transport`
stage** — "the model returned nothing (unreachable, or an empty reply)". The
escalations failed too, which initially looked like a broken ladder.

**It was not.** The ladder never got to test anything:
- The machine had **19 GB of 31 GB RAM used, only 4 GB free**.
- Both the 7B and the 14B (9.9 GB) were resident. They cannot coexist in
  the free memory.
- A bare `"Say OK"` to the 7B took **25.4 seconds**.
- With only the 7B resident, the same request took **1.38 seconds**.

The requests were timing out in the transport before the model ever
produced a tree.

**Why this is worth recording.** In the probe's summary table, a transport
failure and a validation failure look the same — "the document did not
become a body". But they say completely different things: one is about the
machine's memory and load, the other is about the model's reasoning. The
log's stage field is what separates them, and it is the reason the stage
exists.

**Lesson.** *When a model-backed pipeline fails, read the STAGE before
reading the reason.* A transport failure is infrastructure; only a parse or
validate failure is a statement about the model, the prompt or the schema.
Chasing a transport failure as if it were a reasoning failure sends you
looking in the wrong code entirely.

---

## 55. A 20-minute model call in a frame hook freezes the window (2026-11-09)

G15 (the document loop in the workspace) called `InterpretDocument` directly
from `OnFrame()`. The model call takes 20-280 s, so the window would have
been frozen for minutes — which reads as a **crashed app**, not as a slow
model.

**What was done.** The interpretation runs on a **worker thread**. The
worker does the model call, the parse and the validation; the **main**
thread does the bake and the scene insertion, because the scene is not
thread-safe. The corpus gets its own **copy** in the worker and is merged
back when it finishes — sharing one `std::vector` across the two threads
would be a data race.

**Two more thread-safety defects found in the same change:**
- `Logf` appends to a `std::vector` with no lock, and the worker appends
  to it while the main thread paints it. `Logf` now takes a mutex, and
  `Log()` returns a **copy** — a reference would outlive the lock that
  protected it.
- The destructor did not join the new worker. A `std::thread` destroyed
  while joinable calls `std::terminate`, so the app would have **aborted on
  exit** instead of shutting down cleanly.

**Lesson.** *A blocking call in a frame hook is a UI freeze, and a new
thread is a new set of lifetime obligations.* The three defects — the
freeze, the race, and the abort — all came from adding one thread, and all
three would have been invisible in a headless test.

---

## 56. The escalation was taught by a weaker rule than the first attempt (2026-11-09)

After G13 made retrieval the corpus's selection rule, the **escalation**
prompt still took the most *recent* successes. So the first attempt was
taught by similarity and the retry by recency.

**What was done.** The escalation now uses `AsRetrievedContext` too, with
the recency path kept only as a fallback for a cold corpus (where recent
successes beat nothing).

**Lesson.** *Two paths that consume the same data must consume it the same
way.* A retry taught by a different and weaker rule than the original is
not a retry — it is a fresh attempt with extra text, and the extra text
makes it look like a retry while behaving like a worse first attempt.

---

## 57. "Done" cannot mean both "finished" and "never started" (2026-11-09)

G15's document worker used a single `bool doc_thread_done_` to coordinate
between the worker thread and the main thread. It initialised to **true**
("no worker has run yet") and the worker set it to **true** when it
finished ("here is your result").

**Symptom.** Every launch of the workspace reported a **spurious failure**:

```
document: FAILED --
document: corpus is now 0 entries (0 ok, 0 failed)
corpus: saved 0 entries to '/tmp/delirium_doc_corpus.txt'
```

— including launches that never asked for a document, and with an **empty
reason**. It also wrote an empty corpus file on every run.

**Cause.** `PollDocument()` read "done == true" as "apply the result". On
the first frame that was true, so it applied a result that did not exist:
`doc_thread_ok_` was false (its initial value), so the failure branch ran.

**What was done.** A separate `std::atomic<bool> doc_has_result_` — which
starts **false** and means exactly one thing: *there is a result waiting*.
It is stored with **release** ordering by the worker AFTER the result fields
are written, and loaded with **acquire** ordering by the main thread BEFORE
they are read, so the fields are visible.

**Lesson.** *A state flag that can mean two different things will eventually
mean the wrong one.* "Done" and "has result" are different facts, and
collapsing them into one boolean puts an ambiguous state into a state
machine. The atomic's ordering is also load-bearing: without release/acquire
the main thread can observe the flag before the fields it guards are
visible — a race that would look like a random, unreproducible failure.

---

## 58. The model pipeline was fine — the MACHINE was out of memory (2026-11-09)

G14's live escalation failed 6/6 at `transport`. The suspicion was the
client's socket code, and the way to settle it was to compare the client
against the server on the SAME machine in the SAME conditions:

| probe | result |
|---|---|
| `llm_probe` (the real client, streaming) | **OK in 4.91 s** |
| `curl` to `/v1/chat/completions` (the client's endpoint) | **correct SSE stream** |
| `curl` to `/api/generate` | OK in 1.38 s |
| `free -g` | **19 of 31 GB used, 4 GB free** |
| `ps` | `helldivers2.exe` at **100% CPU**, 2 days uptime |

The client was healthy. The machine had no room for two models (the 7B is
8.5 GB, the 14B is 9.9 GB), so Ollama was thrashing them in and out of
memory — a bare `"Say OK"` took **25.4 s**.

After unloading the 14B (free memory **0 GB → 9 GB**), the identical probe
**PASSED**: `attempt 1: OK (276.4s), sheets=2, patches=3766, anchors=4608,
GEOMETRY OK`.

**Lesson.** *When a model-backed pipeline fails, test the transport against
the server directly before touching the client.* The two failures look
identical from the outside — "the model returned nothing" — and one of them
is a bug in your code while the other is a bug in your environment. The
cheapest discriminator is to run the same request with `curl`: if `curl`
works, the client is at fault; if `curl` is also slow, the machine is.

---

## 59. A socket read timeout is not a truncated response (2026-11-09)

The LLM client sets `SO_RCVTIMEO` on its socket, so a `recv` can fail with
`EAGAIN`/`EWOULDBLOCK` — a **timeout** — instead of returning data. That is
the EXPECTED case for a big prompt: the document system prompt is ~2,900
tokens and the server thinks for *minutes* before the first token arrives.

All three read paths treated any failure other than `EINTR` as "the data
has arrived":

| read path | old behaviour | symptom |
|---|---|---|
| headers | break out, report "no response headers" | a transport failure for a perfectly healthy server |
| chunk-size line | break out, set a generic error, then store **`State::Done`** | a close was reported as a **successful empty reply** |
| chunk body | break out, then `body.substr(pos, csize)` on a short buffer | a **truncated** buffer parsed as if complete |

The chunk-size line was the worst: because the state was not already
`Failed`, it stored `State::Done` — so a server that closed mid-response
produced a "successful" response with an empty body, and every caller that
checked `reply.empty()` reported "the model returned nothing".

**The unified rule:** a read **timeout** is **retried** (the server may
simply be thinking); a peer **close** is **named**; an **empty** response is
a **failure** (an empty reply is not an answer); a **partial** answer that
did arrive is **kept** and reported as finished.

**Lesson.** *A timeout is not a disconnect, and a truncated buffer is not a
response.* Every read path must distinguish "no data yet" from "the peer is
gone", and must never parse a short buffer as if it were complete.

**The gate** (`validate_llm_read`, 13 checks) runs **its own local HTTP
server** with modes that stall before the headers, stall mid-chunk, and
close early — a real Ollama cannot be made to misbehave on demand. It is
**proven to catch the bug**: with the old behaviour reintroduced it fails on
exactly "a closed connection must fail".

Two harness traps worth keeping: a thread blocked in `accept()` on a closed
descriptor may never return, so the test **hangs at exit** instead of
failing (give the listening socket a short `SO_RCVTIMEO` so the accept loop
polls); and a silently-failed `bind()` leaves the client connecting to port
0, which looks exactly like a hung server.

---

## 60. The outer bound was SHORTER than the timeout it derived from (2026-11-09)

The probe's outer wait loop was meant to be 30% above the client's socket
timeout, so the CLIENT would always decide when a request was over. The
arithmetic did the opposite:

```cpp
const int max_ticks = (c.timeout_seconds * 13) / 10;   // 100 ms per tick
```

On a 900 s timeout that is 1170 ticks — **117 seconds** — which is
*shorter* than the socket timeout, not longer. The measured model time is
~230 s, so **every request was cancelled by the probe's own loop at 117 s**
and reported as a transport failure.

**The comment described the opposite of what the code did.** The
multiplication has to happen in SECONDS and be converted to ticks
afterwards.

**Lesson.** *Derive a bound in the unit it is expressed in.* "30% above the
timeout" was a statement in seconds; the code computed it in ticks and
never converted. When a constant is derived from another, the conversion
is where the bug lives — and the comment is where it hides.

**The companion defect: the failure path threw away the elapsed time.**
`secs_out` was set only on success, so every failure printed **"0.0s"**.
That number was actively misleading: the requests were dying at exactly
117 s, and "0.0s" made them look like they had failed instantly. The
elapsed time is the single most diagnostic number in a timeout
investigation, and it was the one being thrown away.

→ Class-level version: `procedural-surface-engine` →
`references/llm-retrieval-corpus.md`.

---

## 61. The GPU kernel ported 3% of the work (2026-11-09)

**Goal G16** said: *"`psf_refine` is implemented but never launched. It is
the 38.9 ms/op AMR bottleneck and `refine` is a user-facing tool button."*

**The premise was wrong, and only a fresh measurement showed it.** The
kernel `kernels/psf_refine.cl::split_patches` is a **complete, faithful**
GPU port of the CPU `SplitPatch` — the same 9×9 LSQ grid, the same
quadrant convention, the same quantisation. It is good work. It ports the
wrong 3%.

Measured on the 544-surface torus:

| part | cost | share | the kernel covers it? |
|---|---|---|---|
| `seams.Build` + `RebuildViews` | **0.826 ms** | **98.7%** | **no** — the kernel explicitly defers to the CPU |
| the LSQ fit (4 children × 81 evaluates) | 0.029 ms | 3.4% | **yes** |
| hanging anchors (5 per split) | ~0 | — | no |

Total `SplitPatch`: **0.837 ms**.

**The cause.** `RefineWhere` calls `SplitPatch` in a **loop**, and every
split called `seams.Build(*this)` — a rebuild over **every** patch in a
**growing** surface set. One `RefineWhere` does **42 splits** → 42 full
rebuilds → O(n²). And `seams.Build` is itself super-linear: 1.08 µs per
surface at 544, 1.84 µs at 4,354 (8× the bodies → 13.6× the time).

**The fix was ALGORITHMIC, not a port.** `PSFBody::BeginSeamBatch()` /
`EndSeamBatch()` defer the rebuild to the end of a batch; `SplitPatch` and
`MergeGroup` accumulate their touched patches instead of rebuilding.

**Result: 37.80 → 7.57 ms/op — a 5.0× speedup**, with a **bit-identical**
surface (deviation 0.000e+00) and an identical max seam gap (0.14703% of
the diagonal).

**Why deferring is safe.** `SplitPatch` reads the seam graph for exactly
one thing: `find_existing_mid`, which reuses a hanging anchor a
neighbouring split already created. That is an **optimisation** (deduping
an anchor), not a correctness requirement — the fallback creates a fresh
anchor, which the `AnchorPool`'s position snapping dedupes anyway. The
seam graph is left **consistent** when the batch ends.

**The gate** (`validate_seam_batch`, 21 checks) asserts the deferred body's
surface count, anchor count, **evaluated surface points** and seam metrics
all match the eager path. "Same counts" is not enough: the deferral changes
WHEN the seams are rebuilt, so it can change the stitch values, which
change the surface, which is what the user sees.

**Proven to catch the bug:** with the final rebuild removed, the gate fails
— and the failure is dramatic. The surface tears by **27.3% of the
diagonal** instead of 0.147%, and the seam count drops from 1235 to 1120.
A stale seam graph is not a subtle degradation; it is a destroyed surface.

**Lesson.** *A GPU kernel that ports 3% of the work cannot fix a problem
that is 97% elsewhere.* And the general shape: **a per-item operation that
rebuilds global state is O(n²) in a loop** — the fix is almost never to
accelerate the per-item work, it is to rebuild the global state once.

→ Class-level version: `procedural-surface-engine` →
`references/amr-cost-structure.md`.

---

## How the area-weighted path is measured

| path | lit px | holes | white | bbox aspect |
|---|---|---|---|---|
| classic splat | 16,044 | 173 | 0 | 0.942 |
| layered splat | 20,773 | 150 | 0 | 0.948 |
| **area-weighted** | 16,090 | **155** | **0** | 0.942 |

The area path's lit count is close to classic's (it resolves into the
shared framebuffer, so it occupies the same silhouette) with **fewer
holes** and a **deterministic** result (identical md5 across runs), where
the classic path's per-pixel winner is order-sensitive by construction.

# Still open

- **G13 (the success log as a retrieval corpus) is the next learning goal.**
  The failure log already feeds the escalation prompt; the success log
  should feed the *normal* prompt as a retrieved worked example (the
  nearest prior document → its tree). That turns "accumulates complaints"
  into "improves with use" — few-shot over a local corpus, no neural net
  required.
- **G14** — the escalation path has never been driven against a real
  larger model; it is unit-tested with a stub callback.
- **G15** — the document pipeline is not wired into the workspace UI. The
  `DocLearner` is a library with a gate, not yet a console button.
- **Nothing yet POSES a bone.** The baker emits `BoneRecord`s, the anchors
  carry `bone_index`, the material travels in the record's reserved bytes,
  and the inertia and field layers read them — but no code transforms the
  anchors a bone owns when the bone moves. That is the "live physical
  reparametrization" step, and it is the honest boundary of what is built.
- **Nothing yet SOLVES a field.** The field layer provides the material
  data, the exact relations, and the lumped effective medium. There is no
  PDE solver, and the layer does not pretend otherwise.
- The **closed-body composition rule is not taught to the model yet**. The
  baker now supports it (fix 26), but the document pipeline's prompt and
  validator still need to state the rule that a closed body is an upper
  sheet plus a **flipped sheet at the SAME origin** (`scale:[1,1,-1]`,
  `origin:[0,0,0]`) — a stacked offset does not close it.
- The `ElementSystem` substance vocabulary still has no first-class
  closed-body **helper** — the composition is expressible in the tree, but
  there is no `MakeSphere()` convenience.
- The **area-weighted path** is implemented, verified, and wired into the
  editor (the viewport's path buttons and the app's `--render-path` flag).
- The learned categorizer (stage-1 neural weights) does not exist — the
  procedural generator is still the stand-in.
- The **stair-stepped rim** on baked bodies is fix 14 (the tangent-plane
  chord), not a compositor problem — confirmed by the A/B above.
- `tools/psf_encode.py` / `tools/psf_validate.py` (the CPU/GPU
  bit-compat validator) exist in the proven tree but **not** here.
- `assets/sphere.psf` is the **wrong format** (legacy text PSF) — gallery
  scene N fails with a misleading "MISSING" message.
- Six kernels are implemented but never launched; `psf_refine` is the
  ready-made fix for the 38.9 ms/op AMR bottleneck.
- `kernels/psf_area.cl` is a leftover scratch file (the logic was inlined
  into `psf_splat.cl`); its deletion was blocked by a confirmation gate.
  Nothing references it.
- The **multi-layer bake** now has a dedicated gate
  (`validate_closed_body`, `validate_azimuth_sweep`) — see fixes 30, 43–47.
- The **inertia layer reads the mass-carrying anchors only** — the
  placeholder anchors are excluded (fix 40). What it does not yet do is
  derive *directional* inertia (the tensor's principal axes) as a first
  class result; it reports the diagonalised eigenvalues.
- The **closed body is still two sheets, not one watertight mesh** — they
  meet at a rim that now overlaps (the rim margin) rather than sharing an
  edge. The overlap is deliberate: it is the "average out at the edges"
  rule, and it costs a thin double-surface band at the seam.
