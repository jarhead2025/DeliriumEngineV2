# What each goal achieved — and what it fixed on the way

**Companion to `docs/GOALS.md`** (which states the goals and their "done
when" criteria). This file records the **outcome**: what each goal actually
produced, and — the part that matters most — **the bugs it uncovered along
the way**. Almost every goal turned out to be a *diagnosis* goal rather than
a construction goal: the stated symptom had a different cause than expected,
and the fix was somewhere else.

The number to keep in mind: **the suite went from 22 tests to 46**, and the
engine went from a renderer with a scene that had one visible object to a
docked editor with a live local-LLM → geometry loop.

---

## The shape of the whole effort

Six goals were *re-planned mid-flight* because the measurement contradicted
the premise. Those are marked **[re-planned]**, and in each case the
re-planned version is the one that shipped.

| | goal | achieved | found on the way |
|---|---|---|---|
| G1 | closed-sphere A/B | a symmetric silhouette (1.0002×) | the metric measured **illumination**, not geometry |
| G2 | re-verify 5 render paths | all 5 byte-identical | a **GPU race** in `build_grid` (plain store, not atomic) |
| G3 | workspace torus probe | a permanent gate | the probe measured the **wrong render path** |
| G4 | fixes register | 65 entries | — |
| G5 | skill write-back | 8 reference files | SKILL.md was **over the 100 kB limit** |
| G6 | commit | recoverable state | the repo root is the **parent** directory |
| G7 | bone vocabulary | `bone_index` no longer `0xFFFFFFFF` | the parent index was **pointer arithmetic between unrelated types** |
| G8 | bake the skeleton | bones in the frozen records | mass was a **normalised byte** — 2 kg and 20 kg were indistinguishable |
| G9 | inertia from conservation | centre of mass + inertia tensor | 97% of a rod's mass was **parked at one point** |
| G10 | coarseness as a budget | `lod` → grid + skeleton density | the baker **sampled outside the substance's support** |
| G11 | material → field physics | 4 constitutive relations | — |
| G12 | doc-learner at bones | a document → mechanism → parts | the model's **schema was ambiguous** about a sheet's flip |
| G13 | success log → corpus | live-proven retrieval (85% match) | the log stored **no document text** — retrieval was impossible |
| G14 | escalation vs a larger model | the ladder fires live | the models **could not coexist in RAM** — `keep_alive` was the cause |
| G15 | document loop in the UI | two explicit console modes | the interpretation **froze the window** for minutes |
| G16 | launch `psf_refine` | AMR **37.8 → 7.6 ms/op** | the kernel ports **3% of the cost** — the real cost was O(n²) |
| G17 | bit-compat validator | invariant #2 now has a gate | the CPU and GPU **differ by construction** (`fma` vs `+`) |
| G18 | `assets/sphere.psf` | a valid closed-sphere binary | the **quantisation box is the coordinate system**, not metadata |
| G19 | probes into `tests/` | 45 gates | the area path **averaged across depths** |
| G20 | the closed body's representation | the overlap is **provably safe** | the rims are **exactly coincident** — the seams cannot cross |
| G21 | the support-aware grid | the polar torus wastes **3.5%** (was 38%) | the angular axis **duplicated its endpoint** |

---

## G1. Rebuild + run the closed-sphere A/B render

**Achieved.** A geometric silhouette balance of **1.0002×** — a sphere is a
disc of the same area from any direction.

**Found on the way.** The first version of the A/B compared *luminance*
between an above-seam and a below-seam view and read **2.909×**. The body
was fine: `lit` is a luminance threshold, so a surface turned away from the
key light counted as **absent**. The invariant that is actually correct is
the **silhouette area**, not the brightness.

> **A metric that answers a different question than the one being asked is
> worse than no metric** — it produces a confident wrong answer.

---

## G2. Re-verify all five render paths on the current binary

**Achieved.** All five paths (march, raymarch, splat, splat-layered,
splat-area) verified on the current binary, each with its own banner, and
**byte-identical across two runs**.

**Found on the way.** The march path differed by **1.09% of pixels** between
runs. `build_grid` claimed a grid cell with a **plain unguarded store**
(`grid[cell] = i`), so when several surfels landed in one cell the winner
was whichever work-item wrote last. Fixed with `atomic_max` against the `-1`
empty sentinel — *not* `atomic_min`, which would keep `-1` forever and leave
the grid empty.

> **A GPU cache keyed on "which object is this" must key on content, not on
> a pointer** — allocators reuse addresses.

---

## G3. Re-run the workspace torus probe

**Achieved.** `tests/validate_workspace_render.py` — a permanent gate that
drives the real app and reads the composited window pixels.

**Found on the way.** The probe asserted on the **status bar** (a UI element
that never reaches the log) and it measured the torus's hole on the
**layered splat** — which projects *surfels*, so the torus's inner wall fills
its own hole. Measured across all five paths:

| path | centre-background | hole? |
|---|---|---|
| **march** | **0.884** | **yes** |
| **area-weighted** | **1.000** | **yes** |
| splat-layered | 0.520 | no |
| raymarch | 0.526 | no |
| classic splat | 0.466 | no |

Also: the probe quit on the **wall clock**, but the render is not
frame-stable. Added `Engine::SetAutoQuitFrames(N)` + `--auto-frames N`.

> **Sample at a discrete stop, and let a shared device drain.** A wall
> clock lands on an arbitrary frame; a back-to-back launch competes with the
> previous instance's frames still in flight.

---

## G4. The fixes register · G5. the skill write-back

**Achieved.** `docs/FIXES_REGISTER.md` — **65 entries**, each with the
symptom, the cause, the fix, the evidence, and the lesson. The
`procedural-surface-engine` skill — **8 reference files**, SKILL.md reduced
from over 100 kB to ~34 kB.

**Found on the way.** SKILL.md was **already over the size limit** — so the
new material could not be written at all. Two large blocks were moved into
reference files rather than dropped, because the material was worth keeping.

---

## G6. Commit

**Achieved.** Recoverable state — 20+ commits.

**Found on the way.** The **git root is the parent directory**
(`/home/josias/Desktop/Tartarus`), which holds a second project
(`gameengine_compiled`). A plain `git add -A` from the project directory
**swept 158 sibling files** into the commit. The build directory was also
tracked, so every build churned the index. Both were undone in separate,
scoped commits.

---

## G7. A bone/articulation vocabulary · G8. bake the skeleton

**Achieved.** The element tree can express a bone (`head`, `tail`, `bend`,
`radius`, `mass`, `parent`), and the baker emits real `BoneRecord`s with a
correct parent hierarchy. `bone_index` on the anchors is no longer
`0xFFFFFFFF`.

**Found on the way.** The parent index was computed by **pointer arithmetic
between unrelated types** — `&b - bones.data()`, where `b` is a `Bone*` and
the vector holds `Bone*`. It always yielded 0, so **every bone inherited
bone 0's parent**: the hierarchy was flat.

> **A pointer difference is only a difference if both pointers point into
> the same array.** `Bone* - Bone**` compiles and is meaningless.

---

## G9. Inertia from conservation laws

**Achieved.** A centre of mass and an inertia tensor computed from the
anchor mass distribution — and it matches an **analytic rod** to **0.23%**.

**Found on the way.** Two real defects: mass was a **normalised byte**, so a
2 kg housing and a 20 kg frame were indistinguishable; and **97% of a rod's
mass was parked at a single point** — the baker wrote out-of-support
anchors as well-formed records at the bbox centre, and the mass calculation
counted them.

> **A record that exists but represents nothing is worse than a missing
> record** — it is silently counted.

---

## G10. Coarseness as an explicit budget

**Achieved.** The tree carries an explicit `lod`, and it *derives* both the
surface resolution and the skeleton density. The doc-learner teaches the
model to choose an abstraction level rather than a pixel count.

**Found on the way.** The baker sampled each substance over the **whole
canvas**, so a substance much smaller than the canvas got almost no samples
— a rod on a 1.0-wide canvas at grid 32 got **36 of 1024** samples in
support, and **990 anchors** sat at the bbox centre. The body's *mass* was
then 25× too small.

---

## G11. Material properties → field physics

**Achieved.** Per-bone material properties (permeability, permittivity,
conductivity, emissivity) with the exact relations: `B = μ·H`, `D = ε·E`,
`J = σ·E`, `P/A = εσ_SB·T⁴`, `τ_mag = μσL²`. The presets are chosen to make
the independence visible: **copper is the best conductor and is not
magnetic; mu-metal is the most magnetic and is a poor conductor.**

**What it does NOT do.** It provides data and relations, **not a PDE
solver** — and it says so.

---

## G12. Re-point the doc-learner at bones

**Achieved.** A technical document is now interpreted as a **mechanism and
its parts** — a gearbox document becomes a shaft, a housing and a rotor with
articulations, not four overlapping ellipsoids.

**Found on the way.** The model's schema was **ambiguous about how a sheet
is flipped**: it used a *negative half-extent* (`"sz": -1`) to mean "this
sheet points downward". The schema already had the right mechanism
(`"scale":[1,1,-1]`), so the error message now **names the correct field**
instead of just saying "must be > 0".

> **An error message that only says "you were wrong" makes the model fail
> the same way twice. One that names the correction makes the next attempt
> succeed.**

---

## G13. The success log → a retrieval corpus

**Achieved.** The learner's log became a **corpus**: successes are
retrieved by vocabulary similarity and offered to the **first** attempt, not
only the retry. **Live-proven**: the second gearbox document retrieved the
first at **85% similarity** and produced a valid body — and did it **faster**
(345.8 s vs 582.0 s).

**Found on the way.** The log stored a `document_id` but **not the document's
text** — which made "nearest prior document" *not computable*. An id is a
label, not a content. The corpus needed the **text** before it needed the
ranking.

Also: **without a stopword list**, two unrelated documents score high on
"the"/"of"/"and" alone, and retrieval degenerates into "everything is
equally similar".

---

## G14. Escalation against a real larger model

**Achieved.** The ladder fires live: the local attempt finishes, the 7B
unloads, the 14B loads and answers. Proved by watching `/api/ps`.

**Found on the way.** The claim "the 14B needs ~10 GB and cannot coexist
with the 7B" was **wrong** — the ladder is *sequential*. The real cause was
Ollama's default **`keep_alive` (5 minutes)**, which holds the local model in
RAM *after* it answers. And `keep_alive: 0` is **ignored** on
`/v1/chat/completions` but **honoured** on the native `/api/generate`.

Then the framing bug: the native endpoint's **framing depends on the reply
size** — a short answer is `Content-Length`, a long one is
`Transfer-Encoding: chunked`. A client handling only one fails on *half* the
replies, as "the model returned nothing".

> **A slow model and a failed model look identical from outside.** A live
> ladder test needs a document that produces a **wrong** answer, not a slow
> one.

---

## G15. The document loop in the workspace UI

**Achieved.** The console has **two explicit contracts** — `semantic` ("make
a torus": the prompt *is* the spec) and `document` (a file *is* the spec,
and the corpus, the closed-body rule, the error log and the escalation all
run). A live corpus readout, and `--document` / `--doc-domain` /
`--escalate-model` / `--doc-corpus`.

**Found on the way.** Three defects, each found by looking at the running
app rather than at the code: the interpretation **froze the window** for
minutes (a model call in a frame hook → a worker thread, with the bake on
the main thread because the scene is not thread-safe); a **data race** on
the console log; and `std::terminate` on exit (the destructor now joins the
worker). A fourth, found by *reading the pixels*: every launch reported a
spurious `document: FAILED` — the state flag meant **both** "finished" and
"never started".

---

## G16. Launch `psf_refine` — **[re-planned]**

**Achieved.** The AMR path is **5× faster** (37.80 → 7.57 ms/op) with a
**bit-identical** surface (deviation `0.000e+00`, the same 0.14703% max seam
gap) and the same final topology.

**Found on the way.** The goal said "`psf_refine` is implemented but never
launched". The kernel is a complete, faithful GPU port of the CPU
`SplitPatch` — but it ports the **wrong 3%**:

| part of a 0.837 ms `SplitPatch` | cost | share | the kernel? |
|---|---|---|---|
| `seams.Build` + `RebuildViews` | **0.826 ms** | **98.7%** | **no** |
| the LSQ fit | 0.029 ms | 3.4% | **yes** |

`RefineWhere` called `SplitPatch` in a loop, and each split rebuilt the seam
graph over a *growing* surface set — **42 rebuilds per call**, O(n²). The
fix was **algorithmic** (`BeginSeamBatch`/`EndSeamBatch` defer the rebuild to
the end), not a port. The kernel was correctly left unwired.

> **A per-item operation that rebuilds global state is O(n²) in a loop. The
> fix is almost never to accelerate the per-item work — it is to rebuild the
> global state once.**

---

## G17. The CPU/GPU bit-compat validator

**Achieved.** ARCHITECTURE.md **invariant #2** — "`Patch::Evaluate` must be
bit-compatible with `psf_splat.cl`" — now has a gate
(`tests/validate_bit_compat.cpp`, 18 checks).

**Found on the way.** The invariant as written was **wrong**: the kernel's
`deform_h` is a chain of **`fma`** (one rounding) while the CPU mirror uses
plain `*` and `+` (each rounds separately). They differ in the last bits
**by construction**. So the real invariant is *"they agree within a bound
derived from the representation's resolution"* — and the gate **measures** the
delta rather than assuming it.

Then the gate's own audit was **too weak**: a whole-file grep for `c[7]`
only proves the name is *mentioned*. It now isolates the `deform_h` **body**
and requires each term to be **used there**.

> **A gate that has never failed is not evidence.** Break the thing and
> watch the gate catch it — in every direction, including the one the gate
> does not execute.

---

## G18. `assets/sphere.psf`

**Achieved.** A valid binary closed sphere: 4,608 anchors / 3,666 surfaces,
z-range ±0.4998 against an x-span of 1.0000 (ratio **0.9995**). The gallery
reports **"3 ok, 0 failed"** where scene N previously failed with a
misleading **"MISSING"** — the file was present, just the wrong *format*.

**Found on the way.** The tool **recomputed the bounding box** from the
anchor positions and wrote that — shrinking every stored position by
**14.7%**, turning a sphere into an oblate ball while the counts still
matched.

> **The quantisation box is the coordinate system, not metadata.** Every
> position in the file is stored *relative to it*; changing it reinterprets
> every anchor.

---

## G19. The ad-hoc probes into `tests/`

**Achieved.** Every render probe is now a permanent CTest gate — the suite
went from **22 to 45**.

**Found on the way.** The **area-weighted** path's dither was **4.40** —
worse than the classic path's 3.40. Splitting the dither into interior and
boundary pairs showed it was *better* at the boundary and *worse* in the
interior, which pointed at **depth mixing**: the accumulator averaged every
sample projecting onto a pixel, including the torus's **far wall**.

The first fix was **wrong**, and the measurement said so — an in-kernel depth
gate made the interior dither *worse* (4.437 → 4.522), because the winner is
only known after every sample is offered, so a far sample arriving **first**
passes. The right fix is **two passes**.

> **A gate that reads a value another stage computes must wait for that
> stage.** Comparing against a value that is still being written is not a
> conservative check — it is a check against a *moving target*, and it admits
> exactly the cases you wanted to exclude.

---

## G20. The closed body's representation — **the overlap is provably safe**

**The decision** (the user's, 2026-11-10): *"we can save compute power by
having overlapping surfaces where their seams never cross, when computing
both they average out."* So G20 is **not** "eliminate the overlap" — it is
**prove the overlap is safe**. That is now done.

**Achieved** (`tests/validate_rim_overlap.cpp`, 10 checks at two grid
resolutions):

| property | measured |
|---|---|
| the two rims are **coincident** | nearest-neighbour distance **0.000000** |
| their **normals agree** | worst dot **1.0000** |
| their **colours match** | **0** mismatches |
| the cost is bounded | **13.9%** / **12.8%** of the anchors |

**Why it works, mechanistically.** The rim is a flat annulus at `z = 0` on
**both** sheets. The flip is in **Z**; the rim is in the **XY plane** — so
the flip does **not** mirror the rim. Both sheets' rims carry the **same**
normal (the footprint's outward gradient), which means the overlap is
shaded as **one continuous surface**, and the averaging the user describes
averages two surfaces that mean the same thing.

**The depth tie, and why it is benign.** Two coplanar surfaces at the same
depth are exactly the case the depth-tie diagnostic (fix 33) exists for: if
they carried different colours, the `atom_min` winner would be arbitrary.
They carry the **same** colour, so the tie is between two surfaces that
would render **identically** — the winner is irrelevant.

**Found on the way.** The first version of the gate reported *"the rims do
not overlap"*. They did — the **classification** was wrong: it bucketed the
seam anchors by the **sign of z** (upper if `z >= 0`, lower if `z < 0`), but
**both** sheets' rims are at *exactly* `z = 0`, so every rim anchor landed in
the "upper" bucket and the lower bucket was empty.

> **A measurement can be wrong without the thing being wrong.** The
> difference is whether you checked the *classification* or only the
> *number*.

---

## G21. The support-aware grid, generalised

**Achieved.** A substance can now declare that its support is an **annulus**
(`IsPolar()` / `PolarRadius()`), and the baker samples it in **polar
coordinates** — the grid's `i` around the axis, its `j` across the tube. The
grid is still N×N; the cells are just *placed* on the annulus.

**Measured before** (the waste, at grid 48 on a 1.0 canvas):

| substance | in-support | wasted |
|---|---|---|
| plane / gaussian | 100.0% | 0.0% |
| ellipsoid 0.5 | 85.4% | 14.6% |
| rod 0.5 × 0.02 | 75.3% | 24.7% |
| **torus 0.32/0.12** | **62.0%** | **38.0%** |
| **torus 0.40/0.06** | **34.7%** | **65.3%** |

**After:** the fat torus wastes **3.5%**, the thin one **2.9%**, and the
ellipsoid — which did not opt in — is **unchanged** at 16.3%.

**Found on the way.** Three bugs, and one *gate* bug:
1. **The angular axis duplicated its endpoint** — `i = N-1` landed at angle
   2π, the *same point* as `i = 0`, putting an anchor exactly on the tube's
   inner rim (which the hole test correctly flagged).
2. **The patch-fit target had to be re-mapped too**, or the fit targets a
   different part of the sheet than the patch's corners occupy — which is
   **exactly** the "dandelion spokes" bug.
3. **The connectivity had to wrap**, or the tube is split at angle 0 by a
   crack running its whole length.
4. **The gate's own check was wrong twice**: "are the neighbours collinear?"
   counts every ordinary neighbour, and "is the angle one step apart?" counts
   the ordinary step in the other direction too. **Both passed with the wrap
   disabled.** The discriminator has to be the **index**.

> **A gate that has never failed is not evidence — and if breaking the
> thing does NOT make it fail, the gate is what is broken, not the thing.**

---

## The patterns worth keeping

1. **A metric that answers a different question than the one being asked is
   worse than no metric** — it produces a confident wrong answer.
2. **A pointer is not an identity.** Any cache keyed on "which object is
   this" must key on content; allocators reuse addresses.
3. **A per-item operation that rebuilds global state is O(n²) in a loop.** The
   fix is to rebuild the global state once, not to accelerate the per-item
   work.
4. **A gate that reads a value another stage computes must wait for that
   stage** — otherwise it admits exactly the cases you wanted to exclude.
5. **A gate that has never failed is not evidence.** Break the thing and
   watch the gate catch it.
6. **An error message that only says "you were wrong" makes the failure
   repeat.** One that names the correction makes the next attempt succeed.
7. **A slow model and a failed model look identical from outside.** A live
   test needs a document that produces a **wrong** answer, not a slow one.
8. **The quantisation box is the coordinate system, not metadata.**
9. **Before blaming a knob, check whether it reaches the thing.** "The knob
   does nothing" and "the knob is overridden" look identical from outside.
10. **The biggest wins were the goals whose premise turned out to be
    wrong** — G16, G17, G18, G19 and G20 were all "the stated symptom has a
    different cause than expected", and measuring first is what found it.
