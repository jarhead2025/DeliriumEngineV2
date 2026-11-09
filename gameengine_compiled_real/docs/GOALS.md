# GOALS — ordered execution plan

**Set: 2026-11-08.** These are the agreed goals, in the recommended order.

**Rule:** follow this order. **Alter it only when the underlying code or the
reasoning requires it** — and when you do, record *why* in this file (a new
entry under "Order changes"), so the deviation is visible rather than silent.

**Status vocabulary:** `TODO` · `WIP` · `DONE` · `CANCELLED (reason)`

---

## Tier 0 — Restore confidence in what is already claimed

> Nothing here adds capability. It re-establishes evidence for claims that
> were made *before* a core fix, and therefore need re-measuring.

### G1. Rebuild + run the closed-sphere A/B render — `DONE`
The `complex_scene --scene sphere` build was left broken on a scope error
(`m` not declared). It was fixed by hoisting `m_under`/`m_over`, but **the
build was never re-run**, so no pixel-level evidence existed that the closed
sphere renders from both sides.

**Result:** `side balance (geometry): 1.0003x` — a symmetric silhouette from
both sides; `complex_scene: OK`.

Two real bugs were found and fixed while getting there:

1. **The metric measured illumination, not geometry.** `lit` is a luminance
   threshold (`lum > 28`), so a surface turned away from the key light
   counted as "absent" — the below-seam view read 2.909× dimmer than the
   above view for a *perfectly closed* body. The fix adds a `nonbg` metric
   (pixels that differ from the modal background colour) and the A/B
   assertion now compares **silhouette area**, which is the right invariant:
   a closed sphere is a disc of the same area from any direction.
2. **Cross-body anchor contamination (renderer bug, not a test bug).**
   `SplatRenderer` decided "is this a different body?" with a **pointer**
   comparison (`last_body != &body`). A destroyed body and a newly allocated
   one routinely share the same heap address, so the second body was drawn
   with the **first body's** anchors still on the GPU. Measured: a 4-sheet
   gearbox followed by a 2-sheet sphere at the same address rendered the
   sphere with the gearbox's anchors — 43,218 → 22,909 background pixels,
   roughly half the silhouette. Fixed by making the body identity include
   the record COUNTS (`last_anchor_count`/`last_surface_count`), not just
   the address. Isolated repro after the fix: 22,909 → **43,221**.

> Lesson worth keeping: a pointer is not an identity. Any GPU-side cache
> keyed on "which object is this" must key on content, because allocators
> reuse addresses.

### G2. Re-verify all five render paths on the current binary — `DONE`
The missing-brace bug meant every baked body carried **2× the anchors it
should have** (9216 instead of 4608). Every render path consumes baked
bodies, so **all measurements taken before that fix are stale**: the gallery,
the layered-splat figures, the area-path determinism md5, the white-pixel
count.

**Result:** all five paths render, each with its own banner (`march:` /
`psf_raymarch` / `splat:` / `splat-layered:` / `splat-area:`), each reports
the right name, **zero white pixels** on all five, all five md5s distinct (no
silent fallback), and **all five byte-identical across two runs**.

| path | lit | clusters | white |
|---|---|---|---|
| march | 27,584 | 23 | 0 |
| raymarch | 8,600 | 34 | 0 |
| splat | 21,771 | 19 | 0 |
| splat-layered | 28,023 | 42 | 0 |
| splat-area | 22,240 | 12 | 0 |

**One real bug found and fixed here:** the march path was **non-deterministic**
— 1.09% of pixels (7,534 / 691,200) differed between two runs of the same
binary on the same input (only 9 pixels by more than 16/255, so it was noise,
not a stable artefact). Cause: `build_grid` in `kernels/surfel_march.cl`
claimed a grid cell with a **plain unguarded store** (`grid[cell] = i`),
which is a race when several surfels land in the same cell — the winner was
whichever work-item wrote last. Fixed with `atomic_max` against the -1
"empty" sentinel (NOT `atomic_min`: that would keep -1 forever and the grid
would never receive a surfel). All five paths are now byte-identical across
runs.

### G3. Re-run the workspace torus probe — `DONE`
It asserted "the torus has a hole, the orb is solid" on pre-fix geometry.

**Result:** the probe is now a permanent in-tree gate —
`tests/validate_workspace_render.py`, registered as the
`validate_workspace_render` CTest entry — and it **passes 21/21** on the
current binary.

**Three** of its own defects had to be fixed first, and all three were the
*probe* being wrong rather than the engine:

1. **A change-detector.** It required the literal string `"952 patches"` —
   a snapshot of one particular model reply. The model is
   non-deterministic (this run produced a 3-system tree baking to 5,146
   patches; another produced 200), so the count legitimately varies. The
   project's own rule forbids such assertions, so it was replaced with the
   invariant: the tree contains a `torus` substance and bakes to a
   **non-zero** patch count.
2. **The hole was measured on the wrong path.** The gate measured the hole
   on the **layered splat** — the workspace's default — which cannot
   resolve a torus hole at all (it projects SURFELS; the torus's inner
   wall lands in the hole). Measured across all five paths on the same
   torus, camera and window:

   | path | centre-background | hole? |
   |---|---|---|
   | **march** | **0.884** | **yes** |
   | **area-weighted** | **1.000** | **yes** |
   | splat-layered | 0.520 | no |
   | raymarch | 0.526 | no |
   | classic splat | 0.466 | no |

   That is a **path limitation**, not a geometry bug — the torus surface is
   correct (952 patches kept at grid 48, centre out of support). The gate
   now measures the hole on the march path and asserts **separately** that
   the default path still renders the torus. The orb control runs on the
   same path, so the A/B is apples-to-apples:
   torus centre **0.761** background vs orb centre **0.000**.
3. **It sampled a live process at an arbitrary point** — the flakiness.
   The gate quit the app on the **wall clock**, but the render is **not
   frame-stable** (the march path's adaptive step count is recomputed per
   frame), so a clock-defined frame lands on a different sample each run:
   4 s → 0.868, 6 s → 0.436, 8 s → 0.496, 12 s → 0.148, 16 s → 0.853.
   Fixed by adding `Engine::SetAutoQuitFrames(N)` and the workspace's
   `--auto-frames N`: the app closes after N **rendered frames**. Five
   consecutive runs at 60 frames produced an **identical viewport md5**
   every time.
4. **It ran the app back-to-back on the same GPU.** Even with a fixed
   frame count the gate was still flaky (first run 0.761, repeat 0.807):
   the app is reproducible *in isolation*, but the previous instance's last
   frames are still in flight when the next one creates its context and
   competes for the device. Fixed with a **settle delay** between
   launches. Three consecutive gate runs then passed with identical
   measurements (0.761 / 0.198 every time).

The gate now also asserts its own reproducibility (C3: the same frame must
render the same pixels), so a regression that makes the render
frame-dependent fails the gate instead of making it flaky.

---

## Tier 1 — Land what is built

### G4. Update `docs/FIXES_REGISTER.md` — `DONE`
Seven entries added (30–36), each with symptom / cause / fix / evidence /
the lesson:

30. **A missing brace made the anchor loop run once per sheet** — 9,216
    anchors where 4,608 were expected; the surface count hid it.
31. **The closed-body rule was taught wrong** — the prompt gave two
    different wrong rules; the measured rule is now stated with its
    derivation and both failure modes.
32. **The renderer keyed its GPU anchor cache on a POINTER** — a freed and
    reallocated body share an address, so body 2 drew body 1's anchors.
33. **`build_grid` claimed a cell with an unguarded store** — a GPU race
    that made the march path non-deterministic (1.09% of pixels).
34. **A brightness threshold was standing in for a geometry measurement** —
    the `lit` metric measured illumination, not coverage, and reported a
    perfectly closed body as "not closed".
35. **Not every render path can resolve a hole** — the splat family
    projects surfels, so a torus's inner wall fills its own hole.
36. **A wall-clock quit made every pixel measurement irreproducible** —
    and a shared GPU made it irreproducible *again* even after the frame
    count was fixed. Two layers: sample at a DISCRETE stop, and let the
    device drain between launches.

### G5. Skill write-back — `DONE`
`procedural-surface-engine` already carried the determinism gate,
pointer-vs-identity, the `build_grid` race and its sentinel, the
geometry-vs-illumination metric, and the "prove the gate fails on the
buggy code" discipline. This session added:
- `references/render-path-compositors.md` — **which paths can resolve a
  hole** (the measured table, with the surfel-vs-ray explanation).
- `references/probe-measurement-discipline.md` §0 — **sample at a discrete
  stop, and let a shared device drain**, with both measurements.
- `SKILL.md`'s reference index now lists both.

### G6. Commit — `DONE`
Two commits:

- **`480fac7`** — "feat(engine): workspace editor, document-driven
  reconstruction, 5 render paths". 205 files, +53,999 / −2,324.
- **`10832d1`** — "chore: untrack the build tree". 446 files removed from
  tracking; **the files are untouched on disk** (`git rm --cached`, not a
  deletion).

Three things needed care:

1. **The repo root is `/home/josias/Desktop/Tartarus`**, and 158 files from
   a sibling directory (`gameengine_compiled`) were staged before this
   session — not my work. The commit used an explicit pathspec so the
   sibling was excluded. Verified afterwards: the sibling's 2,281 files are
   **byte-identical to HEAD** and match the initial import.
2. **`build_delirium` was tracked** from the initial import (450 files) and
   churned on every build (168 modified artefacts per build — absolute
   paths, compiler probes, object files). Now gitignored and untracked, with
   a `.gitignore` for `build*/`, `__pycache__/` and editor noise.
3. **`git commit --only --amend` re-adds from the working tree**, not the
   index — so amending after `git rm --cached` silently put the build files
   back. The untracking had to be its own commit.

**Verified after committing:** the tree builds and **27/27 tests pass** from
the committed source, and a focused ad-hoc probe
(`/tmp/hermes-verify-gitignore.py`) asserts the `.gitignore` in BOTH
directions — the build tree and python caches are ignored, and **no source
file is ignored** (the load-bearing check: a pattern reaching into
`engine/` or `tests/` would drop real code from every future commit
silently).

---

## Tier 2 — The bones / articulations spine (the substantive architecture)

> The engine already has the substrate: `AnchorRecord.bone_index` +
> `connectivity[4]`, `BoneRecord` (frozen 64 bytes), `ModelOutput.bones`,
> `mode_hints`. **The baker currently writes `bone_index = 0xFFFFFFFF` on
> every anchor — the fields exist and are dead.**

### G7. A bone/articulation vocabulary in the element tree — `DONE`
The tree can now say "this is a femur, it articulates at the knee, it is
0.4 units long".

**`Bone`** (in `ElementSystem.h`): name, parent (by NAME), head, tail,
bend, radius, mass, stiffness, and named joints. A bone is a **rigid
segment** — the degree of freedom the surface hangs from. An element may
carry BOTH a bone and a substance (a rigid part with skin) or a bone and
no substance (a bare axis or joint).

**The JSON schema** (`"bone": {...}`) with head/tail required, and
`"parent"` given by **NAME** — a model writing JSON cannot know an index
that has not been assigned yet.

**Rejected, not degraded:** an unknown parent name, a duplicate bone name,
a cyclic hierarchy, a missing head or tail, a negative mass/radius, and a
stiffness outside [0,1] are all parse errors with a human reason. A
silently re-rooted bone would produce a skeleton that LOOKS fine and
articulates wrongly.

**Gate:** `tests/validate_bone_tree.cpp` — **38 checks, 0 failures**.

### G8. Bake the skeleton into the frozen records — `DONE`
The substrate already existed (`BoneRecord` with parent/bind_pos/bind_rot/
length, `AnchorRecord::bone_index`, `connectivity[4]`,
`ModelOutput.bones`) but the baker wrote `bone_index = 0xFFFFFFFF` on every
anchor — the fields existed and were dead.

Now the baker emits:
- **one `BoneRecord` per bone**, in the same flat order the parser used to
  resolve parent indices, with the bind pose in **world space** (the
  pivot transformed by the element's origin/scale) and the segment's
  length. The authored radius/stiffness/mass ride in the record's reserved
  bytes, so the inertia and field layers can read them without re-parsing.
- **anchors bound** to their sheet's bone.
- **connectivity within a bone only** — two anchors on different bones are
  NOT connected, because they can move independently. An edge across an
  articulation would constrain a degree of freedom that is supposed to be
  free.

**The load-bearing detail:** the parser resolves a parent into an index
using a recursive pre-order walk; the baker collects sheets with a stack.
If the two orders disagreed, a parent index would point at the wrong bone
— and the skeleton would still validate. The sheet's bone index is
therefore found by **pointer**, so the traversals cannot drift.

**Gate:** `tests/validate_bone_bake.cpp` — **25 checks, 0 failures**.

**Suite: 29/29** (was 22 at the start of the bone work).

### G9. Inertia from conservation laws — `DONE`
With a skeleton: mass distribution per bone → centre of mass, and the
inertia tensor → **rotational** and **directional** inertia. Computed from
the anchor mass distribution, not from a mesh.

**Landed:** `engine/include/delirium/physics/Inertia.h` +
`engine/src/physics/Inertia.cpp`.
- `ComputeMassProperties(body, bones, bone_mass, fallback_density, out, why)`
  → the total mass, the centre of mass, and the symmetric inertia tensor
  about the **centre of mass**.
- `MassProperties::InertiaAbout(unit_axis)` — the **directional** inertia,
  `u . I . u`: the "how hard is it to spin about THIS axis" question,
  answered directly from the tensor.
- `MassProperties::PrincipalMoments(moments, axes)` — the eigenvalues
  (Jacobi) and their axes, sorted descending.
- **The mass belongs to the BONE, not the surface.** A bone's authored mass
  is spread EQUALLY over that bone's anchors, so the total mass is
  independent of the tessellation — which is exactly what the coarseness
  budget (G10) depends on.

**Four bugs were found and fixed on the way** (fixes 39–42 in
`docs/FIXES_REGISTER.md`):
1. the bake sampled the shared **canvas** rather than each substance's own
   support, so a small substance got 3.5% of the grid and 97% of its mass
   piled up at one point;
2. the inertia counted the baker's **placeholder** anchors (the well-formed
   "no surface here" records parked at the bbox centre) as mass;
3. `BakeBody` re-derived the quantization box with a **different sampling**
   than `Bake`, so every position was dequantized against the wrong box;
4. a fidelity gate used an **absolute** threshold across resolutions.

**Measured against the analytic answer** (a rod, `sx=0.5, sy=0.02`,
mass 1): the exact uniform-ellipse value for `⟨x²⟩` is `sx²/4 = 0.062500`;
the computed mass-weighted value is **0.061277** — **98%**. The transverse
moment is **0.061297** against the exact **0.062800** — **97.6%**.

**Gate:** `tests/validate_inertia.cpp` — **30 checks, 0 failures**. It
asserts against ANALYTIC values (a rod, a disc), not snapshots, plus
conservation of mass, translation invariance, LOD-independence of the
mass, and that a body with no mass reports a MISSING result rather than a
zero tensor.

### G10. Coarseness as an explicit budget — `DONE`
"Complete in representation, coarse for speed" maps onto two knobs the
engine already has: surface resolution (`grid_n`) and skeleton density.

The tree carries a `coarseness` / `lod` field that **derives** both, so
the model chooses an abstraction level rather than a pixel count.

**Landed:**
- The tree states `"lod": 0..3` (fine / medium / coarse / sketch). An
  out-of-range or non-numeric level is **REJECTED**, not clamped — a
  clamped level silently ignores the model's intent.
- `Element::GridForLod` derives the surface resolution: 48 / 32 / 16 / 8.
- `Element::BoneKeepForLod` derives the skeleton density: 100% / 100% /
  75% / 40%. The deepest bones are dropped first (a leaf joint changes the
  least), and an orphan is re-parented to the dropped bone's parent so the
  hierarchy stays a tree.
- The prune is **side-effect free**: the hierarchy is pruned in a separate
  `parent_of` map, so baking the same tree twice gives the same skeleton.
- `ElementSystem::BakeLod` / `BakeBodyLod` take the level; an explicit
  `grid_n` still wins, because a caller that knows its own budget should
  not have to argue with the model about it.

**Measured** (3 sheets, 2 bones — `lod` → patches / anchors / bones):

| lod | grid | patches | anchors | bones |
|---|---|---|---|---|
| 0 | 48 | 1123 | 6912 | 2 |
| 1 | 32 | 467 | 3072 | 2 |
| 2 | 16 | 95 | 768 | 2 |
| 3 | 8 | 14 | 192 | **1** |

**Gate:** `tests/validate_coarseness.cpp` — **45 checks, 0 failures**.

> **Order note:** G10 precedes G9 on purpose — coarseness decides *what* the
> inertia is computed over. Changing the LOD after computing inertia means
> recomputing it.

### G11. Material properties → field physics — `DONE`
`Material.h` and `ConstitutiveModel.h` exist. Attach per-bone material
properties (density, permittivity, permeability) so magnetic / electric /
radiative fields have something to read.

**Done when:** a body with assigned material properties reports field
responses (e.g. a magnetic permeability) that come from its bones, not from
a global constant.

**LANDED** — `engine/include/delirium/physics/FieldProperties.h` +
`engine/src/physics/FieldProperties.cpp`, with the gate
`validate_field_properties` (**74 checks, 0 failures**).

**What it is.** The *same pattern* as the inertia layer, one layer over:
properties are authored per **BONE**, spread over that bone's anchors, and
read out as field quantities. The mechanical `Material` already carried
stress-strain curves and fluid models; what was missing was the
electromagnetic and thermal constants, and a home for them.

**The constants** (SI, all real published values): permeability `mu_r`,
permittivity `eps_r`, conductivity `sigma` (S/m), emissivity, absorptivity,
specific heat, thermal conductivity.

**The relations** (the exact equations a solver would use, so the solver's
equations can be checked against a reference):
`B = mu·H` · `M = (mu_r−1)·H` · `u_B = ½·mu·|H|²` · `D = eps·E` ·
`P = eps_0·(eps_r−1)·E` · `J = sigma·E` · `P/A = eps·sigma_SB·T⁴` ·
`q_net = sigma_SB·(eps₁T₁⁴ − eps₂T₂⁴)` · `tau_charge = eps/sigma` ·
`tau_magnetic = mu·sigma·L²`.

**What it is NOT:** a field solver. There is no PDE here, and pretending
otherwise would be the plausible-but-wrong output that makes the layer
untrustworthy. It provides the data, the relations, and the lumped
(mass-weighted) effective medium — which is what lets the document pipeline
ask "does this reconstruction behave like the material the document
described?".

**The presets** are chosen so the independence is visible: copper (best
conductor, **not** magnetic), mu-metal (most magnetic, **poor** conductor),
mild steel, ABS (insulator), water (eps_r 80, weak conductor), neodymium
(recoil permeability — a linear mu_r would be dishonest), muscle tissue,
cortical bone.

**Two design decisions worth keeping:**
- **An unknown material is REJECTED, not defaulted.** Defaulting to
  "generic plastic" would assert physics the document never stated — and
  the document pipeline's whole value is that it does not invent.
- **The lumped medium's absorptivity is NOT stored.** Kirchhoff's law makes
  it equal the lumped emissivity, so the derived rule already produces the
  right number; storing it separately would give the struct two sources of
  truth for one quantity.

**Still not done:** nothing yet *solves* a field with these, and nothing
poses a bone so its anchors move (that is the "live physical
reparametrization" step, and it is the honest boundary of what is built).

### G12. Re-point the doc-learner at bones — `TODO`
The prompt currently teaches substances and sheets. Once bones exist, a
technical document is interpreted as *mechanisms and their parts*: a gearbox
document becomes a shaft + housing + rotor with articulations, not four
overlapping ellipsoids. The closed-body section becomes "a bone needs its
opposing surface".

**Now that G11 has landed**, the doc-learner also has a MATERIAL
vocabulary to teach: a document that says "steel housing" should be able to
name the material, and the reconstruction should carry it into the bone's
field properties. The tree schema needs a `material` field on a bone, and
the parser needs to resolve it through `LookupFieldProperties`. **That is
the concrete form this goal takes now.**

---

## Tier 3 — Make the learning loop genuinely good

### G13. Success log → retrieval corpus — `TODO`
The failure log already feeds the escalation prompt. The **success** log
should feed the *normal* prompt as a retrieved worked example (nearest prior
document → its tree). That turns "accumulates complaints" into "improves with
use" — few-shot over a local corpus, no neural net required.

### G14. Exercise escalation against a real larger model — `TODO`
The ladder is unit-tested against a stub and has never run against a second
endpoint.

### G15. Wire the document loop into the workspace UI — `TODO`
A "load document" affordance and a live error-log panel. Today the learner
is a library with a CLI probe, not something usable.

---

## Tier 4 — Engine gaps (cheap and real)

### G16. Launch `psf_refine` — `TODO`
Implemented but **never launched**. It is the 38.9 ms/op AMR bottleneck *and*
`refine` is a user-facing tool button.

### G17. Add the CPU/GPU bit-compat validator — `TODO`
`psf_validate.py` exists in the proven tree but not here. ARCHITECTURE.md
invariant #2 currently has **no gate**.

### G18. Fix `assets/sphere.psf` — `TODO`
It is the legacy **text** format, so the binary loader rejects it and the
gallery's scene N fails with a misleading "MISSING" message.

### G19. Move the ad-hoc probes into `tests/` — `WIP`
The workspace-torus, layered-splat, path-switch and closed-body probes live
in `/tmp` and will be lost. CTest renders nothing, so these are the *only*
render evidence — they belong in-tree as permanent gates.

**Landed so far — the suite is now 33 gates (was 22):**
- `tests/validate_closed_body.cpp` — **21 checks, 0 failures.** Pins the
  measured closure rule: the correct rule spans exactly `[-sz, +sz]` and is
  centred; an offset origin opens a gap; an unflipped lower sheet is
  inside-out; anchors == `sheets * n^2` at three grid sizes; no orphan
  anchors (every anchor is either referenced or an out-of-support
  placeholder).
- `tests/validate_workspace_torus.cpp` — **the workspace's OWN fat torus**
  (major 0.32, minor 0.12) measured down its own axis, with **two controls**:
  the orb must have a LIT centre (so "the centre is dark" cannot mean "nothing
  rendered"), and no surface anchor may lie inside the hole's radius (a
  geometry-level assertion, independent of the renderer). This gate exists
  separately from the azimuth sweep because the sweep's thin torus cannot see
  the fat torus's failure.
- `tests/validate_azimuth_sweep.cpp` — **19 checks, 0 failures.** Orbits the
  camera and asserts the coverage curve is **flat** on two paths, with a
  **control** that proves the sweep can detect the bug. The control's
  assertion is a **relation** (the half dome must be ≥20% worse than the
  closed body), not a constant — a constant rotted when the geometry improved
  (fix 45).
- `tests/validate_inertia.cpp`, `tests/validate_bone_bake.cpp` — the bones
  spine (G7/G8/G10).
- `render_sphere` (CTest) — the closed-sphere A/B, asserting a **geometric**
  silhouette balance (1.0002×).
- `render_paths` (CTest) — all five paths, each with its own banner, plus
  the lit/cluster counts and the white-pixel check.
- `render_determinism` (CTest) — the same scene rendered twice must be
  **byte-identical**. This is the gate that would have caught the
  `build_grid` race automatically; it now fails the moment a render stops
  agreeing with itself.
- `validate_workspace_render` (CTest, a Python gate) — the windowed app:
  the panels paint, the AI loop completes, the torus's hole is open on a
  path that can resolve it. Uses `--auto-frames N` (a **frame** count, not a
  wall clock) so the measurement is reproducible.

**Still to land:** the layered-splat probe.

---

## G20. The closed body's representation — `TODO`

The closed body is currently **two sheets that overlap at a rim** (the rim
margin, fixes 26/43). That closes the gap and kills the knife-edge, but it
costs a **thin double-surface band** at the seam, and the body is not a
single watertight surface.

**Done when:** the seam band is either eliminated (the sheets share an edge
rather than overlapping) or is explicitly accepted as the "average out at the
edges" rule and documented as the cost. The second is a legitimate answer;
the first is not yet attempted.

---

## G21. The support-aware grid, generalised — `WIP`

`ISubstance::SamplingRefinement()` (fix 46) lets a substance ask for a finer
grid. It is currently a **factor applied to a square grid**, so an annular
support still spends most of its cells on the hole.

**Done when:** either the grid can be shaped to the support (an annulus of
cells rather than a square with a hole in it), or the factor is measured
sufficient for every substance in the vocabulary. The current state: the
factor is sufficient for the torus (8.2 cells across the tube) and the
workspace's fat torus needs none.

---

## The Vulkan deferral — triggers, not a date

`docs/VULKAN_ASSESSMENT.md` has the full reasoning. The short version: the
kernels use **no images, no shared memory, no barriers, no subgroups** — a
pure compute pipeline over storage buffers with integer `atomic_min` /
`atomic_add`. So the port is mechanically easy, and the benefit is
**deployment** (no OpenCL ICD), not rendering quality or speed.

**Port when ANY of these is true — not before:**
1. A target platform has **no installable OpenCL ICD**. This is the only
   trigger that makes the port *necessary*.
2. G7–G12 have landed **and** per-bone GPU transforms are measurably the
   limiting cost. A bone palette changes the storage layout, so the backend
   choice must follow it.
3. A specific kernel needs **subgroup ops**.

**Hard prerequisite:** option (1) means reimplementing 13 kernels in SPIR-V
against the CPU mirror. Without the CPU/GPU bit-compat validator (**G17**,
still `TODO`) that reimplementation is **unverifiable**. If the port is
chosen, **G17 moves ahead of it** — that is an order change, recorded below.

---

## Order changes

*(Record every deviation here: what moved, why the code or reasoning required
it, and what it displaced.)*

- **2026-11-08** — G1/G2/G3 raised above G4/G5/G6. Reason: the render-path
  measurements they protect were taken before the anchor-count fix, so
  landing them first would record stale numbers as current.
- **2026-11-08** — **G17 (the bit-compat validator) is promoted to a
  prerequisite of any Vulkan port**, not merely a parallel goal. Reason: the
  port re-implements 13 kernels, and without a CPU/GPU bit-compat gate that
  work cannot be verified. It does not displace a goal — it gates an
  *option*, not a task.
