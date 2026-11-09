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

### G12. Re-point the doc-learner at bones — `DONE`
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

**LANDED** — the gate is `validate_bone_materials` (**46 checks, 0 failures**).

**The chain it closes:**
```
document says "steel housing"
  -> JSON:  "bone": { ..., "material": "steel" }
  -> parser:  Bone::material
  -> baker:   BoneRecord.reserved[6..27] = "steel\0"
  -> resolver: FieldProperties (steel's REAL constants)
```

**The material is stored as a NAME, not as its properties.** The
`FieldProperties` struct is far larger than the frozen record's reserved
space, and the registry stays the single source of truth for the numbers —
so two "steel" bones in two different documents get **identical**
constants, and the material can be re-resolved if the registry is extended.

**The prompts now teach:**
- the schema prompt lists the `material` key and the vocabulary (steel,
  copper, aluminium, mu-metal, neodymium, ABS, plastic, glass, water,
  muscle, bone) — and the gate asserts **every name it teaches actually
  resolves in the registry**, so the prompt cannot teach a material that
  would be rejected;
- the document prompt has a new section, **"READ IT AS A MECHANISM, NOT A
  PILE OF SHAPES"**: give each moving part a bone, name the joints, give
  each part its material — and do not invent either one the document did
  not state.

**Where the unknown material is caught.** The parser checks the material's
*shape* but not its *existence*: an unknown material is a **DOCUMENT**
problem, not a JSON problem, so it belongs in the error log (which is what
lets the escalation prompt tell the model which word it got wrong). The
resolver rejects it with the offending name.

---

## Tier 3 — Make the learning loop genuinely good

### G13. Success log → retrieval corpus — `DONE`
The failure log already feeds the escalation prompt. The **success** log
should feed the *normal* prompt as a retrieved worked example (nearest prior
document → its tree). That turns "accumulates complaints" into "improves with
use" — few-shot over a local corpus, no neural net required.

**Landed.** `DocErrorLog::RetrieveSimilar` / `AsRetrievedContext` / `Similarity`.
The corpus now reaches the **normal** attempt, not only the escalation — if it
only reached the escalation, escalation would be doing all the work and the
system would be a one-shot generator with a retry button.

Design decisions and their reasons:
- **The log now stores the document TEXT and DOMAIN.** They were not there
  before, which made "nearest prior document" *not computable* — a document id
  is a label, not a content. This was the blocking gap, not the ranking.
- **Similarity is Jaccard token overlap** over stopword-filtered words.
  Dependency-free, deterministic, offline, explainable — and therefore
  auditable, which matters for a prompt a human is meant to be able to check.
- **A stopword list is required.** Without it, two documents about different
  machines score high on "the"/"of"/"and" alone and retrieval degenerates
  into "everything is equally similar" — recency with extra steps.
- **Only successes are candidates.** A worked example must be a tree that
  passed validation; teaching from a rejected tree teaches the failure mode.
- **A document never retrieves its own earlier attempt** (that is the
  escalation path's job, and self-retrieval would let a document teach
  itself).
- **Zero-overlap documents retrieve nothing** — noise in a prompt is a cost.
- **The domain prior is a bounded multiplier (1.25×), not an addition.** An
  addition could promote a zero-overlap cross-domain example over a strongly
  overlapping one, which is the opposite of the prior's purpose.
- **The log format gains two optional trailing fields.** A log written by an
  older build still loads — otherwise the corpus would be unreadable until it
  was rebuilt, and every accumulated failure would be lost.

Gate: `tests/validate_retrieval.cpp` — 43 checks, 0 failures. It asserts the
*invariants* (symmetry, boundedness, discrimination, similarity-beats-recency,
failures-never-offered, no-self-retrieval, corpus-reaches-the-normal-prompt,
round-trip, legacy-log compatibility), not snapshots.

**Live, end-to-end — PROVEN (2026-11-09).** The unit gates prove the
retrieval *works*; the live run proves it *helps*:

```
--- [1/2] doc_gearbox.txt ---
    corpus: no similar worked example (cold for this doc)
    attempt 1: OK (582.0s)
    sheets=4  patches=7354  anchors=9216
    GEOMETRY OK

--- [2/2] doc_gearbox2.txt ---
    corpus: 1 worked example(s), best 'doc_gearbox.txt' (85%)
    attempt 1: OK (345.8s)
    sheets=4  patches=7226  anchors=9216
    GEOMETRY OK

valid trees: 2 | baked bodies: 2 | failures: 0 | successes: 2
```

The second document **retrieved the first as a worked example at 85%
similarity**, and produced a 4-sheet, 7,226-patch, 9,216-anchor body that
baked and passed the geometry check — and did it **faster** (345.8 s vs
582.0 s). The worked example is real context, not decoration.

Both documents also produced **4 sheets**, which is the closed-body rule
working: one sheet is a dome, four compose a housing with a rotor, a shaft
and a top.

**Reaching this took four separate real defects**, each of which presented
as the same opaque "the model returned nothing": the machine was out of
memory (not a code bug); a socket read timeout was treated as a truncated
response (three read paths); a close mid-response was reported as a
*successful empty reply*; and the probe's outer bound was **117 s — shorter
than the timeout it derived from**. Only the live run could see any of
them; the unit gates were green throughout. See FIXES_REGISTER entries 54,
58, 59, 60.

### G14. Exercise escalation against a real larger model — `WIP`
The ladder is unit-tested against a stub and has never run against a second
endpoint.

**Wiring landed.** `tests/validate_escalation.cpp` — 36 checks, 0 failures.
It asserts the CONTRACT of the ladder, not a snapshot:
- the escalated attempt goes to the **larger** model, and **only** to it (the
  local model is not re-asked);
- the escalated attempt carries the **failed stage**, the **failure reason**,
  the **error log**, and the **worked examples**;
- the escalation prompt is **self-contained** — it restates the document,
  because a retry that depends on the caller's memory of the previous turn is
  not a retry;
- the escalation is recorded as escalated, and a **successful** escalation is
  logged as a success — so it becomes retrievable corpus, and the ladder is
  cumulative rather than per-document;
- a **failed** escalation does not erase the original failure (both remain
  logged — losing the original would erase the evidence of what the local
  model got wrong);
- with **no** larger model supplied, the escalation re-asks the local model
  with the enriched context — a legitimate, weaker rung.

**The escalation now uses RETRIEVAL, not recency.** It previously took the
most recent successes, while the normal attempt took the retrieved ones. A
retry taught by a *different and weaker* rule than the first attempt is not a
retry — it is a fresh attempt with extra text. The recency fallback remains
for a cold corpus (nothing retrieved), because recent successes beat nothing.

**Live run against a second endpoint — the first attempt FAILED, and the
reason was environmental, not a code defect.** The run failed **6/6 at the
`transport` stage** — "the model returned nothing (unreachable, or an empty
reply)" — including the escalations. A transport failure means the model
never replied; it is not a statement about the tree.

Diagnosis (measured, not guessed):
- The machine had **19 GB of 31 GB RAM used, only 4 GB free**, with
  `helldivers2.exe` pinned at **100% CPU** (a game, running for 2 days).
- The 7B and the 14B models (9.9 GB) were BOTH resident. The 14B could not
  coexist with the 7B in the free memory.
- A bare `"Say OK"` to the 7B took **25.4 seconds** — pathologically slow.
- With only the 7B resident, the same request took **1.38 seconds**.
- The client itself was **verified healthy** in the same conditions:
  `llm_probe` completed a streaming request in **4.91 s**, and a direct
  `curl` to `/v1/chat/completions` (the endpoint the client uses) returned
  a correct SSE stream. So the fault was the **machine's memory**, not the
  socket code.

After unloading the 14B (free memory went from **0 GB to 9 GB**), the same
probe **PASSED**:

```
--- [1/1] A-100 gearbox housing ---
    attempt 1: OK (276.4s)
    sheets=2  patches=3766  anchors=4608
    GEOMETRY OK
valid trees: 1 | baked bodies: 1 | failures: 0 | successes: 1
```

**Lesson: a transport failure is a statement about the MACHINE, not about
the model or the prompt.** The distinction matters because a transport
failure and a validation failure look identical in the summary table, and
only one of them says anything about the reconstruction quality. Read the
STAGE before reading the reason.

**The mitigation is in the client**: `AskModelSync` raises the token budget
to 1400 and the timeout to 900 s (the defaults — 1024 tokens / 180 s — are
not enough for a schema-shaped JSON answer from a 14B on a loaded machine),
and it uses a **fresh client per call** so a config change cannot corrupt a
request already in flight.

**Still open: the ladder's RESCUE behaviour.** Every live run so far has
**succeeded on attempt 1**, so the escalation rung has never actually fired
against a real second endpoint. The wiring is proven by the 36-check gate;
the *rescue* — "a bigger model fixes what the local model could not" — is
not yet demonstrated. Exercising it needs a document the local model
reliably fails (a single-sheet answer is the obvious candidate, since the
document path rejects those).

### G15. Wire the document loop into the workspace UI — `DONE`
A "load document" affordance and a live error-log panel. Today the learner
is a library with a CLI probe, not something usable.

**Landed.** The console has **two explicit contracts**, and the toggle
between them is a button, not an inference:

| | SEMANTIC | DOCUMENT |
|---|---|---|
| the specification | the prompt itself | a file's path |
| the prompt | the element schema | the schema + the document + the retrieved worked examples |
| validation | one sheet is allowed | one sheet is **rejected** (a closed body needs several) |
| bookkeeping | none | the corpus, the error log, the escalation ladder |

**The mode is explicit because the validation differs.** The document path
*rejects* a single sheet; the semantic path allows one. Inferring the mode
from the text would make that rejection unpredictable — the same prompt
could be accepted or refused depending on how it was phrased.

Wired: `LoadDocument` (reads the TEXT, not just the path; refuses a missing
or empty file and **names the path** in the refusal), `InterpretLoadedDocument`,
`SaveDocCorpus`/`LoadDocCorpus`, a corpus readout in the console
(`corpus: N ok / M failed`), and CLI flags `--document`, `--doc-domain`,
`--escalate-model`, `--doc-corpus`, `--auto-document`.

**Three real defects found and fixed while building it:**
1. **The interpretation froze the window.** The first version called
   `InterpretDocument` directly from `OnFrame()`, and the model call takes
   20-280 s — the window would have been frozen for minutes, which reads as
   a crashed app rather than a slow model. It now runs on a **worker
   thread**; the worker does the model call, the parse and the validation,
   and the **main** thread does the bake and the scene insertion (the scene
   is not thread-safe).
2. **A data race on the console log.** The worker appends to `log_` while
   the main thread paints it. `Logf` now takes a mutex, and `Log()` returns
   a **copy** (a reference would outlive the lock that protected it).
3. **`std::terminate` on exit.** The destructor now joins the document
   worker — a `std::thread` destroyed while joinable aborts the process.

Also: the corpus gets its own **copy** in the worker and is merged back
when it finishes, so the corpus still accumulates without being shared
across threads. The editor drops to 10 FPS while the document worker runs
(the same throttle the semantic path uses — the frame rate competes with
the model for the same cores).

Gate: `tests/validate_document_mode.cpp` — 25 checks, 0 failures. It asserts
the two contracts are mutually exclusive, the corpus persists **with its
documents** across a restart, the self-exclusion survives the round-trip,
the corpus accumulates across documents, and a failure is recorded with its
stage.

---

## Tier 4 — Engine gaps (cheap and real)

### G16. Launch `psf_refine` — `DONE` (but not by launching it)
Implemented but **never launched**. It is the 38.9 ms/op AMR bottleneck *and*
`refine` is a user-facing tool button.

**Landed — as an ALGORITHMIC fix, because the premise was wrong.** The
kernel `psf_refine.cl::split_patches` is a complete, faithful GPU port of
the CPU `SplitPatch`, and it ports the **wrong 3%**:

| part of a 0.837 ms `SplitPatch` | cost | share | the kernel? |
|---|---|---|---|
| `seams.Build` + `RebuildViews` | **0.826 ms** | **98.7%** | **no** |
| the LSQ fit (4 children × 81 evaluates) | 0.029 ms | 3.4% | **yes** |

The real cost: `RefineWhere` calls `SplitPatch` in a **loop**, and each
split rebuilt the seam graph over a **growing** surface set — 42 splits →
42 full rebuilds → O(n²). And `seams.Build` is itself super-linear
(1.08 → 1.84 µs/surface from 544 to 4,354 surfaces).

**The fix:** `PSFBody::BeginSeamBatch()` / `EndSeamBatch()` defer the
rebuild to the end of a batch. `SplitPatch` and `MergeGroup` accumulate
their touched patches instead of rebuilding.

**Result: 37.80 → 7.57 ms/op (5.0×), with a bit-identical surface**
(deviation 0.000e+00, identical max seam gap 0.14703% of the diagonal).

**The kernel stays unwired** — correctly. Porting 3% of the work is not
worth a second GPU pipeline, a second set of buffers, and a second source
of truth for the split. It remains in the tree as a reference
implementation; the register records why it is not launched.

Gate: `tests/validate_seam_batch.cpp` — 27 checks, 0 failures. It compares
the deferred body's surface count, **anchor positions**, **evaluated surface
points** and seam metrics against the eager path. Proven to catch the bug:
removing the final rebuild makes the surface tear by **27.3% of the
diagonal**.

**The anchor COUNT is deliberately not an invariant.** `find_existing_mid`
reuses a hanging anchor a *neighbouring* split already created; with a
stale seam graph the deferred path creates a fresh one instead, snapped to
the current (1-ULP-moved) parent surface, so the dedupe map misses it. The
two bodies hold the same anchor **positions** with a different record
**count** (712 vs 730 on 30 adjacent splits; every eager anchor has an
exact position match). The tolerance is **derived from the
representation's own resolution** (`bbox_span / 65535` = 4.12e-05), not
guessed.

→ `procedural-surface-engine` → `references/amr-cost-structure.md`.

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
