---
agent: pipeline
topic: milestone
updated: 2026-09-15T14:20:23+00:00
---

# M4 — seams + PatternLibrary

Goal: multi-patch bodies with seam-aware hits; PatternLibrary series in fill.
- SeamBridge / SeamGraph: link adjacent patches without meshing.
- PatternLibrary: evaluate series (coeffs × RGB) for fill average.
- SplatRenderer fill must call into pattern eval, not hard-coded colors.
