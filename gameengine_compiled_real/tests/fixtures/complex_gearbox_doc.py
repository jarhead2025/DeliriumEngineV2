#!/usr/bin/env python3
"""
A COMPLEX, MULTI-PART technical document -- the stress test for the model
generation.

WHY THIS DOCUMENT EXISTS (Josias, 2026-11-10): "design a more complex scene
and test the model generation."

The earlier documents were SINGLE-PART: one gearbox housing, one shaft. A
single part can be answered with one ellipsoid, so the test never exercised
the part that matters -- a document that describes a MECHANISM becomes a
tree with several leaves, ARTICULATED to each other, each with its own
material and mass.

So this document describes a GEARBOX ASSEMBLY:

  1. a HOUSING  (a box-ish shell, with a rim: the closed-body rule)
  2. a SHAFT    (a long rod through the housing)
  3. two GEARS  (discs on the shaft, at different places, different sizes)
  4. a BEARING  (a ring around the shaft)
  5. a COVER PLATE

That is six leaves, two of them needing the CLOSED-BODY rule, all of them
needing the ARTICULATION (the position and the orientation relative to the
others), and several needing MATERIALS.

WHAT THE TEST PROVES (and what it cannot):
  - the model produces a MULTI-LEAF tree, not one blob;
  - the leaves are POSITIONED relative to each other (the articulations
    are not all at the origin);
  - the tree passes the structural validator;
  - it bakes into a valid body.
  It CANNOT prove the model understood the MECHANISM -- a 7B model may well
  produce a structurally valid tree that is not the right machine. So the
  report separates the two claims.

THE OUTPUT: a JSON summary the harness can assert on, plus the model's raw
reply (so a failure can be READ).
"""
import json
import sys

DOC = """Industrial gearbox assembly, model GB-220.

The assembly consists of the following parts, mounted on a common frame.

1. HOUSING. A cast-iron housing, roughly 1.0 by 1.0 by 0.6 in overall
   size, forming a closed box. The housing is a closed body: it has a top
   wall and a bottom wall that meet at the side walls. Wall thickness 0.08.
   Material: cast iron.

2. INPUT SHAFT. A steel shaft, diameter 0.12, running horizontally through
   the housing along the long axis, protruding 0.15 beyond the housing on
   the input side. Length 1.3. Material: steel.

3. PINION GEAR. A steel spur gear on the input shaft, 0.30 in diameter and
   0.10 wide, mounted immediately outside the input end of the housing.
   Material: steel.

4. MAIN GEAR. A steel spur gear on the output side of the housing, 0.50 in
   diameter and 0.12 wide, meshing with the pinion. Material: steel.

5. OUTPUT SHAFT. A steel shaft, diameter 0.14, running parallel to and
   below the input shaft, passing through the main gear and protruding
   0.20 beyond the housing on the output side. Length 1.2. Material: steel.

6. BEARING. A bronze bearing, a ring of outer diameter 0.30 and inner
   diameter 0.16, 0.12 wide, supporting the input shaft where it passes
   through the housing wall. Material: bronze.

7. COVER PLATE. An aluminium cover plate, 1.0 by 1.0 by 0.04, bolted to the
   top of the housing. Material: aluminium.

The assembly has a total mass of 12 kg, most of it in the housing.
"""

if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps({"id": "gearbox_assembly", "text": DOC}))
    else:
        print(DOC)
