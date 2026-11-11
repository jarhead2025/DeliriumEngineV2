// =============================================================================
// MutualMesh.h - DERIVE the mesh cells from the POINT SET, instead of
// placing them on a grid.
//
// THE IDEA (Josias, 2026-11-10): "think of a bunch of dots, if you had to
// connect them by grabbing them into two sets, you can use the ones in the
// first set to determine the surround triangle in the second, and vice
// versa, not that we are necessarily using triangular shaped patches, it
// just is a convenient way to stack them on top of another to accurately
// mesh them, overlap them and average them."
//
// Read carefully, this is TWO rules:
//
//   RULE 1 -- MUTUAL SURROUND DETERMINATION. Split the point set into two
//   sets. Use the points of set A to determine the surrounding cells of the
//   points of set B, and vice versa. The triangles are a CONVENIENT
//   STACKING DEVICE -- the output is a quad-patch mesh built from the
//   averaged cells.
//
//   RULE 2 -- OVERLAP AND AVERAGE. Two meshes over the same surface; their
//   cells are averaged into one.
//
// WHY IT MATTERS: the current baker PLACES a grid and hopes the surface
// lands on it. Every grid bug comes from that guessing:
//   - the grid can DUPLICATE AN ENDPOINT (the polar axis's i=N-1 landed on
//     angle 2*pi, the same point as i=0);
//   - the grid can be USED INCONSISTENTLY across the three sites that
//     sample it (the anchors, the patch-fit target, the bbox);
//   - the grid can have a TOPOLOGY THE DATA DOES NOT (a periodic axis
//     needs periodic connectivity).
// A DERIVATION from the point set has none of those three failure modes,
// because it has no parameterisation to get wrong.
//
// HALF OF IT ALREADY EXISTS IN THIS ENGINE:
//   - Rule 1's "one set determines the other's surround" IS what
//     SeamGraph::Build does -- it matches anchor PAIRS to find, for every
//     patch, the patches that surround it.
//   - Rule 2's "overlap and average" IS what SeamGraph::HarmonizeEdgeCoefs
//     does -- it averages the two sides' deformation coefficients so a
//     shared edge becomes ONE curve.
//
// WHAT IS NEW HERE is the BAKER deriving its cells, rather than placing
// them.
//
// THE ALGORITHM (as implemented):
//
//   1. SAMPLE the substance into a point cloud (not a grid): each sample is
//      a point with its surface normal. The cloud is the "bunch of dots".
//
//   2. TRIANGULATE the cloud in OBJECT SPACE with a Delaunay
//      triangulation. The triangles are the "convenient stacking device":
//      each triangle's three points are a candidate cell.
//
//   3. PAIR the triangles into QUADS. The engine's native cell is a quad
//      (four corner anchors, bilinear), so the triangles must be merged
//      into quads. Two triangles sharing an edge form a quad.
//
//   4. OVERLAP AND AVERAGE. Where a quad from one triangulation overlaps a
//      quad from another (a different split of the same point set), the
//      two are averaged into one cell -- the same rule
//      HarmonizeEdgeCoefs applies per edge.
//
// The Delaunay step is not new machinery: engine/src/render/TriInfill.cpp
// already has a Bowyer-Watson implementation. It is screen-space and
// structured there; this needs it in OBJECT space on an arbitrary cloud.
//
// WHY THE TRIANGLES ARE A STACKING DEVICE, NOT THE OUTPUT:
//   a Delaunay triangulation of a surface point cloud is NOT the surface's
//   natural connectivity -- near a boundary it produces long thin slivers,
//   and it has no notion of "which side is up". So the triangles are used
//   for their TOPOLOGY (who is adjacent to whom) and their AREA (a weight
//   for the averaging), and the quad cells are built from the resulting
//   connectivity. The surface itself is still the LSQ fit against the
//   substance -- exactly as the grid version does -- so the geometry
//   quality is unchanged; only the CONNECTIVITY is derived instead of
//   placed.
//
// THE CONTRACT:
//   MutualMesh::Derive(points, ...) -> a set of QUADS (4 anchor indices
//   each) + the anchor list. The caller (the baker) then:
//     - writes the anchors (deduped by position, as the anchor pool does);
//     - fits each quad's 8 corner-vanishing coefficients against the
//       substance, as the grid path does;
//     - writes the SurfaceRecords.
//   So the two paths differ ONLY in how the connectivity is obtained, and
//   they share the entire surface-fitting machinery.
//
// WHAT THIS DOES NOT SOLVE (honestly):
//   - The AMR split/merge rely on a REGULAR lattice (a patch's four
//     children are its four quadrants). A derived mesh has no quadrants.
//     So a derived body cannot be AMR-split the way a grid body can, unless
//     the split is generalised to "split the cell's four corners" -- which
//     is possible for an arbitrary quad but is real work.
//   - The Delaunay triangulation of a surface cloud is O(n^2) in the
//     worst case (Bowyer-Watson is O(n^2) naive, O(n log n) with a good
//     incircle structure). At grid 48 that is ~2300 points per sheet, which
//     is fine, but it is not free.
//   - The "two sets" in Rule 1 are, in this implementation, TWO
//     DELAUNAY TRIANGULATIONS OF THE SAME CLOUD with different tie-breaks
//     (an odd/even split of the points, triangulated separately). That is
//     the honest reading of "grab them into two sets ... and vice versa":
//     each set's triangulation determines the other's surround, and the
//     overlap between the two meshes is what gets averaged.
// =============================================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "delirium/core/Types.h"

namespace DeliRium {

// A sample of the substance: a position and its surface normal. This is
// the "dot" -- the point set the mesh is derived from.
struct MeshPoint {
    Vec3 p;
    Vec3 n;          // the surface normal at p (unit)
    float weight = 1.f;
};

// One derived cell: four corner points. The engine's native cell.
struct MeshQuad {
    uint32_t v[4] = {0, 0, 0, 0};   // indices into the point list
};

// The result of a derivation: the points, the quads, and the statistics
// that let a caller (and a gate) see what happened.
struct MeshDerivation {
    std::vector<MeshPoint> points;
    std::vector<MeshQuad>  quads;
    // How many of the quads came from the A-half's triangulation, the
    // B-half's, and from OVERLAPPING cells that were averaged.
    uint32_t quads_from_a = 0;
    uint32_t quads_from_b = 0;
    uint32_t quads_averaged = 0;
    // The triangles that could NOT be paired into quads (the boundary and
    // the odd ones out). They are reported, not silently dropped.
    uint32_t triangles_unpaired = 0;
    // The DIAGNOSTIC counts: how many triangles each set's triangulation
    // produced, and how many edges it had shared by exactly two
    // triangles (the edges that CAN pair into a quad).
    uint32_t triangles_a = 0, triangles_b = 0;
    uint32_t pairable_edges_a = 0, pairable_edges_b = 0;
};

// Derive the mesh from a point cloud.
//
// `split` selects the two sets: the points are divided by their INDEX
// parity within the Delaunay insertion order (0 = set A, 1 = set B), so
// the two sets INTERLEAVE through the cloud -- which is what makes each
// set's triangulation determine the other's surround.
//
// `epsilon` is the spatial tolerance for "these two points are the same
// vertex" when the two triangulations are merged.
//
// Returns false and sets `why` if the cloud cannot be triangulated.
bool DeriveMutualMesh(const std::vector<MeshPoint>& points,
                      MeshDerivation& out, std::string& why,
                      float epsilon = 1e-5f);

} // namespace DeliRium
