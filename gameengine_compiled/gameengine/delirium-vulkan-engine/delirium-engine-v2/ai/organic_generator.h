#pragma once
/* ── Organic Generation Framework ────────────────────────────────────
 *
 * Generates anatomical structures from semantic descriptions using
 * biological principles: growth patterns, hierarchical anatomy, and
 * material-appropriate deformation.
 *
 * The model "understands" organic structures through material science:
 * bone provides rigidity, cartilage flexibility, tissue volume,
 * collagen connectivity. It crafts parts by assembling mathematical
 * primitives (patches with coefficients) that reproduce biological
 * assembly patterns.
 *
 * Math boundary: operates at the continuum mechanics level (patches,
 * coefficients, seams) — never goes into molecular structure.
 * ──────────────────────────────────────────────────────────────────── */

#include "core/body_topology.h"
#include "core/body_state.h"
#include "core/body_runtime.h"
#include "core/material_registry.h"
#include "ai/domain_classifier.h"
#include "ai/material_science.h"
#include <memory>
#include <string>
#include <vector>

namespace delirium {

/* ── Anatomical part type ─────────────────────────────────────────── */
enum class AnatomyType : std::uint8_t {
    // Body regions
    Head, Neck, Thorax, Abdomen, Pelvis,
    // Limbs
    UpperArm, Forearm, Hand, Fingers,
    Thigh, Shin, Foot, Toes,
    // Appendages
    Wing, Fin, Tail, Antenna,
    // Sensory
    Eye, Ear, Snout,
    // Plant
    Trunk, Branch, Leaf, Flower, Root,
    // Generic
    Segment, Node, Connection,
};

/* ── Growth pattern ───────────────────────────────────────────────── */
enum class GrowthPattern : std::uint8_t {
    Radial,        // sphere-like expansion
    Tubular,       // cylinder-like elongation
    Lamellar,      // sheet-like spreading
    Branching,     // tree-like bifurcation
    Spiral,        // helical growth
    Layered,       // concentric shells
};

/* ── Anatomical segment ───────────────────────────────────────────── */
struct AnatomySegment {
    AnatomyType type;
    std::string name;
    MaterialId material;
    GrowthPattern growth;
    Vec3 scale{1, 1, 1};
    Vec3 position{};
    Vec3 orientation{0, 0, 0}; // euler angles
    std::vector<AnatomySegment> children;
    int anchor_grid[2]{4, 4};  // anchors per axis for patch generation
    float deformation[8]{};    // patch coefficient hints
};

/* ── Organic generator ────────────────────────────────────────────── */
class OrganicGenerator {
public:
    /* Generate a body from semantic description.
     * Returns a configured BodyRuntime ready for simulation. */
    std::unique_ptr<BodyRuntime> generate(
        std::string_view semantic,
        MaterialScienceRegistry& mat_science,
        MaterialRegistry& phys_reg);

    /* Generate a single anatomical segment as a sub-topology. */
    void generate_segment(
        const AnatomySegment& seg,
        BodyTopology& topo,
        std::vector<Vec3>& rest_positions,
        MaterialRegistry& phys_reg);

    /* Build a predefined anatomical template. */
    AnatomySegment build_template(AnatomyType type, std::string_view name,
                                  MaterialScienceRegistry& mat_science,
                                  MaterialRegistry& phys_reg);

private:
    /* Generate anchor grid for a segment. Returns AnchorIds for each grid point. */
    std::vector<AnchorId> generate_anchor_grid(
        BodyTopology& topo,
        std::vector<Vec3>& rest_positions,
        const Vec3& center,
        const Vec3& scale,
        const Vec3& orientation,
        int grid_u, int grid_v,
        GrowthPattern pattern);

    /* Connect anchors into patches with appropriate coefficients. */
    void connect_patches(
        BodyTopology& topo,
        const std::vector<AnchorId>& anchors,
        int grid_u, int grid_v,
        MaterialId material,
        const float* deformation_hints);

    /* Resolve anatomy type from semantic keywords. */
    AnatomyType resolve_type(std::string_view semantic) const;

    /* Resolve growth pattern from semantic keywords. */
    GrowthPattern resolve_growth(std::string_view semantic) const;
};

} // namespace delirium