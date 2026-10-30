#pragma once
/* ── Inorganic Generation Framework ──────────────────────────────────
 *
 * Generates engineered structures from semantic descriptions using
 * material science principles: structural mechanics, assembly methods,
 * and blueprint-driven construction.
 *
 * The model "understands" inorganic structures through:
 * - Material properties (strength, density, deformation behavior)
 * - Assembly methods (welding, bolting, casting, forging)
 * - Blueprint hierarchy (parts → subassemblies → complete structure)
 *
 * For complex objects, the blueprint system decomposes the description
 * into parts with explicit connections and assembly order.
 *
 * Math boundary: operates at the continuum mechanics level — patches
 * represent structural elements, coefficients represent deformation
 * and curvature. No molecular structure.
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

/* ── Structural element type ──────────────────────────────────────── */
enum class StructuralType : std::uint8_t {
    // Basic shapes
    Plate, Beam, Shell, Rod,
    // Mechanical
    Gear, Wheel, Blade, Shaft, Piston,
    // Structural
    Arch, Truss, Frame, Column,
    // Container
    Box, Cylinder, Sphere, Dome,
    // Connection
    Hinge, Joint, Bolt, Weld,
    // Generic
    Part, Subassembly,
};

/* ── Assembly method ──────────────────────────────────────────────── */
enum class AssemblyMethod : std::uint8_t {
    Welded,      // fused at seams (continuous)
    Bolted,      // discrete connection points
    Cast,        // single poured piece
    Forged,      // shaped under pressure
    Assembled,   // multiple parts joined
    Layered,     // deposited layer by layer
};

/* ── Blueprint part ───────────────────────────────────────────────── */
struct BlueprintPart {
    std::string name;
    StructuralType type;
    MaterialId material;
    AssemblyMethod assembly;
    Vec3 scale{1, 1, 1};
    Vec3 position{};
    Vec3 rotation{0, 0, 0};
    std::vector<BlueprintPart> children;
    std::vector<BlueprintPart*> connections; // parts this connects to
    int anchor_grid[2]{4, 4};
    float deformation[8]{};
    bool is_primary; // main structural element
};

/* ── Blueprint definition ─────────────────────────────────────────── */
struct Blueprint {
    std::string name;
    std::vector<BlueprintPart> parts;
    AssemblyMethod primary_method;
    MaterialId primary_material;
};

/* ── Inorganic generator ──────────────────────────────────────────── */
class InorganicGenerator {
public:
    /* Generate a body from semantic description.
     * Returns a configured BodyRuntime ready for simulation. */
    std::unique_ptr<BodyRuntime> generate(
        std::string_view semantic,
        MaterialScienceRegistry& mat_science,
        MaterialRegistry& phys_reg);

    /* Generate from a blueprint definition. */
    std::unique_ptr<BodyRuntime> generate_from_blueprint(
        const Blueprint& blueprint,
        MaterialScienceRegistry& mat_science,
        MaterialRegistry& phys_reg);

    /* Generate a single structural part as a sub-topology. */
    void generate_part(
        const BlueprintPart& part,
        BodyTopology& topo,
        std::vector<Vec3>& rest_positions,
        MaterialRegistry& phys_reg);

    /* Build a predefined structural template. */
    BlueprintPart build_template(StructuralType type, std::string_view name,
                                 MaterialScienceRegistry& mat_science,
                                 MaterialRegistry& phys_reg);

    /* Create a blueprint from semantic description. */
    Blueprint create_blueprint(
        std::string_view semantic,
        MaterialScienceRegistry& mat_science,
        MaterialRegistry& phys_reg);

private:
    /* Generate anchor grid for a structural part. */
    std::vector<AnchorId> generate_anchor_grid(
        BodyTopology& topo,
        std::vector<Vec3>& rest_positions,
        const Vec3& center,
        const Vec3& scale,
        const Vec3& rotation,
        int grid_u, int grid_v,
        StructuralType type);

    /* Connect anchors into patches with structural coefficients. */
    void connect_patches(
        BodyTopology& topo,
        const std::vector<AnchorId>& anchors,
        int grid_u, int grid_v,
        MaterialId material,
        AssemblyMethod method,
        const float* deformation_hints);

    /* Resolve structural type from semantic keywords. */
    StructuralType resolve_type(std::string_view semantic) const;

    /* Resolve assembly method from semantic keywords. */
    AssemblyMethod resolve_assembly(std::string_view semantic) const;

    /* Generate common structural shapes. */
    void generate_plate(BodyTopology&, std::vector<Vec3>&, const Vec3&, const Vec3&, int, int);
    void generate_cylinder(BodyTopology&, std::vector<Vec3>&, const Vec3&, const Vec3&, int, int);
    void generate_sphere(BodyTopology&, std::vector<Vec3>&, const Vec3&, const Vec3&, int, int);
    void generate_beam(BodyTopology&, std::vector<Vec3>&, const Vec3&, const Vec3&, int, int);
};

} // namespace delirium