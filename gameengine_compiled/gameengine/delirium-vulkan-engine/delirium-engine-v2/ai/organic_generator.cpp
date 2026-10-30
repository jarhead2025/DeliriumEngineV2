#include "organic_generator.h"
#include "material_science.h"
#include <algorithm>
#include <cmath>
#include <cctype>

namespace delirium {

namespace {

std::string to_lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return out;
}

std::vector<std::string> tokenize(std::string_view s) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)))
            cur += std::tolower(static_cast<unsigned char>(c));
        else {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

Vec3 rotate_point(const Vec3& p, const Vec3& center, const Vec3& euler) {
    Vec3 v = {p.x - center.x, p.y - center.y, p.z - center.z};
    { float c = std::cos(euler.y), s = std::sin(euler.y);
      float x = v.x * c + v.z * s; float z = -v.x * s + v.z * c; v.x = x; v.z = z; }
    { float c = std::cos(euler.x), s = std::sin(euler.x);
      float y = v.y * c - v.z * s; float z = v.y * s + v.z * c; v.y = y; v.z = z; }
    { float c = std::cos(euler.z), s = std::sin(euler.z);
      float x = v.x * c - v.y * s; float y = v.x * s + v.y * c; v.x = x; v.y = y; }
    return {v.x + center.x, v.y + center.y, v.z + center.z};
}

/* Material-driven coefficients — set at add_patch() time since patch() is const. */
PatchCoefficients make_deformation(const MaterialDefinition& mat_def) {
    PatchCoefficients coeffs;
    switch (mat_def.deformation) {
        case DeformationBehavior::Hyperelastic:
            for (int i = 0; i < 4; ++i) coeffs.scalars[i] = 0.15f * mat_def.max_strain * 10;
            break;
        case DeformationBehavior::Brittle:
            for (int i = 0; i < 8; ++i) coeffs.scalars[i] = 0.01f;
            break;
        case DeformationBehavior::Viscoelastic:
            for (int i = 0; i < 4; ++i) coeffs.scalars[i] = 0.08f;
            for (int i = 4; i < 8; ++i) coeffs.scalars[i] = -0.02f;
            break;
        default:
            for (int i = 0; i < 4; ++i) coeffs.scalars[i] = 0.05f;
            break;
    }
    return coeffs;
}

} // anonymous namespace

/* ── Semantic resolution ──────────────────────────────────────────── */
AnatomyType OrganicGenerator::resolve_type(std::string_view semantic) const {
    auto tokens = tokenize(semantic);
    for (const auto& tok : tokens) {
        if (tok == "head" || tok == "skull") return AnatomyType::Head;
        if (tok == "neck") return AnatomyType::Neck;
        if (tok == "torso" || tok == "thorax" || tok == "chest") return AnatomyType::Thorax;
        if (tok == "abdomen" || tok == "belly" || tok == "gut") return AnatomyType::Abdomen;
        if (tok == "arm" || tok == "forearm") return AnatomyType::UpperArm;
        if (tok == "hand" || tok == "paw" || tok == "claw") return AnatomyType::Hand;
        if (tok == "leg" || tok == "thigh" || tok == "shin") return AnatomyType::Thigh;
        if (tok == "foot" || tok == "toe") return AnatomyType::Foot;
        if (tok == "wing") return AnatomyType::Wing;
        if (tok == "fin") return AnatomyType::Fin;
        if (tok == "tail") return AnatomyType::Tail;
        if (tok == "antenna" || tok == "antennae") return AnatomyType::Antenna;
        if (tok == "eye") return AnatomyType::Eye;
        if (tok == "trunk" || tok == "branch") return AnatomyType::Branch;
        if (tok == "leaf") return AnatomyType::Leaf;
        if (tok == "flower" || tok == "bloom") return AnatomyType::Flower;
        if (tok == "root") return AnatomyType::Root;
        if (tok == "body" || tok == "creature" || tok == "organism") return AnatomyType::Segment;
    }
    return AnatomyType::Segment;
}

GrowthPattern OrganicGenerator::resolve_growth(std::string_view semantic) const {
    auto tokens = tokenize(semantic);
    for (const auto& tok : tokens) {
        if (tok == "sphere" || tok == "round" || tok == "globular") return GrowthPattern::Radial;
        if (tok == "tube" || tok == "cylinder" || tok == "elongated") return GrowthPattern::Tubular;
        if (tok == "flat" || tok == "sheet" || tok == "membrane") return GrowthPattern::Lamellar;
        if (tok == "branch" || tok == "tree" || tok == "bifurcating") return GrowthPattern::Branching;
        if (tok == "spiral" || tok == "helix" || tok == "coil") return GrowthPattern::Spiral;
        if (tok == "layer" || tok == "shell" || tok == "concentric") return GrowthPattern::Layered;
    }
    return GrowthPattern::Radial;
}

/* ── Anchor grid generation ─────────────────────────────────────────
 * Creates anchors in the topology AND appends rest positions.
 * Returns the AnchorId for each grid point so patches can reference them.
 * ─────────────────────────────────────────────────────────────────── */
std::vector<AnchorId> OrganicGenerator::generate_anchor_grid(
    BodyTopology& topo, std::vector<Vec3>& rest_positions,
    const Vec3& center, const Vec3& scale, const Vec3& orientation,
    int grid_u, int grid_v, GrowthPattern pattern)
{
    std::vector<AnchorId> ids;
    ids.reserve(static_cast<size_t>(grid_u) * grid_v);

    for (int vi = 0; vi < grid_v; ++vi) {
        for (int ui = 0; ui < grid_u; ++ui) {
            float u = ui / std::max(1, grid_u - 1);
            float v = vi / std::max(1, grid_v - 1);
            Vec3 local{};
            switch (pattern) {
                case GrowthPattern::Radial:
                    local = {std::cos(u * kTwoPi) * std::sin(v * kPi),
                             std::cos(v * kPi),
                             std::sin(u * kTwoPi) * std::sin(v * kPi)};
                    break;
                case GrowthPattern::Tubular:
                    local = {std::cos(u * kTwoPi) * 0.5f, v - 0.5f,
                             std::sin(u * kTwoPi) * 0.5f};
                    break;
                case GrowthPattern::Lamellar:
                    local = {u - 0.5f, 0, v - 0.5f};
                    break;
                case GrowthPattern::Branching:
                    local = {(u - 0.5f) * std::cos(v * kPi * 0.5f), v - 0.5f, 0};
                    break;
                case GrowthPattern::Spiral:
                    local = {std::cos(u * kTwoPi) * (1 - v), v - 0.5f,
                             std::sin(u * kTwoPi) * (1 - v)};
                    break;
                case GrowthPattern::Layered:
                    local = {std::cos(u * kTwoPi) * 0.5f,
                             std::sin(u * kTwoPi) * 0.5f * std::cos(v * kPi),
                             std::sin(u * kTwoPi) * 0.5f * std::sin(v * kPi)};
                    break;
            }
            Vec3 scaled = {local.x * scale.x + center.x,
                           local.y * scale.y + center.y,
                           local.z * scale.z + center.z};
            Vec3 world = rotate_point(scaled, center, orientation);

            // Create anchor in topology AND record rest position
            AnchorId id = topo.add_anchor();
            rest_positions.push_back(world);
            ids.push_back(id);
        }
    }
    return ids;
}

/* ── Patch connection ─────────────────────────────────────────────── */
void OrganicGenerator::connect_patches(
    BodyTopology& topo, const std::vector<AnchorId>& anchors,
    int grid_u, int grid_v, MaterialId material,
    const float* deformation_hints)
{
    for (int vi = 0; vi < grid_v - 1; ++vi) {
        for (int ui = 0; ui < grid_u - 1; ++ui) {
            AnchorId a00 = anchors[vi * grid_u + ui];
            AnchorId a10 = anchors[vi * grid_u + ui + 1];
            AnchorId a01 = anchors[(vi + 1) * grid_u + ui];
            AnchorId a11 = anchors[(vi + 1) * grid_u + ui + 1];

            PatchCoefficients coeffs;
            if (deformation_hints)
                for (int i = 0; i < 8; ++i) coeffs.scalars[i] = deformation_hints[i];

            topo.add_patch({a00, a10, a01, a11}, material, BasisType::Bilinear, coeffs);
        }
    }
}

/* ── Template building ────────────────────────────────────────────── */
AnatomySegment OrganicGenerator::build_template(
    AnatomyType type, std::string_view name,
    MaterialScienceRegistry& mat_science, MaterialRegistry& phys_reg)
{
    AnatomySegment seg;
    seg.type = type;
    seg.name = std::string(name);
    seg.children = {};

    switch (type) {
        case AnatomyType::Thorax:
            seg.material = mat_science.get_material("tissue", phys_reg);
            seg.growth = GrowthPattern::Radial;
            seg.scale = {0.3f, 0.4f, 0.2f};
            seg.anchor_grid[0] = 6; seg.anchor_grid[1] = 6;
            seg.deformation[0] = seg.deformation[1] = seg.deformation[2] = seg.deformation[3] = 0.12f;
            break;
        case AnatomyType::Head:
            seg.material = mat_science.get_material("bone", phys_reg);
            seg.growth = GrowthPattern::Radial;
            seg.scale = {0.15f, 0.18f, 0.15f};
            seg.anchor_grid[0] = 5; seg.anchor_grid[1] = 5;
            break;
        case AnatomyType::Wing:
            seg.material = mat_science.get_material("cartilage", phys_reg);
            seg.growth = GrowthPattern::Lamellar;
            seg.scale = {0.5f, 0.05f, 0.15f};
            seg.anchor_grid[0] = 6; seg.anchor_grid[1] = 4;
            seg.deformation[0] = 0.08f; seg.deformation[1] = -0.04f;
            break;
        case AnatomyType::Tail:
            seg.material = mat_science.get_material("collagen", phys_reg);
            seg.growth = GrowthPattern::Tubular;
            seg.scale = {0.05f, 0.3f, 0.05f};
            seg.anchor_grid[0] = 4; seg.anchor_grid[1] = 8;
            break;
        case AnatomyType::Leaf:
            seg.material = mat_science.get_material("cellulose", phys_reg);
            seg.growth = GrowthPattern::Lamellar;
            seg.scale = {0.2f, 0.02f, 0.1f};
            seg.anchor_grid[0] = 5; seg.anchor_grid[1] = 4;
            seg.deformation[0] = 0.06f;
            break;
        default:
            seg.material = mat_science.get_material("tissue", phys_reg);
            seg.growth = GrowthPattern::Radial;
            seg.scale = {0.2f, 0.2f, 0.2f};
            seg.anchor_grid[0] = 4; seg.anchor_grid[1] = 4;
            break;
    }
    return seg;
}

/* ── Segment generation ───────────────────────────────────────────── */
void OrganicGenerator::generate_segment(
    const AnatomySegment& seg, BodyTopology& topo,
    std::vector<Vec3>& rest_positions, MaterialRegistry& phys_reg)
{
    // generate_anchor_grid now creates anchors AND returns their IDs
    auto anchors = generate_anchor_grid(topo, rest_positions,
                         seg.position, seg.scale, seg.orientation,
                         seg.anchor_grid[0], seg.anchor_grid[1],
                         seg.growth);

    connect_patches(topo, anchors, seg.anchor_grid[0], seg.anchor_grid[1],
                    seg.material, seg.deformation);
}

/* ── Main generation ──────────────────────────────────────────────── */
std::unique_ptr<BodyRuntime> OrganicGenerator::generate(
    std::string_view semantic,
    MaterialScienceRegistry& mat_science,
    MaterialRegistry& phys_reg)
{
    auto parsed = DomainClassifier{}.parse(semantic);

    BodyTopology topo;
    std::vector<Vec3> rest_positions;

    float offset = 0;
    for (const auto& part : parsed.parts) {
        AnatomyType type = resolve_type(part.name);
        auto seg = build_template(type, part.name, mat_science, phys_reg);
        seg.position.y = offset;
        offset += seg.scale.y * 0.7f;

        if (!part.material.empty())
            seg.material = mat_science.get_material(part.material, phys_reg);

        generate_segment(seg, topo, rest_positions, phys_reg);
    }

    topo.finalize();

    auto runtime = std::make_unique<BodyRuntime>(std::move(topo));
    runtime->initialize(rest_positions);
    runtime->set_materials(phys_reg);
    return runtime;
}

} // namespace delirium