#include "inorganic_generator.h"
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

PatchCoefficients make_deformation(const MaterialDefinition& mat_def, AssemblyMethod method) {
    PatchCoefficients coeffs;
    switch (mat_def.deformation) {
        case DeformationBehavior::Brittle:
            for (int i = 0; i < 8; ++i) coeffs.scalars[i] = 0.005f;
            break;
        case DeformationBehavior::Plastic:
            for (int i = 0; i < 4; ++i) coeffs.scalars[i] = 0.03f * mat_def.max_strain * 100;
            for (int i = 4; i < 8; ++i) coeffs.scalars[i] = -0.01f;
            break;
        default:
            for (int i = 0; i < 4; ++i) coeffs.scalars[i] = 0.05f;
            break;
    }
    if (method == AssemblyMethod::Welded)
        for (int i = 4; i < 8; ++i) coeffs.scalars[i] *= 0.5f;
    else if (method == AssemblyMethod::Bolted)
        for (int i = 0; i < 8; ++i) coeffs.scalars[i] *= 1.2f;
    return coeffs;
}

} // anonymous namespace

/* ── Semantic resolution ──────────────────────────────────────────── */
StructuralType InorganicGenerator::resolve_type(std::string_view semantic) const {
    auto tokens = tokenize(semantic);
    for (const auto& tok : tokens) {
        if (tok == "plate" || tok == "sheet" || tok == "panel") return StructuralType::Plate;
        if (tok == "beam" || tok == "bar" || tok == "rod") return StructuralType::Beam;
        if (tok == "shell" || tok == "hull") return StructuralType::Shell;
        if (tok == "gear" || tok == "cog") return StructuralType::Gear;
        if (tok == "wheel" || tok == "disc") return StructuralType::Wheel;
        if (tok == "blade" || tok == "propeller") return StructuralType::Blade;
        if (tok == "shaft" || tok == "axle") return StructuralType::Shaft;
        if (tok == "piston") return StructuralType::Piston;
        if (tok == "arch" || tok == "bridge") return StructuralType::Arch;
        if (tok == "truss") return StructuralType::Truss;
        if (tok == "frame" || tok == "chassis") return StructuralType::Frame;
        if (tok == "column" || tok == "pillar") return StructuralType::Column;
        if (tok == "box" || tok == "container") return StructuralType::Box;
        if (tok == "cylinder" || tok == "pipe" || tok == "tube") return StructuralType::Cylinder;
        if (tok == "sphere" || tok == "ball") return StructuralType::Sphere;
        if (tok == "dome" || tok == "vault") return StructuralType::Dome;
        if (tok == "hinge" || tok == "joint") return StructuralType::Hinge;
        if (tok == "bolt" || tok == "screw" || tok == "rivet") return StructuralType::Bolt;
        if (tok == "weld" || tok == "welded") return StructuralType::Weld;
    }
    return StructuralType::Part;
}

AssemblyMethod InorganicGenerator::resolve_assembly(std::string_view semantic) const {
    auto tokens = tokenize(semantic);
    for (const auto& tok : tokens) {
        if (tok == "welded" || tok == "weld") return AssemblyMethod::Welded;
        if (tok == "bolted" || tok == "bolt" || tok == "riveted") return AssemblyMethod::Bolted;
        if (tok == "cast" || tok == "casting" || tok == "poured") return AssemblyMethod::Cast;
        if (tok == "forged" || tok == "forging") return AssemblyMethod::Forged;
        if (tok == "assembled" || tok == "joined") return AssemblyMethod::Assembled;
        if (tok == "layered" || tok == "printed" || tok == "deposited") return AssemblyMethod::Layered;
    }
    return AssemblyMethod::Assembled;
}

/* ── Shape generators ─────────────────────────────────────────────── */
void InorganicGenerator::generate_plate(
    BodyTopology& topo, std::vector<Vec3>& rest_positions,
    const Vec3& center, const Vec3& scale, int grid_u, int grid_v)
{
    for (int vi = 0; vi < grid_v; ++vi) {
        for (int ui = 0; ui < grid_u; ++ui) {
            float u = (ui / std::max(1, grid_u - 1)) - 0.5f;
            float v = (vi / std::max(1, grid_v - 1)) - 0.5f;
            rest_positions.push_back({center.x + u * scale.x, center.y + v * scale.y, center.z});
        }
    }
}

void InorganicGenerator::generate_cylinder(
    BodyTopology& topo, std::vector<Vec3>& rest_positions,
    const Vec3& center, const Vec3& scale, int grid_u, int grid_v)
{
    for (int vi = 0; vi < grid_v; ++vi) {
        for (int ui = 0; ui < grid_u; ++ui) {
            float theta = (ui / std::max(1, grid_u - 1)) * kTwoPi;
            float h = (vi / std::max(1, grid_v - 1)) - 0.5f;
            rest_positions.push_back({
                center.x + std::cos(theta) * scale.x * 0.5f,
                center.y + h * scale.y,
                center.z + std::sin(theta) * scale.x * 0.5f
            });
        }
    }
}

void InorganicGenerator::generate_sphere(
    BodyTopology& topo, std::vector<Vec3>& rest_positions,
    const Vec3& center, const Vec3& scale, int grid_u, int grid_v)
{
    for (int vi = 0; vi < grid_v; ++vi) {
        for (int ui = 0; ui < grid_u; ++ui) {
            float u = ui / std::max(1, grid_u - 1);
            float v = vi / std::max(1, grid_v - 1);
            float theta = u * kTwoPi;
            float phi = v * kPi;
            float r = scale.x * 0.5f;
            rest_positions.push_back({
                center.x + r * std::sin(phi) * std::cos(theta),
                center.y + r * std::cos(phi),
                center.z + r * std::sin(phi) * std::sin(theta)
            });
        }
    }
}

void InorganicGenerator::generate_beam(
    BodyTopology& topo, std::vector<Vec3>& rest_positions,
    const Vec3& center, const Vec3& scale, int grid_u, int grid_v)
{
    for (int vi = 0; vi < grid_v; ++vi) {
        for (int ui = 0; ui < grid_u; ++ui) {
            float u = ui / std::max(1, grid_u - 1);
            float v = (vi / std::max(1, grid_v - 1)) - 0.5f;
            float cross = std::abs(v) > 0.3f ? 0.4f : 0.15f;
            rest_positions.push_back({
                center.x + (u - 0.5f) * scale.x,
                center.y + v * scale.y,
                center.z + cross * scale.z
            });
        }
    }
}

/* ── Anchor grid generation ─────────────────────────────────────────
 * Creates anchors in the topology AND appends rest positions.
 * Returns the AnchorId for each grid point so patches can reference them.
 * ─────────────────────────────────────────────────────────────────── */
std::vector<AnchorId> InorganicGenerator::generate_anchor_grid(
    BodyTopology& topo, std::vector<Vec3>& rest_positions,
    const Vec3& center, const Vec3& scale, const Vec3& rotation,
    int grid_u, int grid_v, StructuralType type)
{
    std::vector<AnchorId> ids;
    ids.reserve(static_cast<size_t>(grid_u) * grid_v);

    switch (type) {
        case StructuralType::Plate:
        case StructuralType::Shell:
            generate_plate(topo, rest_positions, center, scale, grid_u, grid_v);
            break;
        case StructuralType::Cylinder:
        case StructuralType::Shaft:
        case StructuralType::Wheel:
            generate_cylinder(topo, rest_positions, center, scale, grid_u, grid_v);
            break;
        case StructuralType::Sphere:
        case StructuralType::Dome:
            generate_sphere(topo, rest_positions, center, scale, grid_u, grid_v);
            break;
        case StructuralType::Beam:
        case StructuralType::Rod:
            generate_beam(topo, rest_positions, center, scale, grid_u, grid_v);
            break;
        default:
            generate_plate(topo, rest_positions, center, scale, grid_u, grid_v);
            break;
    }

    int count = grid_u * grid_v;
    int base = static_cast<int>(rest_positions.size()) - count;

    /* Apply rotation and create anchors in topology. */
    for (int i = 0; i < count; ++i) {
        rest_positions[base + i] = rotate_point(rest_positions[base + i], center, rotation);
        AnchorId id = topo.add_anchor();
        ids.push_back(id);
    }

    return ids;
}

/* ── Patch connection ─────────────────────────────────────────────── */
void InorganicGenerator::connect_patches(
    BodyTopology& topo, const std::vector<AnchorId>& anchors,
    int grid_u, int grid_v, MaterialId material,
    AssemblyMethod method, const float* deformation_hints)
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

            BasisType basis = (method == AssemblyMethod::Welded) ? BasisType::Bilinear : BasisType::Plano;
            topo.add_patch({a00, a10, a01, a11}, material, basis, coeffs);
        }
    }
}

/* ── Template building ────────────────────────────────────────────── */
BlueprintPart InorganicGenerator::build_template(
    StructuralType type, std::string_view name,
    MaterialScienceRegistry& mat_science, MaterialRegistry& phys_reg)
{
    BlueprintPart part;
    part.type = type;
    part.name = std::string(name);
    part.is_primary = true;
    part.children = {};
    part.connections = {};

    switch (type) {
        case StructuralType::Plate:
        case StructuralType::Shell:
            part.material = mat_science.get_material("steel", phys_reg);
            part.assembly = AssemblyMethod::Welded;
            part.scale = {0.5f, 0.5f, 0.02f};
            part.anchor_grid[0] = 6; part.anchor_grid[1] = 6;
            break;
        case StructuralType::Cylinder:
        case StructuralType::Shaft:
            part.material = mat_science.get_material("aluminum", phys_reg);
            part.assembly = AssemblyMethod::Cast;
            part.scale = {0.2f, 0.5f, 0.2f};
            part.anchor_grid[0] = 8; part.anchor_grid[1] = 6;
            break;
        case StructuralType::Sphere:
        case StructuralType::Dome:
            part.material = mat_science.get_material("steel", phys_reg);
            part.assembly = AssemblyMethod::Cast;
            part.scale = {0.3f, 0.3f, 0.3f};
            part.anchor_grid[0] = 8; part.anchor_grid[1] = 6;
            break;
        case StructuralType::Beam:
            part.material = mat_science.get_material("steel", phys_reg);
            part.assembly = AssemblyMethod::Welded;
            part.scale = {0.8f, 0.15f, 0.1f};
            part.anchor_grid[0] = 8; part.anchor_grid[1] = 4;
            break;
        case StructuralType::Gear:
            part.material = mat_science.get_material("steel", phys_reg);
            part.assembly = AssemblyMethod::Forged;
            part.scale = {0.2f, 0.05f, 0.2f};
            part.anchor_grid[0] = 12; part.anchor_grid[1] = 4;
            break;
        case StructuralType::Box:
            part.material = mat_science.get_material("aluminum", phys_reg);
            part.assembly = AssemblyMethod::Assembled;
            part.scale = {0.3f, 0.3f, 0.3f};
            part.anchor_grid[0] = 5; part.anchor_grid[1] = 5;
            break;
        default:
            part.material = mat_science.get_material("steel", phys_reg);
            part.assembly = AssemblyMethod::Assembled;
            part.scale = {0.3f, 0.3f, 0.3f};
            part.anchor_grid[0] = 4; part.anchor_grid[1] = 4;
            break;
    }
    return part;
}

/* ── Part generation ──────────────────────────────────────────────── */
void InorganicGenerator::generate_part(
    const BlueprintPart& part, BodyTopology& topo,
    std::vector<Vec3>& rest_positions, MaterialRegistry& phys_reg)
{
    auto anchors = generate_anchor_grid(topo, rest_positions,
                         part.position, part.scale, part.rotation,
                         part.anchor_grid[0], part.anchor_grid[1],
                         part.type);

    connect_patches(topo, anchors, part.anchor_grid[0], part.anchor_grid[1],
                    part.material, part.assembly, part.deformation);
}

/* ── Blueprint creation ───────────────────────────────────────────── */
Blueprint InorganicGenerator::create_blueprint(
    std::string_view semantic,
    MaterialScienceRegistry& mat_science,
    MaterialRegistry& phys_reg)
{
    Blueprint blueprint;
    blueprint.name = std::string(semantic);
    blueprint.primary_method = resolve_assembly(semantic);
    blueprint.primary_material = mat_science.resolve_from_text(semantic, phys_reg);

    auto parsed = DomainClassifier{}.parse(semantic);
    for (const auto& part_desc : parsed.parts) {
        StructuralType type = resolve_type(part_desc.name);
        BlueprintPart part = build_template(type, part_desc.name, mat_science, phys_reg);

        if (!part_desc.material.empty())
            part.material = mat_science.get_material(part_desc.material, phys_reg);

        part.position = part_desc.position;
        part.scale = part_desc.scale;
        blueprint.parts.push_back(part);
    }

    if (blueprint.parts.empty()) {
        BlueprintPart part = build_template(resolve_type(semantic), semantic,
                                            mat_science, phys_reg);
        blueprint.parts.push_back(part);
    }

    return blueprint;
}

/* ── Blueprint generation ─────────────────────────────────────────── */
std::unique_ptr<BodyRuntime> InorganicGenerator::generate_from_blueprint(
    const Blueprint& blueprint,
    MaterialScienceRegistry& mat_science,
    MaterialRegistry& phys_reg)
{
    BodyTopology topo;
    std::vector<Vec3> rest_positions;

    for (const auto& part : blueprint.parts) {
        generate_part(part, topo, rest_positions, phys_reg);
    }

    topo.finalize();

    auto runtime = std::make_unique<BodyRuntime>(std::move(topo));
    runtime->initialize(rest_positions);
    runtime->set_materials(phys_reg);
    return runtime;
}

/* ── Main generation ──────────────────────────────────────────────── */
std::unique_ptr<BodyRuntime> InorganicGenerator::generate(
    std::string_view semantic,
    MaterialScienceRegistry& mat_science,
    MaterialRegistry& phys_reg)
{
    Blueprint blueprint = create_blueprint(semantic, mat_science, phys_reg);
    return generate_from_blueprint(blueprint, mat_science, phys_reg);
}

} // namespace delirium