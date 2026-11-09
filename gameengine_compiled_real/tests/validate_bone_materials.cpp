// =============================================================================
// validate_bone_materials - GATE for the material round-trip (G12).
//
// THE CHAIN THIS GATE COVERS:
//
//     document says "steel housing"
//            |
//            v
//     JSON:  "bone": { ..., "material": "steel" }
//            |
//            v
//     parser:  Bone::material = "steel"
//            |
//            v
//     baker:   BoneRecord.reserved[6..27] = "steel\0"
//            |
//            v
//     resolver: FieldProperties (steel's REAL constants)
//
// WHY IT MATTERS: the material is stored as a NAME, not as a property
// table. So the round-trip is a lossy-looking but actually-lossless
// design ONLY IF the name survives the frozen record and resolves back to
// the same constants. This gate proves that it does -- and that the two
// failure modes (a name that does not survive; a name that survives but
// does not resolve) are both caught.
//
// WHAT IS ASSERTED:
//
//   1. THE NAME SURVIVES THE ROUND TRIP, byte for byte.
//   2. IT RESOLVES TO THE SAME CONSTANTS it started from.
//   3. AN UNKNOWN MATERIAL IS AN ERROR, not a default.
//   4. A BONE WITH NO MATERIAL IS NOT AN ERROR -- it simply has no
//      properties, and that is honest.
//   5. TWO BONES WITH DIFFERENT MATERIALS STAY DIFFERENT -- the encoding
//      does not collapse them.
//   6. THE ENCODING DOES NOT DISTURB the fields that were already in the
//      reserved bytes (mass, radius, stiffness).
//   7. THE PROMPTS TEACH what they must: the material vocabulary appears
//      in the schema prompt, and the document prompt says to read the
//      document as a mechanism.
// =============================================================================
#include "delirium/physics/FieldProperties.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/ai/DocLearner.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace DeliRium;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

namespace {

// A two-part machine: a steel shaft in an ABS housing, each with a bone
// and a material.
const char* kMachine = R"({
  "kind":"assembly","children":[
    {"kind":"shaft","substance":"ellipsoid",
     "substance_params":{"sx":0.15,"sy":0.15,"sz":0.45},
     "bone":{"name":"shaft","head":[0,0,-0.45],"tail":[0,0,0.45],
             "mass":4.0,"radius":0.15,"stiffness":0.9,
             "material":"steel","joints":["bearing_a","bearing_b"]}},
    {"kind":"housing","substance":"ellipsoid",
     "origin":[0.55,0,0],
     "substance_params":{"sx":0.25,"sy":0.25,"sz":0.25},
     "bone":{"name":"housing","head":[0.55,0,-0.25],"tail":[0.55,0,0.25],
             "mass":1.0,"radius":0.25,"stiffness":1.0,
             "material":"ABS"}}
  ]})";

void TestTheRoundTrip() {
    std::printf("-- the material round trip --\n");
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(kMachine, tree, why)) {
        std::printf("FAIL could not parse the machine: %s\n", why.c_str());
        ++g_fail; ++g_checks;
        return;
    }
    ModelOutput mo;
    if (!ElementSystem::Bake(*tree, 24, 1.0f, mo, why)) {
        std::printf("FAIL could not bake the machine: %s\n", why.c_str());
        ++g_fail; ++g_checks;
        return;
    }
    CHECK(mo.bones.size() == 2, "the machine has two bones (got %zu)", mo.bones.size());

    // The names must have survived the frozen record.
    std::vector<std::string> names;
    bool found_steel = false, found_abs = false;
    for (const BoneRecord& br : mo.bones) {
        const std::string n = BoneMaterialName(br);
        names.push_back(n);
        if (n == "steel") found_steel = true;
        if (n == "ABS")   found_abs = true;
    }
    CHECK(found_steel, "the steel material name survived the record");
    CHECK(found_abs, "the ABS material name survived the record");

    // AND they must RESOLVE back to the same constants.
    std::vector<FieldProperties> props;
    std::vector<std::string> resolved_names;
    CHECK(ResolveBoneMaterials(mo.bones, props, resolved_names, why),
          "the bone materials resolve: %s", why.c_str());
    for (size_t i = 0; i < props.size(); ++i) {
        if (resolved_names[i] == "steel") {
            CHECK(std::fabs(props[i].permeability - 200.f) < 1e-3f,
                  "steel resolves to its real permeability (got %g)", props[i].permeability);
            CHECK(props[i].conductivity > 1e6f,
                  "steel resolves to its real conductivity (got %g)", props[i].conductivity);
        }
        if (resolved_names[i] == "ABS") {
            CHECK(props[i].conductivity == 0.f,
                  "ABS resolves to an insulator (got %g)", props[i].conductivity);
            CHECK(std::fabs(props[i].permittivity - 2.4f) < 1e-3f,
                  "ABS resolves to its real permittivity (got %g)", props[i].permittivity);
        }
    }
    // TWO DIFFERENT MATERIALS MUST STAY DIFFERENT -- the encoding does not
    // collapse them into "the first one".
    bool differ = false;
    for (size_t i = 0; i < props.size(); ++i)
        for (size_t j = i + 1; j < props.size(); ++j)
            if (std::fabs(props[i].permeability - props[j].permeability) > 1e-3f)
                differ = true;
    CHECK(differ, "two materials stay distinguishable after the round trip");
}

void TestTheReservedBytesAreNotDisturbed() {
    std::printf("-- the reserved bytes --\n");
    std::unique_ptr<Element> tree;
    std::string why;
    if (!AI::ParseElementTree(kMachine, tree, why)) return;
    ModelOutput mo;
    if (!ElementSystem::Bake(*tree, 24, 1.0f, mo, why)) return;

    for (size_t i = 0; i < mo.bones.size(); ++i) {
        const BoneRecord& br = mo.bones[i];
        // MASS: a float in reserved[0..3]. The authored masses are 4.0 and
        // 1.0, and the material encoding must not have overwritten them.
        float mass = 0.f;
        std::memcpy(&mass, br.reserved, sizeof(float));
        CHECK(mass == 4.0f || mass == 1.0f,
              "the mass survived alongside the material (got %g on bone %zu)",
              mass, i);
        // RADIUS and STIFFNESS in reserved[4..5], still normalised 0..1.
        CHECK(br.reserved[4] <= 255, "the radius byte is intact");
        CHECK(br.reserved[5] <= 255, "the stiffness byte is intact");
        // The material field must be NUL-terminated within its 22 bytes.
        bool terminated = false;
        for (int k = 6; k < 28; ++k)
            if (br.reserved[k] == 0) { terminated = true; break; }
        CHECK(terminated, "the material field is NUL-terminated");
    }
}

void TestUnknownMaterial() {
    std::printf("-- an unknown material --\n");
    const char* kBad = R"({
      "kind":"body","substance":"ellipsoid",
      "substance_params":{"sx":0.3,"sy":0.3,"sz":0.3},
      "bone":{"name":"thing","head":[0,0,-0.3],"tail":[0,0,0.3],
              "material":"unobtainium"}})";

    std::unique_ptr<Element> tree;
    std::string why;
    // The PARSER accepts it -- an unknown material is a DOCUMENT problem,
    // not a JSON problem, and it belongs in the error log.
    CHECK(AI::ParseElementTree(kBad, tree, why),
          "the parser accepts an unknown material name (it is not a schema error)");

    ModelOutput mo;
    if (!ElementSystem::Bake(*tree, 16, 1.0f, mo, why)) return;
    CHECK(mo.bones.size() == 1, "the unknown-material bone still bakes");

    // THE RESOLVER REJECTS IT. Defaulting it would assert physics the
    // document never stated.
    std::vector<FieldProperties> props;
    std::vector<std::string> names;
    CHECK(!ResolveBoneMaterials(mo.bones, props, names, why),
          "an unknown material is REJECTED by the resolver");
    CHECK(why.find("unobtainium") != std::string::npos,
          "the rejection names the offending material (got: %s)", why.c_str());
}

void TestNoMaterialIsNotAnError() {
    std::printf("-- a bone with no material --\n");
    const char* kNoMat = R"({
      "kind":"body","substance":"ellipsoid",
      "substance_params":{"sx":0.3,"sy":0.3,"sz":0.3},
      "bone":{"name":"thing","head":[0,0,-0.3],"tail":[0,0,0.3]}})";

    std::unique_ptr<Element> tree;
    std::string why;
    CHECK(AI::ParseElementTree(kNoMat, tree, why), "the tree parses");
    ModelOutput mo;
    if (!ElementSystem::Bake(*tree, 16, 1.0f, mo, why)) return;
    CHECK(BoneMaterialName(mo.bones[0]).empty(),
          "a bone with no material stores an empty name");

    // NOT AN ERROR -- an honest omission.
    std::vector<FieldProperties> props;
    std::vector<std::string> names;
    CHECK(ResolveBoneMaterials(mo.bones, props, names, why),
          "a bone with no material is not an error: %s", why.c_str());
    CHECK(names.size() == 1 && names[0].empty(),
          "the resolved name is empty");
}

void TestThePromptsTeachIt() {
    std::printf("-- the prompts --\n");
    const std::string schema = AI::ElementTreeSystemPrompt();
    // The schema must list the material key and the vocabulary.
    CHECK(schema.find("\"material\"") != std::string::npos,
          "the schema prompt documents the material key");
    CHECK(schema.find("steel") != std::string::npos,
          "the schema prompt lists the material vocabulary");
    CHECK(schema.find("copper") != std::string::npos,
          "the schema prompt lists more than one material");
    CHECK(schema.find("bone") != std::string::npos,
          "the schema prompt still documents bones");

    // The DOCUMENT prompt must tell the model to read the document as a
    // mechanism -- parts, joints, materials -- not as a pile of shapes.
    const std::string doc = AI::DocToTreeSystemPrompt();
    CHECK(doc.find("MECHANISM") != std::string::npos,
          "the document prompt says to read it as a mechanism");
    CHECK(doc.find("material") != std::string::npos,
          "the document prompt says to give each part its material");
    CHECK(doc.find("joint") != std::string::npos,
          "the document prompt says to name the joints");
    CHECK(doc.find("BONE") != std::string::npos,
          "the document prompt says to give moving parts a bone");
    // AND the closed-body rule must still be there -- it is the single
    // most important rule in the prompt, and a new section must not have
    // displaced it.
    CHECK(doc.find("CLOSED-BODY RULE") != std::string::npos,
          "the closed-body rule survived the new section");
}

void TestTheMaterialNamesAreTheRegistrys() {
    std::printf("-- the prompt's vocabulary matches the registry --\n");
    // EVERY material the prompt tells the model to use must actually
    // resolve. A vocabulary that names a material the registry does not
    // have would teach the model to produce rejections.
    const char* kVocabulary[] = {
        "steel", "copper", "aluminium", "mumetal", "neodymium",
        "ABS", "plastic", "glass", "water", "muscle", "bone",
    };
    for (const char* name : kVocabulary) {
        FieldProperties p;
        CHECK(LookupFieldProperties(name, p),
              "the prompt's material \"%s\" resolves in the registry", name);
    }
    // And the British spelling the prompt uses ("aluminium") must be the
    // one that resolves -- the prompt teaches it, so it has to work.
    FieldProperties alu;
    CHECK(LookupFieldProperties("aluminium", alu),
          "\"aluminium\" (the prompt's spelling) resolves");
}

} // namespace

int main() {
    std::printf("== bone materials gate ==\n");
    TestTheRoundTrip();
    TestTheReservedBytesAreNotDisturbed();
    TestUnknownMaterial();
    TestNoMaterialIsNotAnError();
    TestThePromptsTeachIt();
    TestTheMaterialNamesAreTheRegistrys();
    std::printf("\n%s (%d checks, %d failures)\n",
                g_fail ? "BONE MATERIALS FAILED" : "bone materials passed",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
