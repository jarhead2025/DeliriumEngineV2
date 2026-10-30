#include "generator.h"

namespace delirium {

std::unique_ptr<BodyRuntime> AIGenerator::generate(
    std::string_view semantic,
    MaterialScienceRegistry& mat_science,
    MaterialRegistry& phys_reg)
{
    Domain domain = classifier_.classify(semantic);
    return generate_as(semantic, domain, mat_science, phys_reg);
}

std::unique_ptr<BodyRuntime> AIGenerator::generate_as(
    std::string_view semantic,
    Domain domain,
    MaterialScienceRegistry& mat_science,
    MaterialRegistry& phys_reg)
{
    switch (domain) {
        case Domain::Organic:
            return organic_gen_.generate(semantic, mat_science, phys_reg);
        case Domain::Inorganic:
            return inorganic_gen_.generate(semantic, mat_science, phys_reg);
    }
    return nullptr;
}

} // namespace delirium