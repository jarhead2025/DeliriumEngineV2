// FILEPATH: engine/src/psf/Reconstructor.cpp
#include "delirium/psf/Reconstructor.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/ai/ElementSystem.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

namespace DeliRium {

std::unique_ptr<PSFBody> FixedReconstructor::Reconstruct(
    const ReconstructionRequest& req) {
    std::string path;
    if (req.semantic.rfind("file:", 0) == 0) {
        path = req.semantic.substr(5);
    } else if (!req.semantic.empty() && req.semantic.find('/') != std::string::npos) {
        path = req.semantic;
    } else if (!req.semantic.empty() && req.semantic.find(".psf") != std::string::npos) {
        path = req.semantic;
    }
    if (path.empty()) {
        std::fprintf(stderr, "FixedReconstructor: no file path in semantic '%s'\n",
                     req.semantic.c_str());
        return nullptr;
    }
    auto body = PSFBody::FromFile(path.c_str());
    if (!body) {
        std::fprintf(stderr, "FixedReconstructor: FromFile failed for '%s'\n",
                     path.c_str());
    }
    return body;
}

bool NeuralReconstructor::LoadModel(const char* /*path*/) { return false; }

std::unique_ptr<PSFBody> NeuralReconstructor::Reconstruct(
    const ReconstructionRequest& req) {
    // The semantic request drives the GENERIC element-system generator:
    // a recursive element tree (overlapping systems -> substance leaves).
    // The baker reduces it to PSF records with the engine's own LSQ. A
    // learned categorizer later emits element trees through this same seam.
    auto tree = ProceduralElementGenerator::Build(req.semantic);
    if (!tree) return nullptr;
    // Grid density from the patch budget: (grid_n-1)^2 patches <= max_patches.
    int grid_n = 10;
    if (req.max_patches >= 4) {
        int g = (int)std::sqrt((double)req.max_patches) + 1;
        if (g < 4) g = 4; else if (g > 64) g = 64;
        grid_n = g;
    }
    std::string why;
    auto body = ElementSystem::BakeBody(*tree, grid_n, 1.0f, why, req.world_offset);
    if (!body)
        std::fprintf(stderr, "NeuralReconstructor: element bake failed for '%s': %s\n",
                     req.semantic.c_str(), why.c_str());
    return body;
}

} // namespace DeliRium
