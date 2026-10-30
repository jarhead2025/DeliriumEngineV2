// FILEPATH: engine/src/psf/Reconstructor.cpp
#include "delirium/psf/Reconstructor.h"
#include "delirium/psf/PSFBody.h"
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
    return FixedReconstructor{}.Reconstruct(req);
}

} // namespace DeliRium
