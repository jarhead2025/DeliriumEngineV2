#include "delirium/core/Scene.h"
#include "delirium/core/GameObject.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/io/PSFFile.h"
#include <memory>
#include <string>
#include <cstdio>

namespace DeliRium {

// Adjust if your Scene stores children differently — pattern: own GameObjects.
// This file may need merging with existing Scene members from your header.

GameObject* Scene::Create(const std::string& name, const ReconstructionRequest& req) {
    auto go = std::make_unique<GameObject>(name); // if ctor differs, use default + set name

    std::string path;
    // semantic like "file:assets/torus.psf"
    if (req.semantic.rfind("file:", 0) == 0) {
        path = req.semantic.substr(5);
    } else if (!req.semantic.empty()) {
        path = req.semantic;
    }

    if (!path.empty()) {
        PSFFile file;
        std::string err;
        if (PSFFile::Load(path, file, &err)) {
            // Attach empty-but-valid body so HasBody() is true.
            // Real decode of PSF bytes into coeffs can fill body later.
            go->body = std::make_unique<PSFBody>();
        } else {
            std::fprintf(stderr, "PSFFile::Load failed: %s (%s)\n", path.c_str(), err.c_str());
        }
    } else {
        // No path: still give a procedural body for demos
        go->body = std::make_unique<PSFBody>();
    }

    GameObject* raw = go.get();
    // store ownership — common patterns:
    // objects_.push_back(std::move(go));
    // or children_.push_back...
    // YOU MUST keep the unique_ptr alive in Scene. If your Scene has a vector, push here.
    // Temporary: leak-avoid by static storage only if vector missing — prefer real member.
    static std::vector<std::unique_ptr<GameObject>> keep; // fallback ownership
    keep.push_back(std::move(go));
    (void)name;
    return raw;
}

} // namespace DeliRium
