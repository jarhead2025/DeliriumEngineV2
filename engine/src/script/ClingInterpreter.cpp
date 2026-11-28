// ============================================================================
// ClingInterpreter.cpp — WS5 Cling scripting interpreter implementation
// ============================================================================
// M5 scaffolding: wraps Cling (LLVM-based C++ interpreter) to expose engine
// API to scripts. Scripts can call PhysicsWorld, Scene, PSFBody APIs directly.
// Part of v2.2Beta milestone M5 (WS5).
// ============================================================================

#include "delirium/script/ClingInterpreter.h"
#include "delirium/physics/PhysicsWorld.h"
#include "delirium/core/Scene.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/GameObject.h"

#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

namespace DeliRium {

ClingInterpreter::ClingInterpreter(PhysicsWorld* world)
    : world_(world), available_(false), verbose_(false), cling_instance_(nullptr) {
    if (!world_) {
        world_ = new PhysicsWorld();
    }
    Initialize();
}

ClingInterpreter::~ClingInterpreter() {
    Shutdown();
}

bool ClingInterpreter::Initialize() {
    // Placeholder: actual Cling initialization requires LLVM libs
    // In production, this would:
    // 1. Load LLVM shared libraries
    // 2. Initialize ClingInterpreter instance
    // 3. Add include/library paths
    // 4. Bind engine API functions
    available_ = true; // Stub — always available for scaffolding
    if (verbose_) {
        std::cout << "[cling] Interpreter initialized (stub mode)" << std::endl;
    }
    return true;
}

void ClingInterpreter::Shutdown() {
    // Placeholder: actual Cling shutdown
    cling_instance_ = nullptr;
    available_ = false;
}

void ClingInterpreter::AddIncludePath(const std::string& path) {
    include_paths_.push_back(path);
    if (verbose_ && available_) {
        std::cout << "[cling] Added include path: " << path << std::endl;
    }
}

void ClingInterpreter::AddLibraryPath(const std::string& path) {
    library_paths_.push_back(path);
    if (verbose_ && available_) {
        std::cout << "[cling] Added library path: " << path << std::endl;
    }
}

bool ClingInterpreter::LoadScript(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        last_result_.success = false;
        last_result_.errors.emplace_back("Failed to open script file: " + file_path, file_path, 0, 0, true);
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    loaded_script_ = ss.str();
    if (verbose_) {
        std::cout << "[cling] Loaded script: " << file_path << " (" << loaded_script_.size() << " bytes)" << std::endl;
    }
    return true;
}

ScriptResult ClingInterpreter::Execute() {
    auto start = std::chrono::high_resolution_clock::now();
    ScriptResult result;

    if (!available_ || loaded_script_.empty()) {
        result.success = false;
        result.errors.emplace_back("Interpreter not available or no script loaded", "", 0, 0, true);
        last_result_ = result;
        return result;
    }

    // Parse script commands and dispatch
    DispatchCommands(loaded_script_);

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.success = true; // Stub — always succeeds in scaffolding mode
    last_result_ = result;
    return result;
}

ScriptResult ClingInterpreter::CompileOnly(const std::string& script) {
    ScriptResult result;
    // Placeholder: actual syntax check via Cling
    result.success = true;
    return result;
}

ScriptResult ClingInterpreter::Evaluate(const std::string& expression) {
    ScriptResult result;
    // Placeholder: actual expression evaluation via Cling
    result.success = true;
    return result;
}

Scene* ClingInterpreter::GetScene() const {
    if (!world_) return nullptr;
    return world_->GetScene();
}

void ClingInterpreter::RegisterFunction(const std::string& name, std::function<void()> fn) {
    native_functions_.emplace_back(name, fn);
    if (verbose_) {
        std::cout << "[cling] Registered function: " << name << std::endl;
    }
}

void ClingInterpreter::BindEngineAPI() {
    // Placeholder: actual Cling API binding
    // In production, this would expose:
    // - PhysicsWorld methods
    // - Scene methods
    // - PSFBody methods
    // - GameObject methods
    // - Force module functions
    // - Render path functions
    if (verbose_) {
        std::cout << "[cling] Engine API bindings registered" << std::endl;
    }
}

void ClingInterpreter::DispatchCommands(const std::string& script_content) {
    // Simple command parser for scaffolding
    // In production, this would use Cling to parse and execute C++ code
    std::istringstream iss(script_content);
    std::string line;
    int line_num = 0;

    while (std::getline(iss, line)) {
        line_num++;
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.empty() || line[0] == '/') continue; // Skip comments/blank

        // Parse commands (simplified for scaffolding)
        if (line.find("scene load") == 0) {
            // Placeholder: load PSF body
            if (verbose_) std::cout << "[cling] scene load command" << std::endl;
        } else if (line.find("object add") == 0) {
            // Placeholder: add object to scene
            if (verbose_) std::cout << "[cling] object add command" << std::endl;
        } else if (line.find("physics step") == 0) {
            // Placeholder: step physics
            if (verbose_) std::cout << "[cling] physics step command" << std::endl;
        } else if (line.find("render path") == 0) {
            // Placeholder: set render path
            if (verbose_) std::cout << "[cling] render path command" << std::endl;
        } else if (line.find("build package") == 0) {
            // Placeholder: build standalone executable
            if (verbose_) std::cout << "[cling] build package command" << std::endl;
        }
    }
}

} // namespace delirium
