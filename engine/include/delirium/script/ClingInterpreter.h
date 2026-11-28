// ============================================================================
// ClingInterpreter.h — WS5 Cling scripting interpreter header
// ============================================================================
// M5 scaffolding: wraps Cling (LLVM-based C++ interpreter) to expose engine
// API to scripts. Scripts can call PhysicsWorld, Scene, PSFBody APIs directly.
// Part of v2.2Beta milestone M5 (WS5).
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

namespace DeliRium {

// Forward declarations
class PhysicsWorld;
class Scene;
class PSFBody;
class GameObject;

// ============================================================================
// ScriptError — error information from script compilation/execution
// ============================================================================
struct ScriptError {
    std::string message;
    std::string file;
    int line;
    int column;
    bool is_fatal;

    ScriptError() : line(0), column(0), is_fatal(false) {}
    ScriptError(const std::string& msg, const std::string& f, int l, int c, bool fatal)
        : message(msg), file(f), line(l), column(c), is_fatal(fatal) {}
};

// ============================================================================
// ScriptResult — result of script compilation/execution
// ============================================================================
struct ScriptResult {
    bool success;
    std::string output;
    std::vector<ScriptError> errors;
    double execution_time_ms;

    ScriptResult() : success(false), execution_time_ms(0.0) {}
};

// ============================================================================
// ClingInterpreter — WS5 Cling scripting interpreter
// ============================================================================
// Wraps Cling (LLVM-based C++ interpreter) to expose engine API to scripts.
// Scripts are C++ files that call engine API directly via #include.
//
// Usage:
//   ClingInterpreter interp(world);
//   interp.AddIncludePath("engine/include");
//   interp.LoadScript("my_script.cpp");
//   interp.Execute();
//
// Commands available to scripts:
//   scene load <file.psf>       — load a PSF body into scene
//   object add <name> <type>    — add object to scene
//   field set <name> <value>    — set physical field value
//   physics step <dt>           — step physics simulation
//   render path <path>          — set render path (march/tri_infill)
//   build package <output>      — build standalone executable
// ============================================================================
class ClingInterpreter {
public:
    explicit ClingInterpreter(PhysicsWorld* world);
    ~ClingInterpreter();

    // Disable copy/move
    ClingInterpreter(const ClingInterpreter&) = delete;
    ClingInterpreter& operator=(const ClingInterpreter&) = delete;

    // Add include path for script #include directives
    void AddIncludePath(const std::string& path);

    // Add library path for linking
    void AddLibraryPath(const std::string& path);

    // Load a script file (does not execute yet)
    bool LoadScript(const std::string& file_path);

    // Execute the currently loaded script
    ScriptResult Execute();

    // Compile a script without executing (syntax check)
    ScriptResult CompileOnly(const std::string& script);

    // Evaluate a single expression
    ScriptResult Evaluate(const std::string& expression);

    // Get the last execution result
    const ScriptResult& LastResult() const { return last_result_; }

    // Check if Cling is available (LLVM libs loaded)
    bool IsAvailable() const { return available_; }

    // Set verbose mode (print compilation messages)
    void SetVerbose(bool verbose) { verbose_ = verbose; }

    // Register a native function callable from scripts
    void RegisterFunction(const std::string& name, std::function<void()> fn);

    // Get scene reference for scripts
    Scene* GetScene() const;

    // Get physics world reference for scripts
    PhysicsWorld* GetPhysicsWorld() const { return world_; }

private:
    // Initialize Cling interpreter (load LLVM libs)
    bool Initialize();

    // Shutdown Cling interpreter
    void Shutdown();

    // Add engine API bindings to Cling
    void BindEngineAPI();

    // Parse script commands and dispatch
    void DispatchCommands(const std::string& script_content);

    PhysicsWorld* world_;              // Physics world (read/write state)
    bool available_;                   // Whether Cling is initialized
    bool verbose_;                     // Print compilation messages
    std::vector<std::string> include_paths_; // Include paths for scripts
    std::vector<std::string> library_paths_; // Library paths for linking
    std::string loaded_script_;        // Currently loaded script content
    ScriptResult last_result_;         // Last execution result
    std::vector<std::pair<std::string, std::function<void()>>> native_functions_; // Native functions
    void* cling_instance_;             // Opaque Cling instance pointer
};

} // namespace delirium
