// ============================================================================
// ScriptRuntime.h — WS5 script runtime header
// ============================================================================
// M5 scaffolding: script execution + hot reload. Commands:
//   scene load, object add, field set, physics step, render path, build package.
// Part of v2.2Beta milestone M5 (WS5).
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>

namespace DeliRium {

// Forward declarations
class PhysicsWorld;
class Scene;
class PSFBody;
class GameObject;
class ClingInterpreter;

// ============================================================================
// ScriptCommand — a parsed script command
// ============================================================================
struct ScriptCommand {
    std::string type;       // Command type (scene/object/field/physics/render/build)
    std::vector<std::string> args; // Command arguments
    int line;               // Source line number

    ScriptCommand() : line(0) {}
    ScriptCommand(const std::string& t, const std::vector<std::string>& a, int l)
        : type(t), args(a), line(l) {}
};

// ============================================================================
// ScriptRuntime — WS5 script runtime engine
// ============================================================================
// Manages script execution lifecycle:
// - Parse script commands (.drs files)
// - Execute commands against PhysicsWorld/Scene state
// - Hot reload: edit script → re-run without restart
// - Debug: step through script, inspect variables, set breakpoints
//
// Usage:
//   ScriptRuntime runtime(world);
//   runtime.LoadScript("scene.drs");
//   runtime.Run();
//   runtime.HotReload(); // After editing script
// ============================================================================
class ScriptRuntime {
public:
    explicit ScriptRuntime(PhysicsWorld* world);
    ~ScriptRuntime();

    // Disable copy/move
    ScriptRuntime(const ScriptRuntime&) = delete;
    ScriptRuntime& operator=(const ScriptRuntime&) = delete;

    // Load a script file (.drs)
    bool LoadScript(const std::string& file_path);

    // Load a script from a string (for testing / inline scripts)
    bool LoadScriptFromString(const std::string& content);

    // Run the loaded script from start
    bool Run();

    // Run from a specific line (for debugging)
    bool RunFromLine(int line);

    // Hot reload: re-parse script and apply changes without restart
    bool HotReload();

    // Step to next command (debug mode)
    bool Step();

    // Set a breakpoint at line
    void SetBreakpoint(int line, bool enabled);

    // Check if breakpoint is set at line
    bool HasBreakpoint(int line) const;

    // Pause execution
    void Pause();

    // Resume execution
    void Resume();

    // Stop execution
    void Stop();

    // Check if running
    bool IsRunning() const { return running_; }

    // Check if paused
    bool IsPaused() const { return paused_; }

    // Get current line being executed
    int CurrentLine() const { return current_line_; }

    // Get current command
    const ScriptCommand& CurrentCommand() const { return current_command_; }

    // Get script commands (for UI display)
    const std::vector<ScriptCommand>& Commands() const { return commands_; }

    // Get last error message
    const std::string& LastError() const { return last_error_; }

    // Set verbose mode
    void SetVerbose(bool verbose) { verbose_ = verbose; }

    // Get physics world reference
    PhysicsWorld* GetPhysicsWorld() const { return world_; }

    // Get scene reference
    Scene* GetScene() const;

private:
    // Parse script file into commands
    bool ParseScript(const std::string& content);

    // Execute a single command
    bool ExecuteCommand(const ScriptCommand& cmd);

    // Handle scene commands
    bool HandleSceneCommand(const ScriptCommand& cmd);

    // Handle object commands
    bool HandleObjectCommand(const ScriptCommand& cmd);

    // Handle field commands
    bool HandleFieldCommand(const ScriptCommand& cmd);

    // Handle physics commands
    bool HandlePhysicsCommand(const ScriptCommand& cmd);

    // Handle render commands
    bool HandleRenderCommand(const ScriptCommand& cmd);

    // Handle build commands
    bool HandleBuildCommand(const ScriptCommand& cmd);

    PhysicsWorld* world_;              // Physics world (read/write state)
    std::string script_path_;          // Path to loaded script
    std::string script_content_;       // Loaded script content
    std::vector<ScriptCommand> commands_; // Parsed commands
    int current_line_;                 // Current execution line
    ScriptCommand current_command_;    // Current command being executed
    bool running_;                     // Whether script is running
    bool paused_;                      // Whether execution is paused
    bool verbose_;                     // Print execution messages
    std::string last_error_;           // Last error message
    std::vector<bool> breakpoints_;    // Breakpoint flags per line
};

} // namespace delirium
