// ============================================================================
// ScriptRuntime.cpp — WS5 script runtime engine implementation
// ============================================================================
// M5 scaffolding: script execution + hot reload. Commands:
//   scene load, object add, field set, physics step, render path, build package.
// Part of v2.2Beta milestone M5 (WS5).
// ============================================================================

#include "delirium/script/ScriptRuntime.h"
#include "delirium/physics/PhysicsWorld.h"
#include "delirium/core/Scene.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/GameObject.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace delirium {

ScriptRuntime::ScriptRuntime(PhysicsWorld* world)
    : world_(world), current_line_(0), running_(false), paused_(false), verbose_(false) {
    if (!world_) {
        world_ = new PhysicsWorld();
    }
}

ScriptRuntime::~ScriptRuntime() {
    Stop();
}

Scene* ScriptRuntime::GetScene() const {
    if (!world_) return nullptr;
    return world_->GetScene();
}

bool ScriptRuntime::LoadScript(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        last_error_ = "Failed to open script file: " + file_path;
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    script_path_ = file_path;
    script_content_ = ss.str();
    if (!ParseScript(script_content_)) {
        return false;
    }
    if (verbose_) {
        std::cout << "[script] Loaded " << commands_.size() << " commands from " << file_path << std::endl;
    }
    return true;
}

bool ScriptRuntime::ParseScript(const std::string& content) {
    commands_.clear();
    breakpoints_.clear();
    std::istringstream iss(content);
    std::string line;
    int line_num = 0;

    while (std::getline(iss, line)) {
        line_num++;
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) {
            breakpoints_.push_back(false);
            continue;
        }
        line = line.substr(start);
        if (line.empty() || line[0] == '/' || line[0] == '#') {
            breakpoints_.push_back(false);
            continue; // Skip comments/blank lines
        }

        // Parse command type and arguments
        std::istringstream cmd_iss(line);
        std::string type;
        cmd_iss >> type;

        ScriptCommand cmd(type, {}, line_num);
        std::string arg;
        while (cmd_iss >> arg) {
            cmd.args.push_back(arg);
        }
        commands_.push_back(cmd);
        breakpoints_.push_back(false);
    }
    return true;
}

bool ScriptRuntime::Run() {
    if (commands_.empty()) {
        last_error_ = "No commands to execute";
        return false;
    }
    running_ = true;
    paused_ = false;
    current_line_ = 0;

    for (size_t i = 0; i < commands_.size(); i++) {
        if (!running_) break;
        while (paused_) {
            // Wait for resume (in production, this would yield to event loop)
        }
        current_line_ = commands_[i].line;
        current_command_ = commands_[i];

        if (breakpoints_[i]) {
            paused_ = true;
            if (verbose_) {
                std::cout << "[script] Breakpoint at line " << current_line_ << std::endl;
            }
        }

        if (!ExecuteCommand(commands_[i])) {
            running_ = false;
            return false;
        }
    }
    running_ = false;
    return true;
}

bool ScriptRuntime::RunFromLine(int line) {
    for (size_t i = 0; i < commands_.size(); i++) {
        if (commands_[i].line >= line) {
            current_line_ = i;
            return Run();
        }
    }
    last_error_ = "Line not found in script";
    return false;
}

bool ScriptRuntime::HotReload() {
    if (script_path_.empty()) {
        last_error_ = "No script loaded";
        return false;
    }
    std::ifstream file(script_path_);
    if (!file.is_open()) {
        last_error_ = "Failed to reopen script file: " + script_path_;
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    script_content_ = ss.str();
    bool ok = ParseScript(script_content_);
    if (ok && verbose_) {
        std::cout << "[script] Hot reload: " << commands_.size() << " commands" << std::endl;
    }
    return ok;
}

bool ScriptRuntime::Step() {
    if (commands_.empty() || current_line_ >= static_cast<int>(commands_.size())) {
        last_error_ = "No more commands to step";
        return false;
    }
    if (!ExecuteCommand(commands_[current_line_])) {
        return false;
    }
    current_line_++;
    return true;
}

void ScriptRuntime::SetBreakpoint(int line, bool enabled) {
    for (size_t i = 0; i < commands_.size(); i++) {
        if (commands_[i].line == line) {
            breakpoints_[i] = enabled;
            return;
        }
    }
}

bool ScriptRuntime::HasBreakpoint(int line) const {
    for (size_t i = 0; i < commands_.size(); i++) {
        if (commands_[i].line == line) {
            return breakpoints_[i];
        }
    }
    return false;
}

void ScriptRuntime::Pause() {
    paused_ = true;
}

void ScriptRuntime::Resume() {
    paused_ = false;
}

void ScriptRuntime::Stop() {
    running_ = false;
    paused_ = false;
}

bool ScriptRuntime::ExecuteCommand(const ScriptCommand& cmd) {
    if (verbose_) {
        std::cout << "[script] Line " << cmd.line << ": " << cmd.type;
        for (const auto& arg : cmd.args) std::cout << " " << arg;
        std::cout << std::endl;
    }

    if (cmd.type == "scene") return HandleSceneCommand(cmd);
    if (cmd.type == "object") return HandleObjectCommand(cmd);
    if (cmd.type == "field") return HandleFieldCommand(cmd);
    if (cmd.type == "physics") return HandlePhysicsCommand(cmd);
    if (cmd.type == "render") return HandleRenderCommand(cmd);
    if (cmd.type == "build") return HandleBuildCommand(cmd);

    last_error_ = "Unknown command: " + cmd.type;
    return false;
}

bool ScriptRuntime::HandleSceneCommand(const ScriptCommand& cmd) {
    if (cmd.args.empty()) {
        last_error_ = "scene command requires subcommand";
        return false;
    }
    if (cmd.args[0] == "load" && cmd.args.size() > 1) {
        // Placeholder: load PSF body into scene
        if (verbose_) std::cout << "[script] Loading scene: " << cmd.args[1] << std::endl;
        return true;
    }
    if (cmd.args[0] == "clear") {
        // Placeholder: clear scene
        if (verbose_) std::cout << "[script] Clearing scene" << std::endl;
        return true;
    }
    last_error_ = "Unknown scene subcommand: " + cmd.args[0];
    return false;
}

bool ScriptRuntime::HandleObjectCommand(const ScriptCommand& cmd) {
    if (cmd.args.empty()) {
        last_error_ = "object command requires subcommand";
        return false;
    }
    if (cmd.args[0] == "add" && cmd.args.size() >= 3) {
        // Placeholder: add object to scene
        if (verbose_) std::cout << "[script] Adding object: " << cmd.args[1] << " type=" << cmd.args[2] << std::endl;
        return true;
    }
    if (cmd.args[0] == "remove" && cmd.args.size() > 1) {
        // Placeholder: remove object from scene
        if (verbose_) std::cout << "[script] Removing object: " << cmd.args[1] << std::endl;
        return true;
    }
    last_error_ = "Unknown object subcommand: " + cmd.args[0];
    return false;
}

bool ScriptRuntime::HandleFieldCommand(const ScriptCommand& cmd) {
    if (cmd.args.size() < 3) {
        last_error_ = "field command requires name and value";
        return false;
    }
    // Placeholder: set physical field value
    if (verbose_) std::cout << "[script] Setting field " << cmd.args[0] << " = " << cmd.args[1] << std::endl;
    return true;
}

bool ScriptRuntime::HandlePhysicsCommand(const ScriptCommand& cmd) {
    if (cmd.args.empty()) {
        last_error_ = "physics command requires subcommand";
        return false;
    }
    if (cmd.args[0] == "step" && cmd.args.size() > 1) {
        // Placeholder: step physics simulation
        float dt = std::stof(cmd.args[1]);
        if (verbose_) std::cout << "[script] Physics step dt=" << dt << std::endl;
        return true;
    }
    last_error_ = "Unknown physics subcommand: " + cmd.args[0];
    return false;
}

bool ScriptRuntime::HandleRenderCommand(const ScriptCommand& cmd) {
    if (cmd.args.empty()) {
        last_error_ = "render command requires subcommand";
        return false;
    }
    if (cmd.args[0] == "path" && cmd.args.size() > 1) {
        // Placeholder: set render path
        if (verbose_) std::cout << "[script] Render path: " << cmd.args[1] << std::endl;
        return true;
    }
    last_error_ = "Unknown render subcommand: " + cmd.args[0];
    return false;
}

bool ScriptRuntime::HandleBuildCommand(const ScriptCommand& cmd) {
    if (cmd.args.empty()) {
        last_error_ = "build command requires subcommand";
        return false;
    }
    if (cmd.args[0] == "package" && cmd.args.size() > 1) {
        // Placeholder: build standalone executable
        if (verbose_) std::cout << "[script] Building package: " << cmd.args[1] << std::endl;
        return true;
    }
    last_error_ = "Unknown build subcommand: " + cmd.args[0];
    return false;
}

} // namespace delirium
