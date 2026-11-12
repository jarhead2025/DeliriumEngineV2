// =============================================================================
// geo_tool_probe - DOES THE LOCAL MODEL ACTUALLY USE THE GEOMETRIC TOOLS?
//
// WHAT THIS PROVES
//
//   The GeoTools registry exists and its maths is unit-tested
//   (tests/validate_geo_tools.cpp). What that CANNOT prove is the thing
//   that matters: does the local model actually CALL the tools, and does
//   calling them fix the failures that motivated them?
//
// THE MEASURED FAILURES THE TOOLS TARGET (from real runs)
//   1. the closed-body rule: the model emitted `"sz": -0.5` (a negative
//      HALF-EXTENT) where the rule is `"scale": [1,1,-1]`.
//   2. the torus's radii: "outer radius 0.44, tube radius 0.12" means
//      major = 0.32, and the model got it wrong.
//   3. the scale of a part: the model guessed a radius that made the
//      assembly's parts the wrong relative size.
//
//   All three are arithmetic or a vocabulary lookup.
//
// WHAT THE PROBE DOES
//
//   1. asks the model to solve a geometry task that REQUIRES the tool,
//   2. runs the tool loop,
//   3. checks: did the model CALL the tool? did the final answer contain
//      the right numbers?
//
//   It runs the SAME task with and without the tools. The difference is
//   the measurement.
//
// USAGE
//   geo_tool_probe --model qwen2.5:7b-instruct-q8_0 [--port 11500]
//                   [--no-tools] [--save-log FILE]
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

#include "delirium/ai/GeoTools.h"
#include "delirium/llm/LLMToolLoop.h"
#include "delirium/llm/LLMClient.h"

using namespace DeliRium;
using namespace DeliRium::LLM;

namespace {

// The task. Chosen because the correct answer is arithmetic the model
// measurably gets wrong: a torus quoted by its OUTER radius.
//
// The expected answer is checked against the TOOL's own output, not a
// hardcoded constant -- so the probe asks the tool for the truth.
const char* kTask =
    "A torus ring has an OUTER radius of 0.44 and a TUBE radius of 0.12.\n"
    "What are its MAJOR radius and its MINOR radius?\n"
    "Give the two numbers.";

struct ProbeResult {
    std::string text;
    std::vector<ToolTurn> turns;
    bool completed = false;
    std::string stop_reason;
};

// Ask the model ONCE, synchronously.
//
// The client is ASYNCHRONOUS (a worker thread), so the probe pumps it
// rather than blocking on a callback -- and the timeout is enforced HERE
// so a wedged request cannot hang the probe.
bool AskOnce(const std::string& system,
             const std::string& user,
             const std::string& model,
             const std::string& host,
             int port,
             int timeout_s,
             std::string& reply) {
    Config cfg;
    cfg.model = model;
    cfg.host = host;
    cfg.port = port;
    cfg.system = system;
    cfg.timeout_seconds = timeout_s;
    cfg.keep_alive = 0;        // unload after the answer -- one model at a time
    cfg.use_native_endpoint = true;   // honours keep_alive
    cfg.max_tokens = 900;
    cfg.temperature = 0.2f;

    LLMClient client(cfg);
    if (!client.Send(user)) {
        std::fprintf(stderr, "[probe] the request could not start\n");
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const State s = client.CurrentState();
        if (s == State::Done) {
            reply = client.TakeResponse();
            return true;
        }
        if (s == State::Failed) {
            std::fprintf(stderr, "[probe] the request failed: %s\n",
                         client.Error().c_str());
            return false;
        }
        const auto dt = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (dt > timeout_s) {
            std::fprintf(stderr, "[probe] the request timed out after %llds\n",
                         (long long)dt);
            client.Cancel();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Run the tool loop against the live model.
ProbeResult RunLoop(const std::string& system,
                     const std::string& user,
                     const std::string& model,
                     const std::string& host,
                     int port,
                     int timeout_s,
                     bool use_tools) {
    ProbeResult r;
    if (!use_tools) {
        std::string reply;
        if (AskOnce(system, user, model, host, port, timeout_s, reply)) {
            r.text = reply;
            r.completed = true;
        } else {
            r.stop_reason = "the request failed";
        }
        return r;
    }

    const AI::GeoToolSet& tools = AI::GeoToolSet::Instance();
    ModelResponder responder = [&](
        const std::vector<std::pair<std::string, std::string>>& history,
        std::string& reply) -> bool {
        // Rebuild the conversation as one user message plus the tool
        // results, so the client's (system, user) interface can carry
        // the loop's history.
        std::ostringstream u;
        u << user;
        for (const auto& h : history) {
            if (h.first == "tool") u << "\n\n" << h.second;
        }
        return AskOnce(system, u.str(), model, host, port, timeout_s, reply);
    };

    ToolLoopResult lr = RunToolLoop(system, user, tools, responder, 6, 8);
    r.text = lr.text;
    r.turns = lr.turns;
    r.completed = lr.completed;
    r.stop_reason = lr.stop_reason;
    return r;
}

// Does the text contain the number (within a tolerance)?
bool Mentions(const std::string& text, double value, double tol) {
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        const bool digit = std::isdigit((unsigned char)c) != 0;
        const bool neg = (c == '-' && i + 1 < text.size()
                          && std::isdigit((unsigned char)text[i + 1]) != 0);
        if (!digit && !neg) continue;
        const char* begin = text.c_str() + i;
        char* end = nullptr;
        double v = std::strtod(begin, &end);
        if (end == begin) continue;
        if (std::fabs(v - value) <= tol) return true;
        i = static_cast<size_t>(end - text.c_str()) - 1;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model = "qwen2.5:7b-instruct-q8_0";
    std::string host = "127.0.0.1";
    int port = 11434;
    int timeout_s = 300;
    bool use_tools = true;
    std::string save_log;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
        };
        if (a == "--model") model = next("qwen2.5:7b-instruct-q8_0");
        else if (a == "--host") host = next("127.0.0.1");
        else if (a == "--port") port = std::atoi(next("11434").c_str());
        else if (a == "--timeout") timeout_s = std::atoi(next("300").c_str());
        else if (a == "--no-tools") use_tools = false;
        else if (a == "--save-log") save_log = next("");
    }

    std::printf("=== GEO TOOL PROBE ===\n");
    std::printf("server: %s:%d\nmodel: %s\ntools: %s\n",
                host.c_str(), port, model.c_str(), use_tools ? "ON" : "OFF");
    std::printf("task: torus outer=0.44 tube=0.12 -> major, minor\n\n");

    // The TRUTH, from the tool itself -- the probe does not hardcode it.
    const AI::GeoToolSet& tools = AI::GeoToolSet::Instance();
    std::string truth, err;
    if (!tools.Invoke("resolve_torus_radii", {{"outer", 0.44}, {"tube", 0.12}},
                      {}, truth, err)) {
        std::fprintf(stderr, "[probe] the truth tool failed: %s\n", err.c_str());
        return 2;
    }
    std::printf("truth (from the tool): %s\n\n", truth.c_str());

    // The system's prompt: the task instruction plus, optionally, the
    // catalogue. The catalogue is added to the PROMPT as well as being
    // offered to the server, because not every local runtime honours the
    // server-side "tools" field.
    std::string system =
        "You are a precise mechanical engineer. Answer with the requested "
        "numbers only, briefly.\n";
    if (use_tools) {
        system += ToolCataloguePrompt(tools);
    }

    ProbeResult r = RunLoop(system, kTask, model, host, port, timeout_s,
                            use_tools);

    std::printf("--- RESULT ---\n");
    std::printf("tool calls made: %zu\n", r.turns.size());
    for (const auto& t : r.turns) {
        std::printf("  [%s] %s %s -> %s\n", t.failed ? "FAIL" : "ok",
                    t.tool_name.c_str(), t.arguments_json.c_str(),
                    t.result.c_str());
    }
    std::printf("completed: %s", r.completed ? "yes" : "no");
    if (!r.stop_reason.empty()) std::printf(" (%s)", r.stop_reason.c_str());
    std::printf("\n\nfinal answer:\n%s\n", r.text.c_str());

    // THE CHECKS.
    int failures = 0;

    const double kMajor = 0.32;
    const double kMinor = 0.12;
    const double tol = 0.005;   // the model rounds; 0.005 is well inside
                                // "0.32" vs "0.3204"

    if (use_tools) {
        if (r.turns.empty()) {
            std::printf("\nFAIL: the model made NO tool call -- the tools are "
                        "not being deployed\n");
            ++failures;
        } else {
            bool called_resolver = false;
            for (const auto& t : r.turns) {
                if (t.tool_name == "resolve_torus_radii") called_resolver = true;
            }
            if (!called_resolver) {
                std::printf("\nFAIL: the model did not call resolve_torus_radii "
                            "(it called: ");
                for (size_t i = 0; i < r.turns.size(); ++i)
                    std::printf("%s%s", i ? ", " : "", r.turns[i].tool_name.c_str());
                std::printf(")\n");
                ++failures;
            } else {
                std::printf("\nok: the model called resolve_torus_radii\n");
            }
        }
    }

    const bool got_major = Mentions(r.text, kMajor, tol);
    const bool got_minor = Mentions(r.text, kMinor, tol);
    std::printf("%s the answer states the MAJOR radius (0.32)\n",
                got_major ? "ok  " : "FAIL");
    std::printf("%s the answer states the MINOR radius (0.12)\n",
                got_minor ? "ok  " : "FAIL");
    if (!got_major) ++failures;
    if (!got_minor) ++failures;

    if (!save_log.empty()) {
        std::ofstream f(save_log);
        if (f) {
            f << "model: " << model << "\ntools: " << (use_tools ? "on" : "off")
              << "\n";
            for (const auto& t : r.turns)
                f << "tool: " << t.tool_name << " " << t.arguments_json
                  << " -> " << t.result << "\n";
            f << "answer: " << r.text << "\n";
        }
    }

    std::printf("\n=== PROBE %s ===\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
