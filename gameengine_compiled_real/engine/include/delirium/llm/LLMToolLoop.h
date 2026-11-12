// =============================================================================
// LLMToolLoop.h - THE TOOL-CALLING LOOP.
//
// WHAT THIS IS
//
// The local model cannot reliably do arithmetic in its head, but it CAN
// reliably CALL a function that does it. MEASURED against the local
// qwen2.5:7b: asked "what is the volume of a sphere of radius 3, use the
// tool", it emits a proper Ollama tool-call block:
//
//     "tool_calls":[{"id":"call_...","function":{"name":"sphere_volume",
//                  "arguments":{"r":3}}}]
//
// So the loop is the standard one:
//
//     1. ask the model, WITH the tool catalogue
//     2. if the model asked for a tool, RUN it, and put the result into
//        the conversation as the tool's answer
//     3. ask again, with the result in the history
//     4. repeat until the model answers with text, or the budget is spent
//
// WHY THE LOOP LIVES HERE AND NOT IN THE WORKSPACE
//
// Because the tools are a CAPABILITY of the engine, not of one UI. The
// workspace's semantic prompt, the document learner and the escalation
// ladder all need the same loop -- and a tool set that only exists in one
// path is a tool set that gets forgotten. (The same argument the tool
// registry itself makes about living in the engine.)
//
// THE BUDGET IS EXPLICIT
//
// A model that keeps calling tools must be stopped, or the loop runs
// forever. The budget is a maximum number of tool CALLS (not turns), and
// when it is exhausted the last assistant text is returned along with a
// note -- so the caller gets a usable answer rather than a hang.
//
// EVERY TOOL CALL IS LOGGED
//
// The transcript records what was asked, what the tool answered, and how
// many calls were made. That is the evidence a learner needs: a successful
// document becomes a worked example (document + the tools it used +
// the tree it produced), and a failure becomes a labelled one.
// =============================================================================
#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>

#include "delirium/ai/GeoTools.h"

namespace DeliRium {
namespace LLM {

// One turn of the tool conversation, for the transcript.
struct ToolTurn {
    std::string tool_name;
    std::string arguments_json;
    std::string result;
    bool        failed = false;
};

struct ToolLoopResult {
    // The model's final TEXT (the answer the caller wants).
    std::string text;
    // The tool calls that were made, in order.
    std::vector<ToolTurn> turns;
    // True if the model produced a final answer (rather than the budget
    // running out).
    bool completed = false;
    // Why the loop ended, if it did not complete.
    std::string stop_reason;
};

// A callback that performs ONE request against the model.
//
// WHY A CALLBACK AND NOT THE CLIENT DIRECTLY: the loop is synchronous
// (ask, run the tool, ask again), while the LLM client is asynchronous
// (it streams on a worker thread, so the render loop keeps running).
// The CALLER owns that asynchrony -- it pumps the client until it
// finishes, then calls back in. So the loop itself is synchronous and
// testable, and the asynchrony stays in one place.
//
// The callback receives the conversation (oldest first) and returns the
// model's reply. The reply may be TEXT or a TOOL CALL -- the loop
// inspects it.
using ModelResponder = std::function<bool(
    const std::vector<std::pair<std::string, std::string>>& history,
    std::string& reply)>;

// The tool-call request the model made, parsed out of its reply.
struct ParsedToolCall {
    bool        present = false;
    std::string name;
    std::string arguments_json;   // the raw JSON object
    std::map<std::string, double> numbers;
    std::map<std::string, std::string> strings;
};

// Parse a model's reply for a tool call.
//
// The reply is the assistant message's CONTENT plus, possibly, a
// tool-call block. The parser handles BOTH shapes:
//   * the tool call in the Ollama-native JSON (the full response body
//     with a "message": {"tool_calls": [...]} block), and
//   * the tool call written as TEXT (a fenced block or a bare
//     {"tool":"name","arguments":{...}} object) -- which is what a small
//     model produces when the server's native tool support is not used.
//
// The second shape matters: it means the tools work even against a
// server that ignores the "tools" field, which is the case for some
// local runtimes.
ParsedToolCall ParseToolCall(const std::string& reply);

// Run the loop.
//
// `system` is the system prompt. `user_message` is the task. The tool
// catalogue is sent with every request (the callback decides how).
//
// `max_tool_calls` bounds the loop. `max_turns` bounds the total number
// of model requests (a model that answers with text but also a tool call
// would otherwise loop forever).
ToolLoopResult RunToolLoop(const std::string& system,
                           const std::string& user_message,
                           const AI::GeoToolSet& tools,
                           const ModelResponder& responder,
                           int max_tool_calls = 6,
                           int max_turns = 8);

// The catalogue the system prompt should advertise, so the model knows
// what it can call EVEN IF the server ignores the "tools" field.
std::string ToolCataloguePrompt(const AI::GeoToolSet& tools);

}  // namespace LLM
}  // namespace DeliRium
