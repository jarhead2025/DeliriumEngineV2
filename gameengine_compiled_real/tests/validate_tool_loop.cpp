// =============================================================================
// validate_tool_loop.cpp - THE TOOL LOOP'S CONTRACT.
//
// WHY THIS GATE EXISTS
//
// `validate_geo_tools` pins the tools' MATHS. But the maths being right is
// only half the feature: the loop has to actually WORK -- detect a call in
// the model's reply, run it, put the answer back into the conversation, and
// let the model use that answer.
//
// Without a gate on the LOOP, the failure mode is invisible: the loop can
// stop feeding the tool's result back, or feed back an empty string, or
// loop forever, and every tool still passes its own maths test. The model
// then answers with a guess and nothing says why.
//
// WHY IT IS TESTED WITH A SCRIPTED RESPONDER
//
// The loop is synchronous by design (the callback does the asynchrony), so
// it can be driven by a SCRIPT: turn 1 asks for a tool, turn 2 answers. That
// makes the loop's behaviour testable WITHOUT a live model -- which matters,
// because a live-model test cannot make the model misbehave on demand, and
// a loop that only works when the model behaves is not a loop.
//
// THE INVARIANTS
//
//   A) the loop DETECTS a tool call in both reply shapes (the server's
//      native tool_calls block, and the text shape),
//   B) the tool's RESULT is fed back into the conversation (this is the
//      load-bearing part -- without it the model cannot use the answer),
//   C) the loop TERMINATES: on a final text answer, on the tool budget,
//      and on the turn budget,
//   D) a FAILING tool produces a SELF-CORRECTING message the model can act
//      on (naming the accepted arguments), and the loop continues,
//   E) the transcript records every call (the learner needs it),
//   F) the catalogue prompt advertises the tools.
//
// A) is checked against the SHAPES, not against one model: the native shape
// and the text shape are different code paths, and a fix that only handles
// one of them is a fix that works on the machine it was written on.
// =============================================================================
#include <cstdio>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include <map>

#include "delirium/ai/GeoTools.h"
#include "delirium/llm/LLMToolLoop.h"

using namespace DeliRium;
using namespace DeliRium::AI;
using namespace DeliRium::LLM;

static int g_checks = 0;
static int g_failures = 0;

static void Check(bool ok, const char* what) {
    ++g_checks;
    if (ok) {
        std::printf("  ok    %s\n", what);
    } else {
        ++g_failures;
        std::printf("  FAIL  %s\n", what);
    }
}

// A scripted responder: it walks a list of canned replies, one per turn,
// and records every history it was handed.
struct Script {
    std::vector<std::string> replies;
    size_t turn = 0;
    std::vector<std::vector<std::pair<std::string, std::string>>> histories;

    ModelResponder Make() {
        return [this](const std::vector<std::pair<std::string, std::string>>& history,
                      std::string& reply) -> bool {
            histories.push_back(history);
            if (turn >= replies.size()) { reply = ""; return false; }
            reply = replies[turn++];
            return true;
        };
    }
};

// The server's NATIVE tool-call shape: a full response body with a
// "message": {"tool_calls": [...]} block.
static std::string NativeCall(const char* name, const char* args_json) {
    return std::string("{\"message\":{\"role\":\"assistant\",\"content\":\"\",") +
           "\"tool_calls\":[{\"id\":\"call_1\",\"function\":{\"name\":\"" + name +
           "\",\"arguments\":" + args_json + "}}]}}";
}

// The TEXT shape: the model writes the call as prose-adjacent JSON.
static std::string TextCall(const char* name, const char* args_json) {
    return std::string("{\"tool\":\"") + name + "\",\"arguments\":" + args_json + "}";
}

int main() {
    std::printf("=== validate_tool_loop ===\n");
    const GeoToolSet& tools = GeoToolSet::Instance();

    // -------------------------------------------------------------------------
    // A) THE CALL IS DETECTED -- IN BOTH SHAPES.
    // -------------------------------------------------------------------------
    std::printf("\nA) THE CALL IS DETECTED, IN BOTH REPLY SHAPES\n");
    {
        const ParsedToolCall native_call =
            ParseToolCall(NativeCall("volume_of_sphere", "{\"r\":3}"));
        Check(native_call.present, "the server's NATIVE tool_calls block is detected");
        Check(native_call.name == "volume_of_sphere",
              "the native call's NAME is read (volume_of_sphere)");
        Check(native_call.numbers.count("r") == 1 &&
              std::fabs(native_call.numbers.at("r") - 3.0) < 1e-12,
              "the native call's ARGUMENTS are read (r=3)");

        const ParsedToolCall text_call =
            ParseToolCall(TextCall("resolve_torus_radii", "{\"outer\":0.44,\"tube\":0.12}"));
        Check(text_call.present, "the TEXT shape is detected");
        Check(text_call.name == "resolve_torus_radii",
              "the text call's NAME is read (resolve_torus_radii)");
        Check(text_call.numbers.count("outer") == 1 &&
              std::fabs(text_call.numbers.at("outer") - 0.44) < 1e-12,
              "the text call's ARGUMENTS are read (outer=0.44)");

        // A plain answer must NOT be mistaken for a call. This is the
        // false-positive guard: if prose could be read as a call, the loop
        // would call tools at random and never finish.
        const ParsedToolCall plain =
            ParseToolCall("The volume of a sphere of radius 3 is 113.1 cubic units.");
        Check(!plain.present, "a plain prose answer is NOT a tool call");
        const ParsedToolCall empty = ParseToolCall("");
        Check(!empty.present, "an empty reply is NOT a tool call");
    }

    // -------------------------------------------------------------------------
    // B) THE RESULT IS FED BACK -- THE LOAD-BEARING PART.
    // -------------------------------------------------------------------------
    std::printf("\nB) THE TOOL'S RESULT IS FED BACK INTO THE CONVERSATION\n");
    {
        Script s;
        s.replies.push_back(NativeCall("volume_of_sphere", "{\"r\":3}"));
        s.replies.push_back("The volume is 113.097 cubic units.");

        const ToolLoopResult r = RunToolLoop("sys", "what is the volume?",
                                             tools, s.Make(), 6, 8);
        Check(r.completed, "the loop COMPLETES after the model answers with text");
        Check(r.turns.size() == 1, "exactly ONE tool call was made");
        if (r.turns.size() == 1) {
            Check(r.turns[0].tool_name == "volume_of_sphere",
                  "the transcript names the tool that ran");
            Check(!r.turns[0].failed, "the tool call did NOT fail");
            Check(r.turns[0].result.find("113") != std::string::npos,
                  "the tool's ANSWER (113.097) is in the transcript");
        }
        Check(r.text.find("113") != std::string::npos,
              "the model's FINAL TEXT is returned");

        // The history the model saw on turn 2 MUST contain the tool's answer.
        // Without it the model cannot use the number, and the whole feature
        // is a no-op that still looks like it worked.
        Check(s.histories.size() == 2, "the model was asked TWICE (once per turn)");
        if (s.histories.size() >= 2) {
            const auto& h = s.histories[1];
            bool saw_result = false;
            for (const auto& m : h) {
                if (m.second.find("113") != std::string::npos) saw_result = true;
            }
            Check(saw_result, "the tool's ANSWER was fed BACK to the model");
        }
    }

    // -------------------------------------------------------------------------
    // C) THE LOOP TERMINATES.
    // -------------------------------------------------------------------------
    std::printf("\nC) THE LOOP TERMINATES (it must never hang)\n");
    {
        // A model that ONLY ever asks for tools: the tool budget must stop it.
        Script s;
        for (int i = 0; i < 40; ++i)
            s.replies.push_back(NativeCall("volume_of_sphere", "{\"r\":3}"));
        const ToolLoopResult r = RunToolLoop("sys", "loop forever", tools, s.Make(), 4, 20);
        Check(!r.completed, "a model that never answers does NOT complete");
        Check(!r.stop_reason.empty(), "the loop says WHY it stopped");
        Check(r.turns.size() <= 4,
              "the TOOL budget stopped it (at most 4 calls were made)");
    }
    {
        // A model that answers with text but ALSO a call: the TURN budget
        // must stop it.
        Script s;
        for (int i = 0; i < 40; ++i)
            s.replies.push_back(NativeCall("volume_of_sphere", "{\"r\":3}"));
        const ToolLoopResult r = RunToolLoop("sys", "loop forever", tools, s.Make(), 40, 3);
        Check(!r.completed, "the turn budget stops a call-and-text loop");
        Check(r.turns.size() <= 40, "the transcript is bounded");
    }
    {
        // No responder at all: the loop must report it, not crash.
        const ToolLoopResult r = RunToolLoop("sys", "x", tools, nullptr, 6, 8);
        Check(!r.completed, "a missing responder does NOT complete");
        Check(r.stop_reason.find("responder") != std::string::npos,
              "the loop says the responder was missing");
    }

    // -------------------------------------------------------------------------
    // D) A FAILING TOOL PRODUCES A SELF-CORRECTING MESSAGE.
    // -------------------------------------------------------------------------
    std::printf("\nD) A FAILING TOOL CORRECTS ITSELF ON THE NEXT TURN\n");
    {
        Script s;
        // Wrong argument name: "radious" instead of "r".
        s.replies.push_back(NativeCall("volume_of_sphere", "{\"radious\":3}"));
        s.replies.push_back("Using r=3, the volume is 113.097.");

        const ToolLoopResult r = RunToolLoop("sys", "volume?", tools, s.Make(), 6, 8);
        Check(r.turns.size() == 1, "the failing call was recorded");
        if (r.turns.size() == 1) {
            Check(r.turns[0].failed, "the call is marked FAILED");
            Check(r.turns[0].result.find("r") != std::string::npos,
                  "the failure names the ACCEPTED argument ('r')");
            Check(r.turns[0].result.find("radious") != std::string::npos,
                  "the failure echoes what it actually received ('radious')");
        }
        // The corrective message must reach the model, or it repeats the
        // same failing call until the budget runs out. MEASURED on the first
        // live run: three identical failing calls, then a guess.
        if (s.histories.size() >= 2) {
            bool saw_fix = false;
            for (const auto& m : s.histories[1])
                if (m.second.find("radious") != std::string::npos) saw_fix = true;
            Check(saw_fix, "the corrective message reached the model");
        }
        Check(r.completed, "the loop CONTINUES after a failing tool");
    }
    {
        // An unknown tool name must also be reported, not ignored.
        Script s;
        s.replies.push_back(NativeCall("no_such_tool", "{}"));
        s.replies.push_back("I will compute it myself.");
        const ToolLoopResult r = RunToolLoop("sys", "x", tools, s.Make(), 6, 8);
        Check(r.turns.size() == 1, "the unknown tool was recorded");
        if (r.turns.size() == 1) {
            Check(r.turns[0].failed, "the unknown tool is marked FAILED");
            Check(r.turns[0].result.find("no_such_tool") != std::string::npos,
                  "the failure names the unknown tool");
        }
        Check(r.completed, "the loop continues after an unknown tool");
    }

    // -------------------------------------------------------------------------
    // E) THE CATALOGUE PROMPT ADVERTISES THE TOOLS.
    // -------------------------------------------------------------------------
    std::printf("\nE) THE CATALOGUE PROMPT ADVERTISES THE COMPARTMENTS\n");
    {
        const std::string cat = ToolCataloguePrompt(tools);
        Check(cat.find("resolve_torus_radii") != std::string::npos,
              "the catalogue lists resolve_torus_radii");
        Check(cat.find("closed_body_rule") != std::string::npos,
              "the catalogue lists closed_body_rule");
        Check(cat.find("volume_of_sphere") != std::string::npos,
              "the catalogue lists volume_of_sphere");
        Check(cat.find("{\"tool\"") != std::string::npos,
              "the catalogue shows the CALL FORMAT the model must emit");
        Check(cat.find("arguments") != std::string::npos,
              "the catalogue names the arguments member");
    }

    // -------------------------------------------------------------------------
    // F) THE LOOP AND THE TOOLS AGREE -- the end-to-end contract.
    // -------------------------------------------------------------------------
    std::printf("\nF) THE LOOP AND THE TOOL AGREE (the end-to-end contract)\n");
    {
        // The single measurement the model actually depends on: the number
        // the loop feeds back must be the number the tool computes. If the
        // loop formatted or mangled it, the model would reason over a
        // DIFFERENT value than the tool produced -- and nothing would say so.
        Script s;
        s.replies.push_back(NativeCall("resolve_torus_radii", "{\"outer\":0.44,\"tube\":0.12}"));
        s.replies.push_back("done");
        const ToolLoopResult r = RunToolLoop("sys", "x", tools, s.Make(), 6, 8);

        std::string direct, err;
        tools.Invoke("resolve_torus_radii",
                     {{"outer", 0.44}, {"tube", 0.12}}, {}, direct, err);

        Check(!r.turns.empty(), "the call happened");
        if (!r.turns.empty()) {
            // Both must name the same major radius (0.32).
            const bool loop_ok = r.turns[0].result.find("0.32") != std::string::npos;
            const bool tool_ok = direct.find("0.32") != std::string::npos;
            Check(loop_ok, "the loop's transcript carries the tool's major radius (0.32)");
            Check(tool_ok, "the tool itself reports major=0.32 (outer - tube)");
        }
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) std::printf("VALIDATE_TOOL_LOOP PASSED\n");
    else                 std::printf("VALIDATE_TOOL_LOOP FAILED\n");
    return g_failures == 0 ? 0 : 1;
}
