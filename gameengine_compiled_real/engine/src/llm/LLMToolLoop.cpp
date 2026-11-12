// =============================================================================
// LLMToolLoop.cpp - the tool-calling loop's implementation.
//
// THE PARSER HANDLES TWO SHAPES, and that matters.
//
// SHAPE 1: the server's native tool support. Ollama's /api/chat returns
//          {"message":{"role":"assistant","content":"",
//                     "tool_calls":[{"function":{"name":"...",
//                                 "arguments":{...}}}]}}
//          and the arguments arrive as a JSON OBJECT.
//
// SHAPE 2: the tool call written as TEXT. A small model asked to "use the
//          tool" will often emit, in its content:
//
//              {"tool":"volume_of_sphere","arguments":{"r":3}}
//
//          or a fenced block containing that.
//
// SHAPE 2 EXISTS BECAUSE NOT EVERY LOCAL RUNTIME HONOURS THE "tools"
// FIELD. The tools must work against whatever server the user happens to
// have, so the loop parses BOTH. That also means the loop degrades
// gracefully: a model that ignores the tools entirely simply answers in
// text, and the caller gets that text.
//
// THE ARGUMENTS ARE FLATTENED INTO TWO MAPS
//
// The engine's JSON reader produces a JsonValue tree; the tools want a
// flat "numbers" map and a flat "strings" map. The flattening walks the
// tree: a number becomes a number, a string becomes a string, and
// anything else (a nested object, an array) is kept as its JSON TEXT in
// the strings map -- so a tool can still see it.
//
// NOTE ON THE JSON WRITER: the engine's reader has no writer, so a small
// one is included here (WriteJson). It only has to handle the shapes the
// parser produces, and it only has to be correct, not fast.
// =============================================================================
#include "delirium/llm/LLMToolLoop.h"
#include "delirium/ai/ElementTreeParser.h"

#include <cstdio>
#include <sstream>
#include <algorithm>

namespace DeliRium {
namespace LLM {

using AI::JsonValue;

namespace {

// A small JSON writer -- the reader has no counterpart.
void WriteJson(const JsonValue& v, std::ostringstream& o) {
    switch (v.type) {
        case JsonValue::Type::Null:   o << "null"; break;
        case JsonValue::Type::Bool:   o << (v.boolean ? "true" : "false"); break;
        case JsonValue::Type::Number: {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", v.number);
            o << buf;
            break;
        }
        case JsonValue::Type::String: {
            o << '"';
            for (char ch : v.text) {
                switch (ch) {
                    case '"':  o << "\\\""; break;
                    case '\\': o << "\\\\"; break;
                    case '\n': o << "\\n"; break;
                    case '\r': o << "\\r"; break;
                    case '\t': o << "\\t"; break;
                    default:
                        if ((unsigned char)ch < 0x20) {
                            char buf[8];
                            std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                            o << buf;
                        } else {
                            o << ch;
                        }
                }
            }
            o << '"';
            break;
        }
        case JsonValue::Type::Array: {
            o << '[';
            for (size_t i = 0; i < v.items.size(); ++i) {
                if (i) o << ',';
                WriteJson(v.items[i], o);
            }
            o << ']';
            break;
        }
        case JsonValue::Type::Object: {
            o << '{';
            for (size_t i = 0; i < v.members.size(); ++i) {
                if (i) o << ',';
                o << '"' << v.members[i].first << "\":";
                WriteJson(v.members[i].second, o);
            }
            o << '}';
            break;
        }
    }
}

// Flatten an arguments object into the two maps.
void FlattenArgs(const JsonValue& args,
                 std::map<std::string, double>& numbers,
                 std::map<std::string, std::string>& strings) {
    if (!args.IsObject()) return;
    for (const auto& kv : args.members) {
        const std::string& k = kv.first;
        const JsonValue& v = kv.second;
        if (v.IsNumber()) {
            numbers[k] = v.number;
        } else if (v.IsString()) {
            strings[k] = v.text;
        } else if (v.IsArray() || v.IsObject()) {
            std::ostringstream o;
            WriteJson(v, o);
            strings[k] = o.str();
        }
    }
}

// Find the tool-call block inside a parsed reply.
bool FindToolCallInJson(const JsonValue& root, ParsedToolCall& out) {
    std::vector<const JsonValue*> stack{ &root };
    while (!stack.empty()) {
        const JsonValue* v = stack.back();
        stack.pop_back();
        if (v->IsObject()) {
            for (const auto& kv : v->members) {
                if (kv.first == "tool_calls" && kv.second.IsArray()) {
                    for (const auto& tc : kv.second.items) {
                        if (!tc.IsObject()) continue;
                        const JsonValue* fn = tc.Find("function");
                        if (!fn || !fn->IsObject()) continue;
                        const JsonValue* name_v = fn->Find("name");
                        if (!name_v || !name_v->IsString()) continue;
                        out.present = true;
                        out.name = name_v->text;
                        const JsonValue* args_v = fn->Find("arguments");
                        if (args_v && args_v->IsObject()) {
                            FlattenArgs(*args_v, out.numbers, out.strings);
                            std::ostringstream o; WriteJson(*args_v, o);
                            out.arguments_json = o.str();
                        } else if (args_v && args_v->IsString()) {
                            out.arguments_json = args_v->text;
                            std::string why;
                            JsonValue parsed;
                            if (AI::ParseJson(args_v->text, parsed, why))
                                FlattenArgs(parsed, out.numbers, out.strings);
                        } else {
                            out.arguments_json = "{}";
                        }
                        return true;
                    }
                } else if (kv.second.IsObject() || kv.second.IsArray()) {
                    stack.push_back(&kv.second);
                }
            }
        } else if (v->IsArray()) {
            for (const auto& e : v->items)
                if (e.IsObject() || e.IsArray()) stack.push_back(&e);
        }
    }
    return false;
}

// Find a tool call written as TEXT: a JSON object with a "tool" (or
// "name" or "function") key and an "arguments" (or "parameters" or
// "args") key.
//
// The text is scanned for the first '{' and its matching '}' -- with a
// depth count so a nested object does not end it early -- and each
// candidate is parsed.
bool FindToolCallInText(const std::string& text, ParsedToolCall& out) {
    size_t start = text.find('{');
    while (start != std::string::npos) {
        int depth = 0;
        bool in_str = false, esc = false;
        size_t end = std::string::npos;
        for (size_t i = start; i < text.size(); ++i) {
            const char ch = text[i];
            if (in_str) {
                if (esc) esc = false;
                else if (ch == '\\') esc = true;
                else if (ch == '"') in_str = false;
                continue;
            }
            if (ch == '"') { in_str = true; continue; }
            if (ch == '{') ++depth;
            else if (ch == '}') {
                --depth;
                if (depth == 0) { end = i; break; }
            }
        }
        if (end != std::string::npos) {
            const std::string cand = text.substr(start, end - start + 1);
            std::string why;
            JsonValue jv;
            if (AI::ParseJson(cand, jv, why) && jv.IsObject()) {
                std::string name;
                const JsonValue* args_v = nullptr;
                std::string args_text;
                for (const auto& kv : jv.members) {
                    if ((kv.first == "tool" || kv.first == "name" || kv.first == "function")
                        && kv.second.IsString()) {
                        name = kv.second.text;
                    } else if (kv.first == "arguments" || kv.first == "parameters"
                               || kv.first == "args") {
                        if (kv.second.IsObject()) {
                            args_v = &kv.second;
                        } else if (kv.second.IsString()) {
                            args_text = kv.second.text;
                        }
                    }
                }
                if (!name.empty()) {
                    out.present = true;
                    out.name = name;
                    if (args_v) {
                        FlattenArgs(*args_v, out.numbers, out.strings);
                        std::ostringstream o; WriteJson(*args_v, o);
                        out.arguments_json = o.str();
                    } else if (!args_text.empty()) {
                        out.arguments_json = args_text;
                        std::string w2;
                        JsonValue p2;
                        if (AI::ParseJson(args_text, p2, w2))
                            FlattenArgs(p2, out.numbers, out.strings);
                    } else {
                        out.arguments_json = "{}";
                    }
                    return true;
                }
            }
        }
        start = text.find('{', start + 1);
    }
    return false;
}

}  // namespace

ParsedToolCall ParseToolCall(const std::string& reply) {
    ParsedToolCall out;
    if (reply.empty()) return out;

    // SHAPE 1: the whole reply is JSON with a tool_calls block.
    {
        std::string why;
        JsonValue jv;
        if (AI::ParseJson(reply, jv, why)) {
            if (FindToolCallInJson(jv, out)) return out;
            if (FindToolCallInText(reply, out)) return out;
        }
    }
    // SHAPE 2: the tool call is inside the text.
    FindToolCallInText(reply, out);
    return out;
}

std::string ToolCataloguePrompt(const AI::GeoToolSet& tools) {
    std::ostringstream o;
    o << "\nYou may call GEOMETRIC ANALYSIS TOOLS to do the arithmetic you "
         "cannot do reliably. Each tool is a compartment: it answers ONE "
         "question with ONE number.\n\n"
         "To call a tool, emit exactly this and nothing else:\n"
         "  {\"tool\":\"<tool_name>\",\"arguments\":{...}}\n\n"
         "The tools are:\n"
      << tools.Catalogue()
      << "\nAfter a tool answers, use its number in your reasoning and then "
         "give your final answer. Do not guess an arithmetic result -- call "
         "the tool.\n";
    return o.str();
}

ToolLoopResult RunToolLoop(const std::string& system,
                           const std::string& user_message,
                           const AI::GeoToolSet& tools,
                           const ModelResponder& responder,
                           int max_tool_calls,
                           int max_turns) {
    ToolLoopResult result;
    if (!responder) {
        result.stop_reason = "no responder";
        return result;
    }

    std::vector<std::pair<std::string, std::string>> history;
    int tool_calls = 0;
    int turns = 0;
    std::string task = user_message;

    for (; turns < max_turns; ++turns) {
        std::string reply;
        if (!responder(history, reply)) {
            result.stop_reason = "the model request failed";
            result.text = reply;
            return result;
        }
        const ParsedToolCall call = ParseToolCall(reply);
        if (!call.present) {
            result.text = reply;
            result.completed = true;
            return result;
        }
        if (tool_calls >= max_tool_calls) {
            result.stop_reason = "the tool-call budget was exhausted";
            result.text = reply;
            return result;
        }
        ++tool_calls;

        // RUN THE TOOL.
        ToolTurn turn;
        turn.tool_name = call.name;
        turn.arguments_json = call.arguments_json;
        std::string out, err;
        if (!tools.Invoke(call.name, call.numbers, call.strings, out, err)) {
            turn.failed = true;
            // -----------------------------------------------------------------------------
            // THE CORRECTIVE MESSAGE GOES BACK TO THE MODEL.
            // -----------------------------------------------------------------------------
            //
            // `out` is the registry's SELF-CORRECTING message: it names the
            // accepted argument names and echoes what was actually received.
            // That is what lets a small model fix its own call on the next
            // turn.
            //
            // A generic "the tool failed" teaches the model nothing, and the
            // model then repeats the same failing call until the budget runs
            // out. MEASURED on the first live run: three identical failing
            // calls, then a guess.
            //
            // The registry's own message is preferred; the generic prefix is
            // only a fallback for a tool that is not in the registry at all.
            if (out.empty()) {
                turn.result = "the tool failed: " + err;
            } else {
                turn.result = out;
            }
            out = turn.result;
        } else {
            turn.result = out;
        }
        result.turns.push_back(turn);

        // THE RESULT GOES BACK INTO THE CONVERSATION, as a tool message.
        history.push_back({ "user", task });
        history.push_back({ "assistant", reply });
        history.push_back({ "tool", "[" + call.name + " " + call.arguments_json
                                    + "] -> " + out });
        // The next request continues from the tool's answer.
        task = "Continue. Use the tool's answer above.";
    }

    result.stop_reason = "the turn budget was exhausted";
    return result;
}

}  // namespace LLM
}  // namespace DeliRium
