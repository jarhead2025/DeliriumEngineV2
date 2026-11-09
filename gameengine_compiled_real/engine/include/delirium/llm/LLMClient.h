// =============================================================================
// LLMClient.h - Minimal local-LLM client (Ollama-compatible HTTP API).
//
// WHY NO libcurl: the machine has libcurl.so but NO curl headers, and adding
// a dependency for "POST a JSON body to localhost" is not worth it. This is
// a plain POSIX socket + hand-written HTTP/1.1 request, which is ~150 lines
// and has zero build-system risk.
//
// WHY A BACKGROUND THREAD: a 7B model takes seconds to answer. The render
// loop must keep running at 60 FPS while the model thinks. The request runs
// on a worker thread; the UI polls the accumulated text every frame and
// paints whatever has arrived so far (streaming).
//
// THE CONTRACT: `Send()` is non-blocking (returns immediately). `Poll()` is
// called from the render loop and returns the current state. `TakeResponse()`
// hands over the finished text exactly once.
//
// PROTOCOL: Ollama's OpenAI-compatible endpoint /v1/chat/completions with
// "stream": true returns SSE ("data: {...}\n\n", terminated by "data: [DONE]").
// We parse the JSON with the engine's OWN strict, dependency-free reader
// (ai/ElementTreeParser.h -- DeliRium::AI::ParseJson) and concatenate
// choices[0].delta.content.
//
// WHY NOT nlohmann/json: CMake declares nlohmann_json OPTIONAL, but the
// implementation sits in the engine/src/*.cpp GLOB -- so an unconditional
// include would break the WHOLE engine lib (and every CTest gate) on a
// machine without it, not just the LLM client. The in-tree reader removes
// the optional dependency from the critical path entirely.
// =============================================================================
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace DeliRium {
namespace LLM {

enum class State {
    Idle,        // nothing in flight
    Connecting,  // socket opened, request sent
    Streaming,   // receiving chunks
    Done,        // finished successfully (response ready)
    Failed       // error; `error()` explains
};

struct Config {
    std::string host = "127.0.0.1";
    int         port = 11434;
    std::string model = "qwen2.5:7b-instruct-q8_0";
    // System prompt sent with every request. The workspace overrides this to
    // teach the model the element-tree JSON schema.
    std::string system = "You are a helpful assistant.";
    float       temperature = 0.7f;
    int         max_tokens = 1024;
    int         timeout_seconds = 180;
};

class LLMClient {
public:
    explicit LLMClient(const Config& cfg = Config());
    ~LLMClient();

    void SetConfig(const Config& c) { cfg_ = c; }
    const Config& GetConfig() const { return cfg_; }

    // Start a request. Returns false if one is already in flight.
    // `history` is the conversation so far (oldest first), each entry
    // {"role": "user"|"assistant", "content": "..."}.
    bool Send(const std::string& user_message,
              const std::vector<std::pair<std::string, std::string>>& history = {});

    // Poll the worker. Safe to call every frame from the render thread.
    State CurrentState() const { return state_.load(std::memory_order_acquire); }
    bool  IsBusy() const {
        const State s = CurrentState();
        return s == State::Connecting || s == State::Streaming;
    }

    // The text received so far (grows while streaming).
    std::string Partial() const;

    // The finished response. Valid once CurrentState() == Done. Returns the
    // text and clears the Done state (the caller takes ownership).
    std::string TakeResponse();

    // The error message when CurrentState() == Failed.
    std::string Error() const;

    // Cancel an in-flight request (the worker stops at the next chunk).
    void Cancel();

    // --- server discovery -------------------------------------------------
    // GET /v1/models -> the list of model ids the local server serves.
    // Blocking; call it from a worker or at startup.
    static bool ListModels(const std::string& host, int port,
                           std::vector<std::string>* out, std::string* error);
    // GET /api/tags -> the Ollama-native model list (name + size).
    static bool ListOllamaModels(const std::string& host, int port,
                                 std::vector<std::string>* out, std::string* error);
    // A cheap reachability probe (connect + GET /v1/models).
    static bool Ping(const std::string& host, int port, int timeout_seconds = 3);

private:
    Config cfg_;
    std::atomic<State> state_{ State::Idle };
    std::thread worker_;
    mutable std::mutex mu_;
    std::string partial_;
    std::string final_;
    std::string error_;
    std::atomic<bool> cancel_{ false };
    std::atomic<bool> has_response_{ false };

    void Run(const std::string& payload);
};

// Build the JSON request body for /v1/chat/completions.
std::string BuildChatPayload(const Config& cfg,
                              const std::string& user_message,
                              const std::vector<std::pair<std::string, std::string>>& history);

// Extract the concatenated assistant text from a full (non-streaming)
// /v1/chat/completions response body.
std::string ExtractChatText(const std::string& body);

// Extract the assistant text from ONE SSE chunk body (the JSON after
// "data: "). Returns "" for the "[DONE]" sentinel.
std::string ExtractDeltaText(const std::string& chunk_json);

} // namespace LLM
} // namespace DeliRium
