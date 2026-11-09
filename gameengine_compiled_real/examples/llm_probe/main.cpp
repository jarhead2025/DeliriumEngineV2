// =============================================================================
// llm_probe - Proves the local LLM client works end-to-end WITHOUT the UI.
//
// This is the "does the socket code actually talk to the server" check. A
// unit test of BuildChatPayload() proves nothing; what matters is that a
// real request to a real local model comes back with real text.
//
//   cmake --build build_delirium --target llm_probe
//   ./build_delirium/llm_probe                 (from the project root)
//   ./build_delirium/llm_probe --host H --port P --model M --prompt "..."
//   ./build_delirium/llm_probe --list          (list the server's models)
//
// Exit codes: 0 = success, 1 = failure.
// =============================================================================
#include "delirium/llm/LLMClient.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace DeliRium::LLM;

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 11434;
    std::string model = "qwen2.5:7b-instruct-q8_0";
    std::string prompt = "Reply with exactly: DELIRIUM_OK";
    bool list = false;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--list")) list = true;
    }

    // --- discovery ---
    std::printf("llm_probe: probing %s:%d ...\n", host.c_str(), port);
    if (!LLMClient::Ping(host, port, 5)) {
        std::fprintf(stderr, "llm_probe: FAIL -- no LLM server at %s:%d\n",
                     host.c_str(), port);
        return 1;
    }
    std::printf("llm_probe: server is up\n");

    std::vector<std::string> models;
    std::string err;
    if (LLMClient::ListModels(host, port, &models, &err)) {
        std::printf("llm_probe: /v1/models -> %zu model(s):\n", models.size());
        for (const auto& m : models) std::printf("  - %s\n", m.c_str());
    } else {
        std::printf("llm_probe: /v1/models failed: %s\n", err.c_str());
    }
    if (list) return 0;

    // --- a real chat round-trip, streaming ---
    Config cfg;
    cfg.host = host; cfg.port = port; cfg.model = model;
    cfg.system = "You are a test harness. Answer with the requested text only.";
    cfg.max_tokens = 64;
    cfg.timeout_seconds = 120;

    LLMClient client(cfg);
    std::printf("llm_probe: sending to model '%s' ...\n", model.c_str());
    if (!client.Send(prompt)) {
        std::fprintf(stderr, "llm_probe: FAIL -- could not start the request\n");
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    int last_len = 0;
    int ticks = 0;
    while (client.IsBusy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const std::string p = client.Partial();
        if (int(p.size()) != last_len) {
            std::printf("  [%5.1fs] %zu bytes: %s\n",
                        std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(),
                        p.size(), p.substr(0, 120).c_str());
            last_len = int(p.size());
        }
        if (++ticks > 2400) {   // 120 s
            std::fprintf(stderr, "llm_probe: FAIL -- timeout\n");
            client.Cancel();
            return 1;
        }
    }

    if (client.CurrentState() == State::Failed) {
        std::fprintf(stderr, "llm_probe: FAIL -- %s\n", client.Error().c_str());
        return 1;
    }
    const std::string answer = client.TakeResponse();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::printf("llm_probe: response in %.2fs (%zu bytes):\n---\n%s\n---\n",
                secs, answer.size(), answer.c_str());
    if (answer.empty()) {
        std::fprintf(stderr, "llm_probe: FAIL -- empty response\n");
        return 1;
    }
    std::printf("llm_probe: OK\n");
    return 0;
}
