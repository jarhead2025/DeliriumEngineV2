// =============================================================================
// ai_pipeline_probe - Proves the WHOLE AI path end-to-end, headless.
//
// The chain under test:
//
//   prompt -> LLMClient (socket, SSE streaming) -> model reply
//          -> ExtractJsonObject -> ParseElementTree (validate)
//          -> ElementSystem::BakeBody -> a REAL PSFBody
//          -> the body's field/geometry sanity checks
//
// WHY A SEPARATE PROBE: the workspace does the same thing behind a UI, but
// a UI-driven test proves only that "a button was pressed". This probe
// exercises every link and reports the body it produced, so a failure names
// the BROKEN LINK (transport / parse / bake / geometry) rather than
// "the AI button did nothing".
//
//   cmake --build build_delirium --target ai_pipeline_probe
//   ./build_delirium/ai_pipeline_probe
//   ./build_delirium/ai_pipeline_probe --prompt "a weathered stone torus"
//   ./build_delirium/ai_pipeline_probe --model qwen2.5:14b-instruct-q3_K_M
//
// Exit codes: 0 = the full chain worked, 1 = it did not.
// =============================================================================
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/llm/LLMClient.h"
#include "delirium/psf/PSFBody.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::AI;

static int fail(const char* what, const std::string& detail = "") {
    std::fprintf(stderr, "ai_pipeline_probe: FAIL -- %s%s%s\n", what,
                 detail.empty() ? "" : ": ", detail.c_str());
    return 1;
}

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 11434;
    std::string model = "qwen2.5:7b-instruct-q8_0";
    std::string prompt = "a weathered stone torus with cracks";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
    }

    // --- 1. transport ------------------------------------------------------
    std::printf("[1/5] transport: pinging %s:%d ...\n", host.c_str(), port);
    if (!LLM::LLMClient::Ping(host, port, 5))
        return fail("no LLM server", host + ":" + std::to_string(port));
    std::printf("      server is up\n");

    LLM::Config cfg;
    cfg.host = host; cfg.port = port; cfg.model = model;
    cfg.system = ElementTreeSystemPrompt();
    cfg.max_tokens = 900;
    cfg.timeout_seconds = 300;

    LLM::LLMClient client(cfg);
    std::printf("[2/5] transport: asking the model ...\n");
    std::printf("      prompt: %s\n", prompt.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    if (!client.Send(prompt)) return fail("could not start the request");

    int ticks = 0;
    while (client.IsBusy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (++ticks > 3200) {          // 320 s
            client.Cancel();
            return fail("model timed out");
        }
    }
    if (client.CurrentState() == LLM::State::Failed)
        return fail("model request failed", client.Error());
    const std::string reply = client.TakeResponse();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    if (reply.empty()) return fail("model returned an empty reply");
    std::printf("      reply in %.1fs (%zu bytes)\n", secs, reply.size());

    // --- 3. extract + parse ------------------------------------------------
    std::printf("[3/5] parse: extracting the JSON object ...\n");
    std::string json = reply;
    JsonValue probe;
    std::string why;
    if (!ParseJson(json, probe, why)) {
        std::string extracted;
        if (ExtractJsonObject(reply, extracted)) {
            json = extracted;
            std::printf("      (extracted from prose/fences)\n");
        } else {
            return fail("no JSON object in the reply", reply.substr(0, 300));
        }
    }

    std::unique_ptr<Element> tree;
    if (!ParseElementTree(json, tree, why)) {
        // Dump the JSON we actually tried to parse. Without it, a parse
        // failure is a dead end: the caller cannot tell a schema violation
        // from a malformed NUMBER in the payload.
        std::fprintf(stderr, "--- extracted json (%zu bytes) ---\n%s\n--- end ---\n",
                     json.size(), json.c_str());
        return fail("element tree rejected", why);
    }
    std::printf("      %s\n", DescribeTree(*tree).c_str());

    // Print the SUBSTANCES the model actually chose. Without this, "the
    // body came out flat" has two possible causes -- the model chose a flat
    // substance, or the baker could not resolve the one it chose -- and the
    // reader cannot tell them apart.
    {
        struct Walker {
            static void Visit(const Element& el, int depth) {
                std::printf("      %*s- %-16s", depth * 2, "", el.kind.c_str());
                const char* sub = el.SubstanceName();
                if (sub[0]) std::printf(" substance=%s", sub);
                std::printf("\n");
                for (const auto& c : el.children) Visit(*c, depth + 1);
            }
        };
        std::printf("      tree:\n");
        Walker::Visit(*tree, 1);
    }

    // --- 4. bake -----------------------------------------------------------
    std::printf("[4/5] bake: ElementSystem::BakeBody ...\n");
    // N=48, not 12: the baker reconstructs each patch as a tangent plane, so
    // a feature near the sample spacing gets chorded. At N=12 a torus's ring
    // flattens into a disc (the body kept 19% of the field's relief); at
    // N=48 it keeps 67%. The bake costs ~8 ms.
    auto body = ElementSystem::BakeBody(*tree, 48, 1.0f, why);
    if (!body) return fail("bake failed", why);
    const size_t patches = body->Patches().size();
    const size_t anchors = body->AnchorBlob().size();
    std::printf("      baked: %zu patches, %zu anchors\n", patches, anchors);
    if (patches == 0) return fail("baked body has no patches");
    if (anchors == 0) return fail("baked body has no anchors");

    // --- 5. geometry sanity ------------------------------------------------
    //
    // A body that bakes but is DEGENERATE is still a failure: the model
    // asked for a shape and the engine produced a flat card or a point.
    // The checks are invariants (non-degenerate extent, finite bounds),
    // not snapshots of a particular model output.
    std::printf("[5/5] geometry: sanity checks ...\n");
    const Vec3 mn = body->BBoxMin(), sp = body->BBoxSpan();
    std::printf("      bbox min (%.3f %.3f %.3f) span (%.3f %.3f %.3f)\n",
                mn.x, mn.y, mn.z, sp.x, sp.y, sp.z);
    const bool finite = std::isfinite(mn.x) && std::isfinite(mn.y) && std::isfinite(mn.z)
                     && std::isfinite(sp.x) && std::isfinite(sp.y) && std::isfinite(sp.z);
    if (!finite) return fail("bbox is not finite");
    if (sp.x <= 0.f || sp.y <= 0.f || sp.z <= 0.f)
        return fail("body is degenerate (zero extent on some axis)");

    std::printf("\nai_pipeline_probe: OK -- prompt -> model -> tree -> body\n");
    return 0;
}
