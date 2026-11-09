// FILEPATH: examples/doc_learner_probe/main.cpp
// Drive the DOCUMENT-driven reconstruction loop against a LIVE local model.
//
// This is the keystone probe for the "learn closed bodies from technical
// documents" idea: it takes a real technical document, asks the local
// model to turn it into an element tree, validates the tree, BAKES it
// into a real PSF body, and reports exactly which stage succeeded.
//
// It also drives the ESCALATION ladder: when the local model fails, the
// same document is retried with the failure reason and the accumulated
// error log attached -- and, if a second model is configured, against that
// larger model instead.
//
// Usage:
//   doc_learner_probe [--host H] [--port P] [--model M]
//                      [--escalate-model M2] [--docs N] [--out DIR]
//                      [--save-log PATH]
//
// Exit code: 0 if at least one document produced a real baked body.
#include "delirium/ai/DocLearner.h"
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

namespace {

// THE DOCUMENT CORPUS.
//
// Real technical descriptions, in the style of a parts list or a
// maintenance manual. Each one names parts, their relationships, and
// their relative proportions -- which is exactly the information the
// element tree needs.
const std::vector<Document> kCorpus = {
    {"A-100 gearbox housing",
     "Assembly A-100 is a gearbox housing. The main body is a cylindrical "
     "shell, 120mm in diameter and 80mm tall, with a domed top. A rotor "
     "of 40mm diameter sits concentrically inside the housing, reaching "
     "halfway up. A drive shaft of 15mm diameter passes horizontally "
     "through the lower part of the housing and protrudes on both sides.",
     "machine"},

    {"M-220 hydraulic pump",
     "Assembly M-220 is a hydraulic pump. The body is a block roughly "
     "square in section, 100mm on a side and 90mm deep, with a flat "
     "mounting flange at the bottom. A cylindrical motor, 70mm diameter "
     "and 120mm long, is mounted coaxially to one face of the block. A "
     "smaller outlet elbow, 25mm diameter, projects from the top.",
     "machine"},

    {"B-500 valve body",
     "Assembly B-500 is a globe valve body. The main casting is a squat "
     "cylinder, 90mm diameter and 60mm tall, with two flanged pipe "
     "connections, 40mm diameter, projecting horizontally on opposite "
     "sides. A bonnet rises vertically from the top, 55mm diameter and "
     "70mm tall, and is capped by a smaller cover plate.",
     "machine"},
};

// Ask a model, synchronously, through the socket client.
std::string ask_model(const LLM::Config& cfg, const std::string& system,
                       const std::string& user, double* secs_out,
                       std::string* err_out) {
    LLM::Config c = cfg;
    c.system = system;
    c.max_tokens = 1400;
    c.timeout_seconds = 300;
    LLM::LLMClient client(c);
    const auto t0 = std::chrono::steady_clock::now();
    if (!client.Send(user)) {
        if (err_out) *err_out = "could not start the request";
        return {};
    }
    int ticks = 0;
    while (client.IsBusy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (++ticks > 3200) { client.Cancel(); if (err_out) *err_out = "timed out"; return {}; }
    }
    if (client.CurrentState() == LLM::State::Failed) {
        if (err_out) *err_out = client.Error();
        return {};
    }
    const std::string reply = client.TakeResponse();
    if (reply.empty() && err_out) *err_out = "the client finished with an empty reply";
    if (secs_out)
        *secs_out = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    return reply;
}

// Count the substances (sheets) in a tree.
size_t sheet_count(const Element& root) {
    size_t n = 0;
    std::vector<const Element*> stack{&root};
    while (!stack.empty()) {
        const Element* e = stack.back();
        stack.pop_back();
        if (e->substance) ++n;
        for (const auto& c : e->children) stack.push_back(c.get());
    }
    return n;
}

} // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 11434;
    std::string model = "qwen2.5:7b-instruct-q8_0";
    std::string escalate_model;
    std::string log_path = "/tmp/doc_learner_log.txt";
    int ndocs = (int)kCorpus.size();
    bool escalate = false;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "--escalate-model") && i + 1 < argc) {
            escalate_model = argv[++i]; escalate = true;
        }
        else if (!std::strcmp(argv[i], "--docs") && i + 1 < argc) ndocs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--save-log") && i + 1 < argc) log_path = argv[++i];
        else if (!std::strcmp(argv[i], "--escalate")) escalate = true;
    }
    if (ndocs < 1) ndocs = 1;
    if (ndocs > (int)kCorpus.size()) ndocs = (int)kCorpus.size();

    std::printf("=== DOC LEARNER PROBE (live model) ===\n");
    std::printf("server: %s:%d\nmodel:  %s\n", host.c_str(), port, model.c_str());
    if (!escalate_model.empty())
        std::printf("escalate to: %s\n", escalate_model.c_str());
    std::printf("documents: %d\n\n", ndocs);

    if (!LLM::LLMClient::Ping(host, port, 5)) {
        std::fprintf(stderr, "FAIL: no LLM server at %s:%d\n", host.c_str(), port);
        return 1;
    }
    std::printf("server is up\n\n");

    LLM::Config cfg;
    cfg.host = host; cfg.port = port; cfg.model = model;

    DocErrorLog log;
    int ok_docs = 0, failed_docs = 0, baked_docs = 0;

    for (int di = 0; di < ndocs; ++di) {
        const Document& doc = kCorpus[di];
        std::printf("--- [%d/%d] %s ---\n", di + 1, ndocs, doc.id.c_str());

        double secs = 0.0;
        std::string err;
        const auto ask = [&](const std::string& system, const std::string& user) {
            return ask_model(cfg, system, user, &secs, &err);
        };

        DocResult r = InterpretDocument(doc, ask, log);
        std::printf("    attempt 1: %s (%.1fs)\n",
                    r.ok ? "OK" : ("FAILED at " + r.failed_stage).c_str(), secs);
        if (!r.ok) {
            std::printf("      reason: %s\n", r.reason.c_str());
            if (!err.empty()) std::printf("      transport: %s\n", err.c_str());
        }

        // The ESCALATION attempt: the same document, the failure reason
        // and the accumulated log, and (if configured) a LARGER model.
        if (!r.ok && escalate) {
            const bool use_larger = !escalate_model.empty();
            LLM::Config cfg2 = cfg;
            if (use_larger) cfg2.model = escalate_model;
            const auto ask2 = [&](const std::string& system, const std::string& user) {
                return ask_model(cfg2, system, user, &secs, &err);
            };
            std::function<std::string(const std::string&, const std::string&)> larger;
            if (use_larger) larger = ask2;
            std::printf("    attempt 2: escalating to %s ...\n",
                        use_larger ? escalate_model.c_str() : model.c_str());
            DocResult r2 = InterpretDocument(doc, ask2, log, /*escalate=*/true,
                                              use_larger ? &larger : nullptr);
            std::printf("    attempt 2: %s (%.1fs)\n",
                        r2.ok ? "OK" : ("FAILED at " + r2.failed_stage).c_str(), secs);
            if (!r2.ok) std::printf("      reason: %s\n", r2.reason.c_str());
            if (r2.ok) r = std::move(r2);
        }

        if (!r.ok) { ++failed_docs; std::printf("\n"); continue; }
        ++ok_docs;

        // THE GEOMETRY: the tree must BAKE into a real body. A tree that
        // parses is not a body.
        std::unique_ptr<PSFBody> body;
        std::string why;
        body = ElementSystem::BakeBody(*r.tree, 48, 1.0f, why);
        if (!body) {
            std::printf("    BAKE FAILED: %s\n", why.c_str());
            ++failed_docs; --ok_docs; std::printf("\n");
            continue;
        }
        ++baked_docs;
        std::printf("    sheets=%zu  patches=%zu  anchors=%zu\n",
                    sheet_count(*r.tree), body->Patches().size(),
                    body->AnchorsConst().Count());
        std::printf("    GEOMETRY OK\n\n");
    }

    std::printf("=== SUMMARY ===\n");
    std::printf("documents:        %d\n", ndocs);
    std::printf("valid trees:      %d\n", ok_docs);
    std::printf("baked bodies:     %d\n", baked_docs);
    std::printf("failures logged:  %zu\n", log.FailureCount());
    std::printf("successes logged: %zu\n", log.SuccessCount());
    std::printf("escalations:      %zu\n", log.EscalatedCount());
    if (log.Save(log_path))
        std::printf("error log saved:  %s\n", log_path.c_str());

    std::printf("\n%s\n", baked_docs > 0 ? "PROBE PASSED" : "PROBE FAILED");
    return baked_docs > 0 ? 0 : 1;
}
