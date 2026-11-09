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
#include <fstream>
#include <iterator>
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
    // The document prompt is LARGE: the system prompt alone is ~2,900
    // tokens (11.6 KB), and the document and the worked examples add more.
    // 1,400 completion tokens is the measured requirement for a
    // schema-shaped JSON answer.
    c.max_tokens = 1400;
    // The timeout must exceed the MEASURED worst case by a wide margin.
    //
    // A measured successful run took 276.4 s on a loaded machine -- and
    // the old 300 s socket timeout sat only 8% above it. Under any
    // additional load the request crossed the timeout and was reported as
    // a TRANSPORT failure, which looks exactly like "the model is
    // unreachable" and sends you looking at the wrong code entirely.
    //
    // 900 s is ~3x the measured worst case. The request that would be
    // killed by it has already failed for other reasons.
    c.timeout_seconds = 900;
    LLM::LLMClient client(c);
    const auto t0 = std::chrono::steady_clock::now();
    if (!client.Send(user)) {
        if (err_out) *err_out = "could not start the request";
        return {};
    }
    // The outer bound is DERIVED from the client's timeout rather than
    // hard-coded.
    //
    // WHY: the two were 320 s and 300 s -- independent constants that must
    // agree. When they disagree, the outer loop can give up while the
    // client is still legitimately working, and the request is cancelled
    // mid-flight and reported as a transport failure.
    //
    // (The first attempt at this derivation was WRONG in a way that
    // mattered: `timeout_seconds * 13 / 10` on a 900 s timeout gives 1170
    // ticks, which at 100 ms per tick is **117 s** -- SHORTER than the
    // socket timeout, not longer. The measured model time is ~230 s, so
    // every request was cancelled by the outer loop at 117 s.)
    //
    // The bound is 30% ABOVE the socket timeout, so the CLIENT is always
    // the thing that decides the request is over. The arithmetic is in
    // SECONDS, then converted to ticks.
    const int max_ticks = (c.timeout_seconds * 13 / 10) * 10;   // 100 ms per tick
    int ticks = 0;
    // Report the elapsed time on EVERY path, including the failures.
    //
    // WHY: the old code set `secs_out` only on the success path, so every
    // failure reported "0.0s". That number was actively misleading: the
    // requests were dying at exactly 117 s (the outer bound), and "0.0s"
    // made it look like they had failed instantly. The elapsed time is the
    // single most diagnostic number in a timeout investigation, and it was
    // the one number being thrown away.
    auto elapsed = [&]() -> double {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    };
    while (client.IsBusy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (++ticks > max_ticks) {
            client.Cancel();
            if (err_out) *err_out = "timed out";
            if (secs_out) *secs_out = elapsed();
            return {};
        }
    }
    if (client.CurrentState() == LLM::State::Failed) {
        if (err_out) *err_out = client.Error();
        if (secs_out) *secs_out = elapsed();
        return {};
    }
    const std::string reply = client.TakeResponse();
    if (reply.empty() && err_out) *err_out = "the client finished with an empty reply";
    if (secs_out) *secs_out = elapsed();
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
    // --file PATH: interpret a DOCUMENT FROM DISK instead of the built-in
    // corpus.
    //
    // WHY: the built-in corpus is fixed, so it cannot be used to test a
    // specific document -- and testing the escalation ladder needs a
    // document the local model is known to struggle with. A file makes the
    // probe's input a parameter instead of a constant.
    //
    // It can be given MORE THAN ONCE: each --file adds a document, and
    // they are processed in order. That is how the CORPUS accumulates
    // across documents in one run -- the second document benefits from
    // the first.
    std::vector<std::string> files;
    std::string domain = "machine";

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
        else if (!std::strcmp(argv[i], "--file") && i + 1 < argc) files.push_back(argv[++i]);
        else if (!std::strcmp(argv[i], "--domain") && i + 1 < argc) domain = argv[++i];
    }
    if (ndocs < 1) ndocs = 1;
    if (ndocs > (int)kCorpus.size()) ndocs = (int)kCorpus.size();

    // The documents to process: the built-in corpus, or the files.
    std::vector<Document> docs;
    if (!files.empty()) {
        for (const std::string& path : files) {
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "FAIL: cannot read '%s'\n", path.c_str());
                return 1;
            }
            std::string text((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
            if (text.empty()) {
                std::fprintf(stderr, "FAIL: '%s' is empty\n", path.c_str());
                return 1;
            }
            // The id is the file's stem.
            std::string id = path;
            const size_t slash = id.find_last_of('/');
            if (slash != std::string::npos) id = id.substr(slash + 1);
            docs.push_back(Document{id, text, domain});
        }
    } else {
        docs.assign(kCorpus.begin(), kCorpus.begin() + ndocs);
    }
    ndocs = (int)docs.size();

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

    // The CORPUS is LOADED before the documents are interpreted, so the
    // FIRST document in this run already benefits from every document
    // interpreted in a PREVIOUS run.
    //
    // This is the whole point of the corpus: a corpus that starts empty on
    // every run is not a corpus, it is a within-run cache. The load is
    // best-effort -- a missing log is the cold-start case, not an error.
    DocErrorLog log;
    if (log.Load(log_path)) {
        std::printf("corpus: loaded %zu entries (%zu ok, %zu failed) from '%s'\n",
                    log.Size(), log.SuccessCount(), log.FailureCount(),
                    log_path.c_str());
    } else {
        std::printf("corpus: no corpus at '%s' (cold start)\n", log_path.c_str());
    }
    int ok_docs = 0, failed_docs = 0, baked_docs = 0;

    for (int di = 0; di < ndocs; ++di) {
        const Document& doc = docs[di];
        std::printf("--- [%d/%d] %s ---\n", di + 1, ndocs, doc.id.c_str());

        // What the CORPUS supplies for THIS document. It is printed because
        // a learning system whose learning is invisible cannot be
        // distinguished from one that has not learned -- and because the
        // whole point of the corpus is that this line changes as the corpus
        // grows.
        const auto retrieved = log.RetrieveSimilar(doc.text, doc.domain, 3);
        if (retrieved.empty()) {
            std::printf("    corpus: no similar worked example (cold for this doc)\n");
        } else {
            std::printf("    corpus: %zu worked example(s), best '%s' (%.0f%%)\n",
                        retrieved.size(), retrieved[0].document_id.c_str(),
                        retrieved[0].similarity * 100.0);
        }

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
            // The learner's reason is GENERIC ("the model returned nothing"),
            // which hides WHY. The client's own error is the specific one --
            // "no response headers", "the response timed out while reading a
            // chunk", "the server closed the connection mid-response" -- and
            // the difference between them is the difference between a
            // configuration problem and a network one.
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
